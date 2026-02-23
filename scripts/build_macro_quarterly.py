"""
거시지표 분기 CSV 생성.

data/raw/소비자심리지수(CCSI).xlsx, data/raw/소비자물가지수_20260223183741.xlsx 를 읽어
분기별 macro_quarterly.csv 를 생성한다.
기준금리·실업률은 두 파일에 없으므로 근사 시계열(한국 은행/통계청 추이 반영) 사용.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# 엑셀 파일 경로 (data/raw 직접)
CCSI_PATH = DATA_RAW / "소비자심리지수(CCSI).xlsx"
CPI_PATH = DATA_RAW / "소비자물가지수_20260223183741.xlsx"


def _load_ccsi_monthly(path: Path) -> pd.DataFrame:
    """CCSI xlsx: Sheet1, 컬럼 0=기간(YYYY-MM), 1=값. 7행부터 데이터."""
    df = pd.read_excel(path, sheet_name="Sheet1", engine="openpyxl", header=None)
    df = df.iloc[7:].rename(columns={0: "period", 1: "ccsi"})
    df["ccsi"] = pd.to_numeric(df["ccsi"], errors="coerce")
    df = df.dropna(subset=["ccsi"])
    df["period"] = df["period"].astype(str).str.strip()
    # YYYY-MM -> year, month
    parts = df["period"].str.split("-", n=1, expand=True)
    df["year"] = pd.to_numeric(parts[0], errors="coerce")
    df["month"] = pd.to_numeric(parts[1], errors="coerce")
    df = df.dropna(subset=["year", "month"]).astype({"year": int, "month": int})
    return df[["year", "month", "ccsi"]]


def _ccsi_monthly_to_quarterly(monthly: pd.DataFrame) -> pd.DataFrame:
    """월별 CCSI -> 분기별 평균."""
    monthly = monthly.copy()
    monthly["quarter"] = ((monthly["month"] - 1) // 3) + 1
    q = monthly.groupby(["year", "quarter"], as_index=False)["ccsi"].mean()
    return q.rename(columns={"ccsi": "ccsi"})


def _load_cpi_annual(path: Path) -> pd.DataFrame:
    """CPI xlsx: 시트 '데이터', 연도별 블록에서 '연간 (2023=100)' 행의 총지수(col 2) 추출."""
    df = pd.read_excel(path, sheet_name="데이터", engine="openpyxl", header=None)
    # col0=시점(연도), col1=항목, col2=총지수. 연도 행 다음 2행이 '연간 (2023=100)'
    rows = []
    last_year = None
    for i in range(len(df)):
        r = df.iloc[i]
        try:
            y = int(float(r[0]))
            if 2016 <= y <= 2030:
                last_year = y
        except (TypeError, ValueError):
            pass
        if r[1] == "연간 (2023=100)" and last_year is not None:
            try:
                cpi = float(r[2])
                rows.append({"year": last_year, "cpi": cpi})
            except (TypeError, ValueError):
                pass
    if not rows:
        return pd.DataFrame(columns=["year", "cpi"])
    return pd.DataFrame(rows)


def _cpi_annual_to_quarterly(annual: pd.DataFrame) -> pd.DataFrame:
    """연도별 CPI -> 분기별 (같은 연도는 동일 값)."""
    if annual.empty:
        return pd.DataFrame(columns=["year", "quarter", "cpi"])
    rows = []
    for _, r in annual.iterrows():
        for q in range(1, 5):
            rows.append({"year": int(r["year"]), "quarter": q, "cpi": r["cpi"]})
    return pd.DataFrame(rows)


def _policy_rate_unemployment_quarterly(years: list[int]) -> pd.DataFrame:
    """기준금리·실업률 근사치 (한국 2020~2024 추이). 두 엑셀에 없으므로 고정 시계열."""
    # 기준금리(%): 2020 낮음 -> 2022~2023 인상 -> 2024 소폭 완화
    policy = {
        2020: [0.5, 0.5, 0.5, 0.5],
        2021: [0.5, 0.75, 1.0, 1.0],
        2022: [1.25, 1.75, 2.5, 3.0],
        2023: [3.25, 3.5, 3.5, 3.5],
        2024: [3.5, 3.25, 3.0, 2.75],
    }
    # 실업률(%): 대략 3.5~4.5 구간
    unemp = {
        2020: [4.0, 4.2, 4.1, 3.8],
        2021: [3.9, 3.8, 3.6, 3.4],
        2022: [3.6, 3.4, 3.0, 2.9],
        2023: [2.9, 2.7, 2.6, 2.6],
        2024: [2.6, 2.5, 2.6, 2.7],
    }
    rows = []
    for y in years:
        for q in range(1, 5):
            rows.append({
                "year": y,
                "quarter": q,
                "policy_rate": policy.get(y, [2.5] * 4)[q - 1],
                "unemployment": unemp.get(y, [3.0] * 4)[q - 1],
            })
    return pd.DataFrame(rows)


def build_macro_from_excel(
    ccsi_path: Path | None = None,
    cpi_path: Path | None = None,
) -> pd.DataFrame:
    """
    CCSI·CPI 엑셀에서 분기별 거시지표 DataFrame 생성.
    기준금리·실업률은 근사 시계열로 채움.
    """
    ccsi_path = ccsi_path or CCSI_PATH
    cpi_path = cpi_path or CPI_PATH

    # CCSI: 월별 -> 분기 평균
    if ccsi_path.exists():
        monthly = _load_ccsi_monthly(ccsi_path)
        ccsi_q = _ccsi_monthly_to_quarterly(monthly)
    else:
        ccsi_q = pd.DataFrame(columns=["year", "quarter", "ccsi"])

    # CPI: 연도별 -> 분기(동일값). 2024 없으면 2023 값으로 채움
    if cpi_path.exists():
        annual = _load_cpi_annual(cpi_path)
        if not annual.empty and annual["year"].max() < 2024:
            last = annual.loc[annual["year"].idxmax()]
            annual = pd.concat([annual, pd.DataFrame([{"year": 2024, "cpi": last["cpi"]}])], ignore_index=True)
        cpi_q = _cpi_annual_to_quarterly(annual)
    else:
        cpi_q = pd.DataFrame(columns=["year", "quarter", "cpi"])

    # 공통 분기 범위 (2020Q1 ~ 2024Q4)
    years = list(range(2020, 2025))
    policy_unemp = _policy_rate_unemployment_quarterly(years)

    # merge: year, quarter 기준
    base = pd.DataFrame([
        {"year": y, "quarter": q}
        for y in years
        for q in range(1, 5)
    ])
    out = base.merge(policy_unemp, on=["year", "quarter"], how="left")
    if not ccsi_q.empty:
        out = out.merge(ccsi_q, on=["year", "quarter"], how="left")
    else:
        out["ccsi"] = None
    if not cpi_q.empty:
        out = out.merge(cpi_q, on=["year", "quarter"], how="left")
    else:
        out["cpi"] = None

    # 컬럼 순서
    cols = ["year", "quarter", "cpi", "policy_rate", "unemployment", "ccsi"]
    out = out[[c for c in cols if c in out.columns]]
    out = out.sort_values(["year", "quarter"]).reset_index(drop=True)
    return out


def _make_placeholder_macro() -> pd.DataFrame:
    """엑셀 없을 때: 2020Q1~2024Q4 placeholder."""
    rows = []
    base_cpi = 100.0
    for year in range(2020, 2025):
        for q in range(1, 5):
            ccsi = 95.0 + (year - 2020) * 2 + q * 0.5
            if year <= 2021:
                ccsi -= 5
            cpi = base_cpi * (1.02) ** ((year - 2020) * 4 + q)
            policy_rate = 1.5 + (year - 2020) * 0.25 + (q - 1) * 0.05
            rows.append({
                "year": year,
                "quarter": q,
                "cpi": round(cpi, 2),
                "policy_rate": round(policy_rate, 2),
                "unemployment": 4.0 - (year - 2020) * 0.1,
                "ccsi": round(ccsi, 1),
            })
    return pd.DataFrame(rows)


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / "macro_quarterly.csv"

    if CCSI_PATH.exists() or CPI_PATH.exists():
        df = build_macro_from_excel()
        if df.empty or (df["ccsi"].isna().all() and df["cpi"].isna().all()):
            print("엑셀에서 유효한 데이터를 읽지 못함. placeholder 사용.")
            df = _make_placeholder_macro()
        else:
            print("엑셀 기반 거시지표 생성:", CCSI_PATH.name, CPI_PATH.name)
    else:
        print("엑셀 파일 없음. placeholder 사용.")
        df = _make_placeholder_macro()

    df.to_csv(out_path, index=False)
    print(f"저장: {out_path} (rows={len(df)})")


if __name__ == "__main__":
    main()
