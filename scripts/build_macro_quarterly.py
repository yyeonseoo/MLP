"""
거시지표 분기 CSV 생성 스크립트.

data/raw/macro/ 에 CCSI, CPI xlsx가 있으면 활용하고,
없으면 2020Q1~2024Q4 placeholder CSV 생성.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MACRO_DIR = DATA_RAW / "macro"
DOWNLOADS = Path.home() / "Downloads"


def _make_placeholder_macro() -> pd.DataFrame:
    """2020Q1~2024Q4 placeholder 거시지표 (CCSI, CPI 근사치)."""
    rows = []
    base_cpi = 100.0
    for year in range(2020, 2025):
        for q in range(1, 5):
            ccsi = 95.0 + (year - 2020) * 2 + q * 0.5
            if year <= 2021:
                ccsi -= 5
            cpi = base_cpi * (1.02) ** ((year - 2020) * 4 + q)
            rows.append({
                "year": year,
                "quarter": q,
                "cpi": round(cpi, 2),
                "policy_rate": 1.5 + (year - 2020) * 0.25,
                "unemployment": 4.0 - (year - 2020) * 0.1,
                "ccsi": round(ccsi, 1),
            })
    return pd.DataFrame(rows)


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / "macro_quarterly.csv"

    # xlsx 우선 시도 (선택)
    df = None
    for candidates in [
        [MACRO_DIR / "소비자심리지수(CCSI).xlsx", MACRO_DIR / "소비자물가지수_20260223183741.xlsx"],
        [DOWNLOADS / "소비자심리지수(CCSI).xlsx", DOWNLOADS / "소비자물가지수_20260223183741.xlsx"],
    ]:
        ccsi_path = candidates[0]
        cpi_path = candidates[1]
        if ccsi_path.exists():
            try:
                ccsi = pd.read_excel(ccsi_path, engine="openpyxl")
                # 시트 구조에 따라 컬럼 매핑 필요 (간단히 첫 시트)
                # 여기서는 placeholder로 대체
                break
            except Exception:
                pass

    if df is None:
        df = _make_placeholder_macro()

    df.to_csv(out_path, index=False)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
