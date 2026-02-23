"""
인플레 충격에 대한 업종별 회복력(립스틱 충격 점수) 계산.

기준기 : 2021Q3 ~ 2022Q2 (CCSI 등 상대적으로 안정 구간)
충격기 : 2022Q3 ~ 2023Q1 (CCSI 하위 20% 구간 = 인플레 충격기)

이 스크립트가 계산하는 것:
- 업종별 기준기 매출 합계 (baseline_sales)
- 업종별 충격기 매출 합계 (shock_sales)
- 증가율 shock_rate = (shock_sales - baseline_sales) / baseline_sales
- 전체 평균 대비 상대 점수 lipstick_score = shock_rate - mean(shock_rate)
- Z-score 표준화 → 최종 립스틱(인플레 충격) 점수 (z_score)

Z-score 상위 = 충격기에도 매출이 상대적으로 잘 유지/성장한 업종(회복력 높음).
Z-score 하위 = 충격기에 매출이 상대적으로 많이 줄어든 업종.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 프로젝트 루트
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.seoul_sales import SalesConfig, load_seoul_sales

# -----------------------------
# 설정: 기준기 / 충격기
# -----------------------------
# 기준_년분기_코드 5자리: YYYYQ (20213 = 2021년 3분기)
BASELINE_QUARTER_CODES = [20213, 20214, 20221, 20222]  # 2021Q3 ~ 2022Q2
SHOCK_QUARTER_CODES = [20223, 20224, 20231]  # 2022Q3 ~ 2023Q1

# (year, quarter)로도 사용
BASELINE_QUARTERS = [(2021, 3), (2021, 4), (2022, 1), (2022, 2)]
SHOCK_QUARTERS = [(2022, 3), (2022, 4), (2023, 1)]


def _ensure_quarter_code(df: pd.DataFrame, quarter_col: str = "기준_년분기_코드") -> pd.DataFrame:
    """year + quarter → 기준_년분기_코드 (정수 20213 형태) 생성. 없으면 채움."""
    if "year" in df.columns and "quarter" in df.columns:
        if quarter_col not in df.columns:
            df[quarter_col] = df["year"].astype(int) * 10 + df["quarter"].astype(int)
    return df


def load_sales_from_raw(data_dir: Path | None = None) -> pd.DataFrame:
    """서울시 추정매출 CSV에서 (기준_년분기_코드, 서비스_업종_코드_명, 당월_매출_금액) 로드."""
    if data_dir is None:
        data_dir = ROOT / "data" / "raw"
    config = SalesConfig()
    pattern = "*(추정매출-상권)*.csv"
    files = sorted(data_dir.glob(pattern))
    if not files:
        # 연도별 파일명 (공백 포함 2024 버전 포함)
        for name in [
            "서울시_상권분석서비스(추정매출-상권)_2020년.csv",
            "서울시_상권분석서비스(추정매출-상권)_2021년.csv",
            "서울시_상권분석서비스(추정매출-상권)_2022년.csv",
            "서울시_상권분석서비스(추정매출-상권)_2023년.csv",
            "서울시_상권분석서비스(추정매출-상권)_2024년.csv",
            "서울시 상권분석서비스(추정매출-상권)_2024년.csv",
        ]:
            p = data_dir / name
            if p.exists():
                files.append(p)
    if not files:
        raise FileNotFoundError(f"추정매출 CSV를 찾을 수 없습니다: {data_dir}")
    sales = load_seoul_sales(files, config)
    return sales


def compute_sector_inflation_shock_score(sales: pd.DataFrame) -> pd.DataFrame:
    """
    업종별 기준기/충격기 매출 합계 → 증가율 → 상대 점수 → Z-score.
    load_seoul_sales 출력(sector_name, sales, quarter_code) 또는
    원본 컬럼(서비스_업종_코드_명, 당월_매출_금액, 기준_년분기_코드) 모두 지원.
    """
    if "sector_name" in sales.columns and "sales" in sales.columns:
        sector_col = "sector_name"
        sales_col = "sales"
        if "quarter_code" in sales.columns:
            quarter_col = "quarter_code"
        else:
            sales["quarter_code"] = sales["year"].astype(int) * 10 + sales["quarter"].astype(int)
            quarter_col = "quarter_code"
    else:
        quarter_col = "기준_년분기_코드"
        sector_col = "서비스_업종_코드_명"
        sales_col = "당월_매출_금액"
        if quarter_col not in sales.columns:
            sales = _ensure_quarter_code(sales, quarter_col)

    baseline = sales[sales[quarter_col].isin(BASELINE_QUARTER_CODES)]
    shock = sales[sales[quarter_col].isin(SHOCK_QUARTER_CODES)]

    base_sum = baseline.groupby(sector_col)[sales_col].sum()
    shock_sum = shock.groupby(sector_col)[sales_col].sum()

    result = pd.DataFrame({
        "baseline_sales": base_sum,
        "shock_sales": shock_sum,
    }).dropna()

    result = result[result["baseline_sales"] > 0]

    # (1) 단순 증가율
    result["shock_rate"] = (
        result["shock_sales"] - result["baseline_sales"]
    ) / result["baseline_sales"]

    # (2) 전체 평균 대비 상대 점수
    mean_rate = result["shock_rate"].mean()
    result["lipstick_score"] = result["shock_rate"] - mean_rate

    # (3) Z-score 표준화 (최종 립스틱 충격 점수)
    std = result["lipstick_score"].std()
    if std and std > 0:
        result["z_score"] = (result["lipstick_score"] - result["lipstick_score"].mean()) / std
    else:
        result["z_score"] = 0.0

    result = result.sort_values("z_score", ascending=False)
    return result


def main() -> None:
    data_dir = ROOT / "data" / "raw"
    print("매출 데이터 로드 중...")
    sales = load_sales_from_raw(data_dir)
    print("업종별 인플레 충격 점수 계산 (기준기 2021Q3~2022Q2, 충격기 2022Q3~2023Q1)...")
    result = compute_sector_inflation_shock_score(sales)

    print("\n상위 10개 업종 (인플레 충격기에서 상대적으로 강한 업종)")
    print(result.head(10)[["shock_rate", "z_score"]].to_string())

    print("\n하위 10개 업종 (인플레 충격기에서 상대적으로 약한 업종)")
    print(result.tail(10)[["shock_rate", "z_score"]].to_string())

    # 저장 (선택)
    out_path = ROOT / "outputs" / "sector_inflation_shock_score.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, encoding="utf-8-sig")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
