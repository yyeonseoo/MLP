"""
거시경제 충격지수: 매출 데이터 기반 연속형 지수.

과정:
1. 분기별 매출 합계 (전체)
2. 코로나 전후 구간 구분 (baseline 분기 설정)
3. 업종별 분기 매출 → 업종별 증감률 (QoQ)
4. 전체 대비 상대 증감률 (업종 증감률 − 전체 증감률)
5. 분기별로 요약 후 Z-score 표준화 → 충격지수
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


# 코로나 이전 기준 분기 (데이터 있으면 2019Q4, 없으면 None으로 첫 분기 사용)
COVID_PRE_BASELINE: Tuple[int, int] = (2019, 4)
COVID_START: Tuple[int, int] = (2020, 1)
COVID_END: Tuple[int, int] = (2022, 1)


def _ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["year", "quarter"]).reset_index(drop=True)


def compute_shock_index_from_sales(
    sales: pd.DataFrame,
    *,
    covid_pre_baseline: Optional[Tuple[int, int]] = COVID_PRE_BASELINE,
    covid_start: Tuple[int, int] = COVID_START,
    covid_end: Tuple[int, int] = COVID_END,
    min_quarters: int = 2,
) -> pd.DataFrame:
    """
    매출 패널(region_id, sector_code, year, quarter, sales)에서
    분기별 거시경제 충격지수(연속값)를 계산한다.

    Returns
    -------
    DataFrame with columns:
      - year, quarter, year_quarter
      - total_sales : 분기별 매출 합계
      - total_growth_qoq : 전체 QoQ 증감률
      - shock_index_raw : 분기별 요약 (업종 상대 증감률 기반, 높을수록 충격 큼)
      - shock_index_z : Z-score 표준화된 충격지수
      - shock_index_01 : 0~1 스케일 (min-max 정규화, 선택적 해석용)
    """
    need = ["year", "quarter", "sales"]
    for c in need:
        if c not in sales.columns:
            raise ValueError(f"sales에 컬럼 필요: {need}")

    sales = sales.copy()
    sales["year"] = sales["year"].astype(int)
    sales["quarter"] = sales["quarter"].astype(int)
    sales["sales"] = pd.to_numeric(sales["sales"], errors="coerce").fillna(0)

    # 1) 분기별 매출 합계
    total = (
        sales.groupby(["year", "quarter"], as_index=False)["sales"]
        .sum()
        .rename(columns={"sales": "total_sales"})
    )
    total = _ensure_sorted(total)
    total["total_growth_qoq"] = total["total_sales"].pct_change()

    # 2) 업종별 분기 매출
    sector_col = "sector_code" if "sector_code" in sales.columns else "sector_name"
    if sector_col not in sales.columns:
        raise ValueError("sales에 sector_code 또는 sector_name 필요")
    by_sector = (
        sales.groupby([sector_col, "year", "quarter"], as_index=False)["sales"]
        .sum()
    )
    by_sector = by_sector.sort_values([sector_col, "year", "quarter"]).reset_index(drop=True)

    # 3) 업종별 증감률 (QoQ)
    by_sector["sector_sales_prev"] = by_sector.groupby(sector_col, sort=False)["sales"].shift(1)
    by_sector["sector_growth_qoq"] = (
        by_sector["sales"] / by_sector["sector_sales_prev"].replace(0, np.nan) - 1.0
    )

    # 4) 전체 대비 상대 증감률 (분기·업종마다)
    by_sector = by_sector.merge(
        total[["year", "quarter", "total_growth_qoq"]],
        on=["year", "quarter"],
        how="left",
    )
    by_sector["relative_growth"] = (
        by_sector["sector_growth_qoq"].fillna(0) - by_sector["total_growth_qoq"].fillna(0)
    )

    # 5) 분기별 요약: 상대 증감률의 부호 반대 평균 (전체보다 못한 업종이 많으면 값이 커짐 → 충격 큼)
    quarter_summary = (
        by_sector.groupby(["year", "quarter"], as_index=False)["relative_growth"]
        .agg(shock_index_raw=lambda x: -float(x.mean()))
    )
    quarter_summary = quarter_summary.merge(
        total[["year", "quarter", "total_sales", "total_growth_qoq"]],
        on=["year", "quarter"],
        how="left",
    )
    quarter_summary = _ensure_sorted(quarter_summary)

    # 6) Z-score 표준화
    raw = quarter_summary["shock_index_raw"].replace([np.inf, -np.inf], np.nan)
    mu, sigma = raw.mean(), raw.std()
    if sigma is None or sigma == 0:
        quarter_summary["shock_index_z"] = 0.0
    else:
        quarter_summary["shock_index_z"] = (quarter_summary["shock_index_raw"] - mu) / sigma

    # 0~1 스케일 (선택)
    z = quarter_summary["shock_index_z"]
    z_min, z_max = z.min(), z.max()
    if z_max > z_min:
        quarter_summary["shock_index_01"] = (z - z_min) / (z_max - z_min)
    else:
        quarter_summary["shock_index_01"] = 0.5

    quarter_summary["year_quarter"] = (
        quarter_summary["year"].astype(str) + "Q" + quarter_summary["quarter"].astype(str)
    )
    return quarter_summary


def merge_shock_index_into_macro(
    macro: pd.DataFrame,
    shock_quarterly: pd.DataFrame,
    *,
    binary_quantile: Optional[float] = 0.75,
) -> pd.DataFrame:
    """
    macro 테이블에 매출 기반 충격지수를 병합한다.
    shock_index_z를 shock_score로 쓰고, binary_quantile 이상을 macro_shock=1로 둔다.
    """
    macro = macro.copy()
    cols = ["year", "quarter", "shock_index_raw", "shock_index_z", "shock_index_01"]
    cols = [c for c in cols if c in shock_quarterly.columns]
    shock_merge = shock_quarterly[cols].drop_duplicates(["year", "quarter"])

    macro = macro.merge(shock_merge, on=["year", "quarter"], how="left")
    z_col = "shock_index_z"
    if z_col in macro.columns:
        macro["shock_score"] = macro[z_col].fillna(0)
        if binary_quantile is not None:
            q = macro[z_col].quantile(binary_quantile)
            macro["macro_shock"] = (macro[z_col].fillna(-np.inf) >= q).astype(int)
    return macro
