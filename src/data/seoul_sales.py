from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from src.config.lipstick_config import add_lipstick_flag_by_name


@dataclass
class SalesConfig:
    """서울시 상권분석서비스(추정매출-상권) 공통 설정."""

    # pandas read_csv 기본 옵션
    encoding: str = "cp949"  # 공공데이터포털 기본 인코딩
    quarter_col: str = "기준_년분기_코드"
    region_code_col: str = "상권_코드"
    region_name_col: str = "상권_코드_명"
    sector_code_col: str = "서비스_업종_코드"
    sector_name_col: str = "서비스_업종_코드_명"
    sales_col: str = "당월_매출_금액"
    txn_col: str = "당월_매출_건수"


def _parse_quarter_code(code: str | int) -> tuple[int, int, str]:
    """
    '20241' -> (2024, 1, '2024Q1') 형태로 변환.
    """
    s = str(code).strip()
    if len(s) != 5 or not s.isdigit():
        raise ValueError(f"Unexpected quarter code format: {code!r}")
    year = int(s[:4])
    q = int(s[4])
    if q not in (1, 2, 3, 4):
        raise ValueError(f"Unexpected quarter number in code: {code!r}")
    return year, q, f"{year}Q{q}"


def load_seoul_sales(
    paths: Sequence[str | Path],
    config: SalesConfig | None = None,
) -> pd.DataFrame:
    """
    서울시 상권분석서비스(추정매출-상권) CSV 여러 개를 로드해서 하나의 DataFrame으로 합친다.

    Parameters
    ----------
    paths:
        2020~2024년 CSV 파일들의 경로 리스트.
        예) data/raw/seoul_sales/서울시_상권분석서비스(추정매출-상권)_2020년.csv
    config:
        기본 컬럼명을 담은 설정. 필요 시 사용자 정의 가능.
    """
    if config is None:
        config = SalesConfig()

    frames: List[pd.DataFrame] = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(p)
        df = pd.read_csv(p, encoding=config.encoding)
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)

    # 기준 년분기 파싱
    quarters = raw[config.quarter_col].map(_parse_quarter_code)
    raw["year"] = quarters.map(lambda t: t[0])
    raw["quarter"] = quarters.map(lambda t: t[1])
    raw["year_quarter"] = quarters.map(lambda t: t[2])

    # 핵심 컬럼만 추출
    core = raw[
        [
            config.quarter_col,
            "year",
            "quarter",
            "year_quarter",
            config.region_code_col,
            config.region_name_col,
            config.sector_code_col,
            config.sector_name_col,
            config.sales_col,
            config.txn_col,
        ]
    ].rename(
        columns={
            config.quarter_col: "quarter_code",
            config.region_code_col: "region_id",
            config.region_name_col: "region_name",
            config.sector_code_col: "sector_code",
            config.sector_name_col: "sector_name",
            config.sales_col: "sales",
            config.txn_col: "transactions",
        }
    )

    # 립스틱 업종 플래그
    core["is_lipstick"] = add_lipstick_flag_by_name(core["sector_name"])

    return core


def add_growth_features(
    df: pd.DataFrame,
    group_keys: Iterable[str] = ("region_id", "sector_code"),
    min_sales_prev: float = 100_000.0,
    require_contiguous_quarter: bool = True,
) -> pd.DataFrame:
    """
    전분기 대비 매출/건수 성장률을 추가한다.

    - 반드시 (year, quarter) 숫자 기준 정렬. year_quarter 문자열 정렬 시 오류 가능.
    - sales_prev가 0 또는 min_sales_prev 미만이면 성장률 NaN.
    - require_contiguous_quarter=True: 직전 분기가 실제로 “직전 분기”인 경우만 성장률 유효.
      (분기 누락 시 shift가 “몇 분기 전”을 가리켜 폭발하는 것 방지)
    """
    sort_keys = list(group_keys) + ["year", "quarter"]
    df_sorted = df.sort_values(sort_keys).copy()

    group = df_sorted.groupby(list(group_keys), sort=False)

    # 분기 연속성: t = year*4 + (quarter-1), 직전 행이 t-1이어야 유효
    df_sorted["_period"] = df_sorted["year"] * 4 + (df_sorted["quarter"] - 1)
    df_sorted["_period_prev"] = group["_period"].shift(1)

    df_sorted["sales_prev"] = group["sales"].shift(1)
    df_sorted["transactions_prev"] = group["transactions"].shift(1)

    # 성장률 계산
    df_sorted["sales_growth_qoq"] = df_sorted["sales"] / df_sorted["sales_prev"] - 1.0
    df_sorted["txn_growth_qoq"] = (
        df_sorted["transactions"] / df_sorted["transactions_prev"] - 1.0
    )
    df_sorted["sales_growth_log"] = (
        np.log1p(df_sorted["sales"]) - np.log1p(df_sorted["sales_prev"].fillna(0))
    )
    df_sorted.loc[df_sorted["sales_prev"].isna(), "sales_growth_log"] = np.nan

    # 누락 구간: 직전 행이 직전 분기가 아니면 성장률 무효화 (분모 왜곡 방지)
    if require_contiguous_quarter:
        gap = df_sorted["_period"] - df_sorted["_period_prev"]
        not_contiguous = gap.isna() | (gap != 1)
        df_sorted.loc[not_contiguous, "sales_growth_qoq"] = np.nan
        df_sorted.loc[not_contiguous, "sales_growth_log"] = np.nan
        df_sorted.loc[not_contiguous, "txn_growth_qoq"] = np.nan

    # 분모가 0이거나 너무 작으면 성장률 무효화
    bad_prev = (
        df_sorted["sales_prev"].isna()
        | (df_sorted["sales_prev"] <= 0)
        | (df_sorted["sales_prev"] < min_sales_prev)
    )
    df_sorted.loc[bad_prev, "sales_growth_qoq"] = np.nan
    df_sorted.loc[bad_prev, "sales_growth_log"] = np.nan

    df_sorted.drop(columns=["_period", "_period_prev"], inplace=True)
    return df_sorted


def compute_lipstick_share_by_region_quarter(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    상권 x 분기 단위로 립스틱 업종 매출 비중(립스틱 지수의 기본)을 계산한다.

    반환 컬럼
    --------
    - region_id, region_name, year, quarter, year_quarter
    - sales_total
    - sales_lipstick
    - sales_non_lipstick
    - lipstick_share : sales_lipstick / sales_total
    """
    key_cols = ["region_id", "region_name", "year", "quarter", "year_quarter"]

    g = df.groupby(key_cols, as_index=False)
    agg = g.agg(
        sales_total=("sales", "sum"),
        sales_lipstick=("sales", lambda s: float(s[df.loc[s.index, "is_lipstick"]].sum())),
    )
    agg["sales_non_lipstick"] = agg["sales_total"] - agg["sales_lipstick"]
    agg["lipstick_share"] = np.where(
        agg["sales_total"] > 0,
        agg["sales_lipstick"] / agg["sales_total"],
        np.nan,
    )
    return agg


def compute_lipstick_index_relative_growth(
    lipstick_share: pd.DataFrame,
) -> pd.DataFrame:
    """
    립스틱 지수(전분기 대비 립스틱 비중의 상대적 변화)를 계산한다.

    정의 예시:
    - lipstick_share_t / lipstick_share_(t-1) - 1
    - region_id 단위로 계산
    """
    df = lipstick_share.sort_values(["region_id", "year", "quarter"]).copy()
    grp = df.groupby("region_id", sort=False)

    df["lipstick_share_prev"] = grp["lipstick_share"].shift(1)
    df["lipstick_index_rel"] = df["lipstick_share"] / df["lipstick_share_prev"] - 1.0

    return df

