from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def make_next_quarter_growth_dataset(
    sales_panel: pd.DataFrame,
    group_keys: Iterable[str] = ("region_id", "sector_code"),
    target_col: str = "sales_growth_qoq",
) -> pd.DataFrame:
    """
    X_t -> y_{t+1} 구조의 학습용 데이터셋을 만든다.

    - 그룹 단위(기본: 상권 x 업종)로 정렬(year, quarter)
    - target_col(t+1)을 y로 사용
    - 마지막 시점(다음 분기 타깃이 없는 행)은 제거
    """
    df = sales_panel.copy()
    sort_keys = list(group_keys) + ["year", "quarter"]
    df = df.sort_values(sort_keys)

    grp = df.groupby(list(group_keys), sort=False)
    df["target_next"] = grp[target_col].shift(-1)

    # 타깃이 없는 마지막 분기는 제거
    ds = df.dropna(subset=["target_next"]).copy()

    return ds


def suggest_growth_outlier_bounds(
    growth_series: pd.Series,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> Tuple[float, float]:
    """
    성장률 이상치 컷의 기본 범위를 제안한다.

    - 기본적으로 1%~99% 분위수를 기준으로 사용.
    """
    s = growth_series.replace([np.inf, -np.inf], np.nan).dropna()
    lower = float(s.quantile(lower_q))
    upper = float(s.quantile(upper_q))
    return lower, upper


def temporal_train_test_split(
    df: pd.DataFrame,
    test_start_year: int,
    test_start_quarter: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    연-분기 기준으로 train/test를 나눈다.

    - test_start_year, test_start_quarter 이상을 test로 사용.
    """
    cond_test = (df["year"] > test_start_year) | (
        (df["year"] == test_start_year) & (df["quarter"] >= test_start_quarter)
    )
    test = df.loc[cond_test].copy()
    train = df.loc[~cond_test].copy()
    return train, test

