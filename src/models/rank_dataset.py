"""
추천 서비스용 Learning-to-Rank 데이터셋.

- 타깃: 다음 분기 매출 수준 log(sales_{t+1}+1)
- 그룹: (region_id, year_quarter) — 상권·분기별로 업종을 줄 세우는 문제
- 피처: lag/trend, 상권·업종 정체성(encoding), 거시 보조
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


RANK_TARGET_COL: str = "target_log_sales_next"
RANK_GROUP_KEY: str = "rank_group"  # region_id + "_" + year_quarter


@dataclass
class RankDatasetConfig:
    """랭킹 데이터셋 설정."""
    target_col: str = RANK_TARGET_COL
    min_sales: float = 1_000_000.0  # 현재 분기 매출 필터 (100만원 이상만 후보)
    rolling_window: int = 4


def add_rank_features(
    df: pd.DataFrame,
    group_keys: Tuple[str, ...] = ("region_id", "sector_code"),
) -> pd.DataFrame:
    """
    Lag/trend 피처 추가. 반드시 (region_id, sector_code, year, quarter) 정렬 후 호출.
    """
    sort_keys = list(group_keys) + ["year", "quarter"]
    out = df.sort_values(sort_keys).copy()
    grp = out.groupby(list(group_keys), sort=False)

    out["log_sales"] = np.log1p(out["sales"].fillna(0))
    out["log_sales_prev"] = np.log1p(out["sales_prev"].fillna(0))
    out["rolling_mean_log_sales_4q"] = grp["log_sales"].transform(
        lambda s: s.rolling(4, min_periods=1).mean()
    )
    out["rolling_std_log_sales_4q"] = grp["log_sales"].transform(
        lambda s: s.rolling(4, min_periods=1).std().fillna(0)
    )
    out["txn_growth_qoq"] = out["txn_growth_qoq"].fillna(0)
    return out


def add_target_encoding(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    encode_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    train에서 컬럼별 target 평균을 구해 train/test에 붙임. (누수 방지)
    """
    train = train.copy()
    test = test.copy()
    for col in encode_cols:
        if col not in train.columns:
            continue
        means = train.groupby(col)[target_col].transform("mean")
        train[f"{col}_te"] = means
        # test: train의 그룹별 평균으로 매핑, 없는 그룹은 전역 평균
        global_mean = train[target_col].mean()
        train_agg = train.groupby(col)[target_col].agg("mean").to_dict()
        test[f"{col}_te"] = test[col].map(train_agg).fillna(global_mean)
    return train, test


def build_rank_dataset(
    panel: pd.DataFrame,
    cfg: Optional[RankDatasetConfig] = None,
    min_sales: Optional[float] = None,
) -> pd.DataFrame:
    """
    랭킹 학습용 DataFrame 생성.

    - target_log_sales_next 있는 행만 유지
    - 현재 분기 sales >= min_sales 인 행만 (노이즈·무의미 후보 제거)
    - rank_group = region_id + "_" + year_quarter (LightGBM group용)
    - lag/trend 피처 추가
    """
    if cfg is None:
        cfg = RankDatasetConfig()
    thr = min_sales if min_sales is not None else cfg.min_sales

    if cfg.target_col not in panel.columns:
        raise ValueError(f"패널에 {cfg.target_col} 컬럼이 없습니다. add_growth_features 후 사용하세요.")

    df = panel.copy()
    df = df.dropna(subset=[cfg.target_col])
    df = df.loc[df["sales"].fillna(0) >= thr].copy()

    if "log_sales" not in df.columns:
        df = add_rank_features(df)

    df[RANK_GROUP_KEY] = df["region_id"].astype(str) + "_" + df["year_quarter"].astype(str)
    return df


def get_rank_feature_columns(
    df: pd.DataFrame,
    exclude: Optional[List[str]] = None,
) -> List[str]:
    """랭커에 넣을 수치형 피처 목록 (id/타깃/원본 sales 제외)."""
    exclude = exclude or []
    drop = {
        "region_id", "sector_code", "sector_name", "region_name",
        "year", "quarter", "year_quarter", RANK_GROUP_KEY,
        "sales", "sales_prev", "transactions", "transactions_prev",
        "sales_growth_qoq", "sales_growth_log", RANK_TARGET_COL,
        "quarter_code",
    } | set(exclude)
    numeric = [
        c for c in df.columns
        if str(df[c].dtype) in ("int16", "int32", "int64", "float16", "float32", "float64")
        and c not in drop
    ]
    return numeric
