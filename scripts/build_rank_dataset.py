"""
추천 서비스용 랭킹 데이터셋 생성.

- 타깃: log(sales_{t+1}+1)
- 그룹: (region_id, year_quarter) — 상권·분기별 업종 랭킹
- 최소 매출 필터, lag/trend 피처, region/sector target encoding

선행: scripts/build_panel_and_regression.py (sales_panel.csv)
출력: data/processed/rank_dataset_train.csv, rank_dataset_test.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.datasets import temporal_train_test_split
from src.models.rank_dataset import (
    RANK_GROUP_KEY,
    RANK_TARGET_COL,
    add_rank_features,
    add_target_encoding,
    build_rank_dataset,
    get_rank_feature_columns,
    RankDatasetConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    panel_path = DATA_PROCESSED / "sales_panel.csv"
    if not panel_path.exists():
        raise FileNotFoundError(
            f"sales_panel.csv 없음. 먼저 PYTHONPATH=. python scripts/build_panel_and_regression.py 실행: {panel_path}"
        )

    panel = pd.read_csv(panel_path)
    # target_log_sales_next / sales_prev 없으면 여기서 생성 (패널이 오래된 경우)
    if "sales" in panel.columns:
        panel = panel.sort_values(["region_id", "sector_code", "year", "quarter"])
        g = panel.groupby(["region_id", "sector_code"], sort=False)
        if RANK_TARGET_COL not in panel.columns:
            panel[RANK_TARGET_COL] = np.log1p(g["sales"].shift(-1))
            panel = panel.dropna(subset=[RANK_TARGET_COL])
        if "sales_prev" not in panel.columns:
            panel["sales_prev"] = g["sales"].shift(1)
        if "txn_growth_qoq" not in panel.columns and "transactions" in panel.columns:
            t_prev = g["transactions"].shift(1)
            panel["txn_growth_qoq"] = panel["transactions"] / t_prev - 1.0
            panel.loc[t_prev.isna() | (t_prev <= 0), "txn_growth_qoq"] = np.nan
    cfg = RankDatasetConfig(min_sales=1_000_000.0)

    # 랭킹용 행만 추출 + lag/trend 피처
    rank_df = build_rank_dataset(panel, cfg=cfg)
    if rank_df.empty:
        raise ValueError("랭킹 데이터셋이 비었습니다. target_log_sales_next 및 min_sales 조건 확인.")

    # 시점 분할 (2023Q1부터 test)
    train, test = temporal_train_test_split(rank_df, test_start_year=2023, test_start_quarter=1)

    # 상권·업종 정체성: target encoding (train 통계로 test 매핑)
    train, test = add_target_encoding(
        train, test,
        target_col=RANK_TARGET_COL,
        encode_cols=["region_id", "sector_code"],
    )

    feature_cols = get_rank_feature_columns(train)
    print(f"랭킹 피처 수: {len(feature_cols)} — {feature_cols[:10]} ...")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    train.to_csv(DATA_PROCESSED / "rank_dataset_train.csv", index=False)
    test.to_csv(DATA_PROCESSED / "rank_dataset_test.csv", index=False)

    n_groups_train = train[RANK_GROUP_KEY].nunique()
    n_groups_test = test[RANK_GROUP_KEY].nunique()
    print(f"저장: rank_dataset_train.csv (행 {len(train)}, 그룹 {n_groups_train})")
    print(f"저장: rank_dataset_test.csv (행 {len(test)}, 그룹 {n_groups_test})")


if __name__ == "__main__":
    main()
