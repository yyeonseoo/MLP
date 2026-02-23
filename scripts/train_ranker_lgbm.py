"""
LightGBM Ranker 학습 — 추천 서비스용 Learning-to-Rank.

- group = (region_id, year_quarter): 상권·분기별로 업종을 줄 세움
- 타깃: target_log_sales_next (다음 분기 매출 수준)
- 평가: NDCG@20, Precision@20, HitRate@20

선행: scripts/build_rank_dataset.py
출력: data/processed/ranker_lgbm.txt (모델), 평가 지표 출력
"""

from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import pandas as pd

from src.models.rank_dataset import (
    RANK_GROUP_KEY,
    RANK_TARGET_COL,
    get_rank_feature_columns,
)
from src.models.rank_metrics import evaluate_rank_groups


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = DATA_PROCESSED / "ranker_lgbm.txt"


def main() -> None:
    train_path = DATA_PROCESSED / "rank_dataset_train.csv"
    test_path = DATA_PROCESSED / "rank_dataset_test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "rank_dataset_train/test.csv 없음. 먼저 PYTHONPATH=. python scripts/build_rank_dataset.py 실행."
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    feature_cols = get_rank_feature_columns(train)
    if not feature_cols:
        raise ValueError("랭킹 피처가 없습니다. build_rank_dataset에서 피처 컬럼 확인.")

    # 그룹 순서 유지 (동일 그룹이 연속되도록 정렬)
    train = train.sort_values([RANK_GROUP_KEY]).reset_index(drop=True)
    test = test.sort_values([RANK_GROUP_KEY]).reset_index(drop=True)

    group_sizes_train = train.groupby(RANK_GROUP_KEY, sort=False).size()
    groups_train = group_sizes_train.values.tolist()

    X_train = train[feature_cols].fillna(0).values
    y_train = train[RANK_TARGET_COL].values
    X_test = test[feature_cols].fillna(0).values
    y_test = test[RANK_TARGET_COL].values

    try:
        from lightgbm import LGBMRanker
    except (ImportError, OSError) as e:
        if "libomp" in str(e) or "LoadLibrary" in str(e):
            raise RuntimeError(
                "LightGBM 로드 실패 (Mac에서는 OpenMP 필요). "
                "해결: brew install libomp 후 재실행."
            ) from e
        raise ImportError("lightgbm 필요: pip install lightgbm") from e

    ranker = LGBMRanker(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=64,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[20],
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    ranker.fit(X_train, y_train, group=groups_train)

    # 예측 및 평가
    test = test.copy()
    test["pred_rank"] = ranker.predict(X_test)
    metrics = evaluate_rank_groups(
        test,
        group_key=RANK_GROUP_KEY,
        actual_col=RANK_TARGET_COL,
        pred_col="pred_rank",
        k=20,
    )

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    ranker.booster_.save_model(str(MODEL_PATH))
    with open(DATA_PROCESSED / "ranker_feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False)

    print("\n=== LightGBM Ranker (추천 서비스용) ===")
    print(f"NDCG@20     : {metrics['ndcg@20']:.4f}")
    print(f"Precision@20: {metrics['precision@20']:.4f}")
    print(f"HitRate@20   : {metrics['hit_rate@20']:.4f}")
    print(f"모델 저장: {MODEL_PATH}")


if __name__ == "__main__":
    main()
