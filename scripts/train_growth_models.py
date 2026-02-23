from __future__ import annotations

"""
분기별 매출 성장률(t+1) 예측 모델 학습 및 평가 스크립트.

사용 데이터
-----------
- data/processed/growth_dataset_train.csv
- data/processed/growth_dataset_test.csv

출력
-----
- 모델별 RMSE / MAE / Top-20 Recall@year_quarter 콘솔 출력
"""

from pathlib import Path

import pandas as pd

from src.models.growth_models import FeatureConfig, train_baseline_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    train_path = DATA_PROCESSED / "growth_dataset_train.csv"
    test_path = DATA_PROCESSED / "growth_dataset_test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"학습용 데이터셋이 없습니다. 먼저 scripts/eda_lipstick_effect.py 를 실행하세요: {train_path}, {test_path}"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_cfg = FeatureConfig()

    results = train_baseline_models(train_df, test_df, feature_cfg)

    print("\n=== 성장률 예측 베이스라인 성능 ===")
    for name, metrics in results.items():
        print(f"\n[{name}]")
        print(f"RMSE        : {metrics['rmse']:.4f}")
        print(f"MAE         : {metrics['mae']:.4f}")
        print(f"Top-20 Recall: {metrics['top20_recall']:.4f}")


if __name__ == "__main__":
    main()

