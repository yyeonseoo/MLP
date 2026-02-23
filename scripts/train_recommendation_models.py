"""
추천용 모델: 다음 분기 로그매출(target_log_sales_next) 예측 → 랭킹 기준.

성장률 타깃 대신 로그매출 사용 시:
- 분모 문제 없음, 분포 안정
- 예측 vs 실제 산점도 정상화
- Top-20 Recall (예측 로그매출 기준 랭킹) 개선 기대

사용 데이터
-----------
- data/processed/log_sales_dataset_train.csv
- data/processed/log_sales_dataset_test.csv
(eda_lipstick_effect.py 실행 시 자동 생성)

출력
-----
- 모델별 RMSE / MAE / Top-20 Recall@year_quarter (실제·예측 모두 로그매출 기준)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.growth_models import FeatureConfig, train_baseline_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    train_path = DATA_PROCESSED / "log_sales_dataset_train.csv"
    test_path = DATA_PROCESSED / "log_sales_dataset_test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"추천용 데이터셋이 없습니다. 먼저 scripts/eda_lipstick_effect.py 를 실행하세요: {train_path}, {test_path}"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_cfg = FeatureConfig(target_col="target_log_sales_next")

    results = train_baseline_models(
        train_df,
        test_df,
        feature_cfg,
        actual_col="target_log_sales_next",
    )

    print("\n=== 추천용 모델 (타깃: 다음 분기 로그매출) ===")
    for name, metrics in results.items():
        print(f"\n[{name}]")
        print(f"RMSE         : {metrics['rmse']:.4f}")
        print(f"MAE          : {metrics['mae']:.4f}")
        print(f"Top-20 Recall: {metrics['top20_recall']:.4f}")


if __name__ == "__main__":
    main()
