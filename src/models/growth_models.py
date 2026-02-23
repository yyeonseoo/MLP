from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


NUMERIC_DTYPES = ("int16", "int32", "int64", "float16", "float32", "float64")


@dataclass
class FeatureConfig:
    """
    성장률 예측에 사용할 특징 선택 규칙.

    기본 전략:
      - 수치형 컬럼 중에서 ID/시간/타깃 관련 컬럼은 제외
      - 남는 수치형을 모두 feature로 사용
    """

    target_col: str = "target_next"
    id_cols: Tuple[str, ...] = (
        "region_id",
        "sector_code",
    )
    time_cols: Tuple[str, ...] = (
        "year",
        "quarter",
    )
    drop_cols_explicit: Tuple[str, ...] = (
        "sales",
        "sales_prev",
        "transactions",
        "transactions_prev",
        "sales_growth_qoq",
        "sales_growth_log",  # 타깃과 동일 스케일(현재 분기 log-diff) → 제외
    )


def build_feature_matrix(
    df: pd.DataFrame,
    cfg: FeatureConfig | None = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    DataFrame에서 X, y, feature_names를 생성한다.
    """
    if cfg is None:
        cfg = FeatureConfig()

    data = df.copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[cfg.target_col])

    # 수치형 컬럼만 후보
    numeric_cols = [
        c
        for c in data.columns
        if str(data[c].dtype) in NUMERIC_DTYPES and c != cfg.target_col
    ]

    exclude = set(cfg.id_cols) | set(cfg.time_cols) | set(cfg.drop_cols_explicit)
    feature_cols = [c for c in numeric_cols if c not in exclude]

    X = data[feature_cols].fillna(0.0)
    y = data[cfg.target_col].astype(float)
    return X, y, feature_cols


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def compute_topk_recall(
    df: pd.DataFrame,
    k: int = 20,
    group_key: str = "year_quarter",
    actual_col: str = "target_next",
    pred_col: str = "pred",
) -> float:
    """
    기간(group_key)별로 실제 상위 k vs 예측 상위 k 교집합 비율의 평균을 계산.
    """
    recalls: List[float] = []
    for _, g in df.groupby(group_key):
        if len(g) < k:
            continue
        actual_top = (
            g.sort_values(actual_col, ascending=False)
            .head(k)
            .index.to_list()
        )
        pred_top = (
            g.sort_values(pred_col, ascending=False)
            .head(k)
            .index.to_list()
        )
        inter = len(set(actual_top) & set(pred_top))
        recalls.append(inter / float(k))

    return float(np.mean(recalls)) if recalls else float("nan")


def train_baseline_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cfg: FeatureConfig | None = None,
) -> Dict[str, Dict[str, float]]:
    """
    LinearRegression, RandomForest, LightGBM 세 가지 베이스라인 학습 및 평가.

    반환값:
      {
        "linear": {"rmse": ..., "mae": ..., "top20_recall": ...},
        "rf": {...},
        "lgbm": {...},
      }
    """
    if feature_cfg is None:
        feature_cfg = FeatureConfig()

    X_train, y_train, feature_cols = build_feature_matrix(train_df, feature_cfg)
    test_subset = test_df.copy()
    for c in feature_cols + [feature_cfg.target_col]:
        if c not in test_subset.columns:
            test_subset[c] = 0.0
    X_test, y_test, _ = build_feature_matrix(
        test_subset[feature_cols + [feature_cfg.target_col]], feature_cfg
    )

    results: Dict[str, Dict[str, float]] = {}

    models: dict = {
        "linear": LinearRegression(),
        "rf": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            n_jobs=-1,
            random_state=42,
        ),
    }
    try:
        from lightgbm import LGBMRegressor
        models["lgbm"] = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
    except Exception:
        pass

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "rmse": rmse(y_test.values, y_pred),
            "mae": mae(y_test.values, y_pred),
        }

        # Top-20 recall
        eval_df = test_df.copy()
        eval_df = eval_df.loc[X_test.index].copy()
        eval_df["pred"] = y_pred
        metrics["top20_recall"] = compute_topk_recall(eval_df, k=20)

        results[name] = metrics

    return results

