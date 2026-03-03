"""
전망 모델 비교: LightGBM, CatBoost, XGBoost, HistGradientBoosting, 앙상블(LGBM+CatBoost).

동일 데이터·동일 train/test 분할로 학습 후 MAE/RMSE 및 분류 지표(정확도, F1) 비교.
분류: "다음 분기 매출 > 당분기 매출" 여부(이진)로 변환 후 Accuracy, F1 계산.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.models.datasets import temporal_train_test_split
from src.models.rank_dataset import (
    add_rank_features,
    add_target_encoding,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FORECAST_TARGET_COL = "target_next"


def _ensure_target_and_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["region_id", "sector_code", "year", "quarter"]).copy()
    g = df.groupby(["region_id", "sector_code"], sort=False)
    if "log_sales" not in df.columns:
        df["log_sales"] = np.log1p(df["sales"].fillna(0))
    df[FORECAST_TARGET_COL] = g["log_sales"].shift(-1)
    if "sales_prev" not in df.columns:
        df["sales_prev"] = g["sales"].shift(1)
    if "txn_growth_qoq" not in df.columns and "transactions" in df.columns:
        t_prev = g["transactions"].shift(1)
        df["txn_growth_qoq"] = df["transactions"] / t_prev - 1.0
        df.loc[t_prev.isna() | (t_prev <= 0), "txn_growth_qoq"] = np.nan
    df = add_rank_features(df)
    for q in (2, 3, 4):
        if f"quarter_{q}" not in df.columns:
            df[f"quarter_{q}"] = (df["quarter"] == q).astype(int)
    return df


def get_forecast_feature_columns(df: pd.DataFrame) -> list[str]:
    drop = {
        "region_id", "sector_code", "sector_name", "region_name",
        "year", "quarter", "year_quarter", "rank_group",
        "sales", "sales_prev", "transactions", "transactions_prev",
        "sales_growth_qoq", "sales_growth_log", "log_sales",
        "target_log_sales_next", FORECAST_TARGET_COL,
        "quarter_code",
    }
    return [
        c for c in df.columns
        if str(df[c].dtype) in ("int16", "int32", "int64", "float16", "float32", "float64")
        and c not in drop
    ]


def main() -> None:
    panel_path = DATA_PROCESSED / "sales_panel.csv"
    if not panel_path.exists():
        raise FileNotFoundError(
            f"sales_panel.csv 없음. 먼저 PYTHONPATH=. python scripts/build_panel_and_regression.py 실행: {panel_path}"
        )

    panel = pd.read_csv(panel_path)
    df = _ensure_target_and_features(panel)
    df = df.dropna(subset=[FORECAST_TARGET_COL])
    df = df.loc[df["sales"].fillna(0) >= 1_000_000.0].copy()
    if df.empty:
        raise ValueError("조건을 만족하는 행이 없습니다.")

    train, test = temporal_train_test_split(df, test_start_year=2023, test_start_quarter=1)
    train, test = add_target_encoding(
        train, test,
        target_col=FORECAST_TARGET_COL,
        encode_cols=["region_id", "sector_code"],
    )
    feature_cols = get_forecast_feature_columns(train)
    if not feature_cols:
        raise ValueError("전망 피처가 없습니다.")

    X_train = train[feature_cols].fillna(0)
    y_train = train[FORECAST_TARGET_COL].values
    X_test = test[feature_cols].fillna(0)
    y_test = test[FORECAST_TARGET_COL].values
    # 분류용: 다음 분기 매출 > 당분기 매출 → 1 (상승), else 0
    current_log_sales = test["log_sales"].values
    y_true_class = (y_test > current_log_sales).astype(int)

    results = {}
    predictions = {}

    # 1. LightGBM (p50)
    try:
        from lightgbm import LGBMRegressor
        m = LGBMRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=31,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbosity=-1, objective="regression",
        )
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        predictions["lightgbm"] = pred
        pred_cls = (pred > current_log_sales).astype(int)
        results["lightgbm"] = {
            "mae": float(np.abs(y_test - pred).mean()),
            "rmse": float(np.sqrt(((y_test - pred) ** 2).mean())),
            "accuracy": float((y_true_class == pred_cls).mean()),
            "f1": float(f1_score(y_true_class, pred_cls, zero_division=0)),
        }
        print("[1] LightGBM done.")
    except Exception as e:
        results["lightgbm"] = {"mae": None, "rmse": None, "error": str(e)}
        print(f"[1] LightGBM skip: {e}")

    # 2. CatBoost
    try:
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(
            iterations=300, learning_rate=0.05, depth=6,
            subsample=0.8, colsample_bylevel=0.8, random_seed=42,
            verbose=0,
        )
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        predictions["catboost"] = pred
        pred_cls = (pred > current_log_sales).astype(int)
        results["catboost"] = {
            "mae": float(np.abs(y_test - pred).mean()),
            "rmse": float(np.sqrt(((y_test - pred) ** 2).mean())),
            "accuracy": float((y_true_class == pred_cls).mean()),
            "f1": float(f1_score(y_true_class, pred_cls, zero_division=0)),
        }
        print("[2] CatBoost done.")
    except Exception as e:
        results["catboost"] = {"mae": None, "rmse": None, "error": str(e)}
        print(f"[2] CatBoost skip: {e}")

    # 3. 앙상블 (LightGBM + CatBoost)
    if "lightgbm" in predictions and "catboost" in predictions:
        pred_ens = (predictions["lightgbm"] + predictions["catboost"]) / 2.0
        pred_cls = (pred_ens > current_log_sales).astype(int)
        results["ensemble_lgbm_catboost"] = {
            "mae": float(np.abs(y_test - pred_ens).mean()),
            "rmse": float(np.sqrt(((y_test - pred_ens) ** 2).mean())),
            "accuracy": float((y_true_class == pred_cls).mean()),
            "f1": float(f1_score(y_true_class, pred_cls, zero_division=0)),
        }
        print("[3] Ensemble (LGBM+CatBoost) done.")
    else:
        results["ensemble_lgbm_catboost"] = {"mae": None, "rmse": None}
        print("[3] Ensemble skip (need lightgbm + catboost).")

    # 4. XGBoost
    try:
        from xgboost import XGBRegressor
        m = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        )
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        pred_cls = (pred > current_log_sales).astype(int)
        results["xgboost"] = {
            "mae": float(np.abs(y_test - pred).mean()),
            "rmse": float(np.sqrt(((y_test - pred) ** 2).mean())),
            "accuracy": float((y_true_class == pred_cls).mean()),
            "f1": float(f1_score(y_true_class, pred_cls, zero_division=0)),
        }
        print("[4] XGBoost done.")
    except Exception as e:
        results["xgboost"] = {"mae": None, "rmse": None, "error": str(e)}
        print(f"[4] XGBoost skip: {e}")

    # 5. HistGradientBoostingRegressor (sklearn)
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=6,
            random_state=42,
        )
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        pred_cls = (pred > current_log_sales).astype(int)
        results["hist_gradient_boosting"] = {
            "mae": float(np.abs(y_test - pred).mean()),
            "rmse": float(np.sqrt(((y_test - pred) ** 2).mean())),
            "accuracy": float((y_true_class == pred_cls).mean()),
            "f1": float(f1_score(y_true_class, pred_cls, zero_division=0)),
        }
        print("[5] HistGradientBoosting done.")
    except Exception as e:
        results["hist_gradient_boosting"] = {"mae": None, "rmse": None, "error": str(e)}
        print(f"[5] HistGradientBoosting skip: {e}")

    # 출력 테이블 (회귀 + 분류)
    print("\n" + "=" * 80)
    print("전망 모델 비교 (Test set, 동일 데이터·분할)")
    print("분류: 다음 분기 매출 > 당분기 매출 여부 → Accuracy, F1")
    print("=" * 80)
    print(f"{'모델':<28} {'MAE':>8} {'RMSE':>8} {'Accuracy':>10} {'F1':>8}")
    print("-" * 80)
    for name, r in results.items():
        mae = r.get("mae")
        rmse = r.get("rmse")
        acc = r.get("accuracy")
        f1 = r.get("f1")
        if mae is not None and rmse is not None:
            acc_s = f"{acc:.4f}" if acc is not None else "—"
            f1_s = f"{f1:.4f}" if f1 is not None else "—"
            print(f"{name:<28} {mae:>8.4f} {rmse:>8.4f} {acc_s:>10} {f1_s:>8}")
        else:
            print(f"{name:<28} {'—':>8} {'—':>8} {'—':>10} {'—':>8}  ({r.get('error', '')})")
    print("=" * 80)

    # 저장 (숫자만, predictions 제외)
    out = {
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "error"} for k, v in results.items()},
        "eval_set": "test",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "forecast_compare_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {MODELS_DIR}/forecast_compare_results.json")


if __name__ == "__main__":
    main()
