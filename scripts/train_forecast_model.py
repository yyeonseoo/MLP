"""
전망 모델 학습 — "이번 분기 정보로 다음 분기 매출(로그)" 예측.

(1) 데이터 로드: data/processed/sales_panel.csv
(2) 타깃: log_sales = log(sales+1), target_next = 다음 분기 log_sales
(3) 피처: lag, rolling mean/std(4q), 계절(quarter), 거시(CPI/금리/CCSI 등), region/sector 인코딩
(4) 시간 분할: 2022 이하 train, 2023~ test
(5) LightGBM Quantile(p10/p50/p90) 학습
(6) MAE/RMSE 출력
(7) 모델 저장: models/forecast_p10.pkl, forecast_p50.pkl, forecast_p90.pkl
(8) 학습 직후 샘플 상권 3개에 대해 다음 분기 예상 매출/보수~낙관/전분기 대비 % 출력

선행: scripts/build_panel_and_regression.py (sales_panel.csv)

Mac에서 LightGBM 오류 시: brew install libomp
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.datasets import temporal_train_test_split
from src.models.rank_dataset import (
    add_rank_features,
    add_target_encoding,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# 타깃 컬럼명 (다음 분기 로그 매출)
FORECAST_TARGET_COL = "target_next"


def _ensure_target_and_features(panel: pd.DataFrame) -> pd.DataFrame:
    """target_next = 다음 분기 log_sales, lag/rolling/quarter 등 피처 추가."""
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
    # quarter 더미 (rank_dataset에 없을 수 있음)
    for q in (2, 3, 4):
        if f"quarter_{q}" not in df.columns:
            df[f"quarter_{q}"] = (df["quarter"] == q).astype(int)
    return df


def get_forecast_feature_columns(df: pd.DataFrame) -> list[str]:
    """전망용 수치형 피처 (target_next, id, sales 등 제외)."""
    drop = {
        "region_id", "sector_code", "sector_name", "region_name",
        "year", "quarter", "year_quarter", "rank_group",
        "sales", "sales_prev", "transactions", "transactions_prev",
        "sales_growth_qoq", "sales_growth_log", "log_sales",
        "target_log_sales_next", FORECAST_TARGET_COL,
        "quarter_code",
    }
    numeric = [
        c for c in df.columns
        if str(df[c].dtype) in ("int16", "int32", "int64", "float16", "float32", "float64")
        and c not in drop
    ]
    return numeric


def main() -> None:
    panel_path = DATA_PROCESSED / "sales_panel.csv"
    if not panel_path.exists():
        raise FileNotFoundError(
            f"sales_panel.csv 없음. 먼저 PYTHONPATH=. python scripts/build_panel_and_regression.py 실행: {panel_path}"
        )

    panel = pd.read_csv(panel_path)
    df = _ensure_target_and_features(panel)
    df = df.dropna(subset=[FORECAST_TARGET_COL])
    min_sales = 1_000_000.0
    df = df.loc[df["sales"].fillna(0) >= min_sales].copy()
    if df.empty:
        raise ValueError("조건을 만족하는 행이 없습니다. min_sales 또는 패널 데이터를 확인하세요.")

    # 시간 기준 분할: 2023년부터 test
    train, test = temporal_train_test_split(df, test_start_year=2023, test_start_quarter=1)
    # target encoding (train 통계로 test 매핑)
    train, test = add_target_encoding(
        train, test,
        target_col=FORECAST_TARGET_COL,
        encode_cols=["region_id", "sector_code"],
    )

    feature_cols = get_forecast_feature_columns(train)
    if not feature_cols:
        raise ValueError("전망 피처가 없습니다. 패널에 거시/quarter 컬럼이 있는지 확인하세요.")

    X_train = train[feature_cols].fillna(0)
    y_train = train[FORECAST_TARGET_COL].values
    X_test = test[feature_cols].fillna(0)
    y_test = test[FORECAST_TARGET_COL].values

    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        raise ImportError("lightgbm 필요: pip install lightgbm") from None

    base = dict(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    models = {}
    for name, alpha in [("p10", 0.1), ("p50", 0.5), ("p90", 0.9)]:
        m = LGBMRegressor(**base, objective="quantile", alpha=alpha)
        m.fit(X_train, y_train)
        models[name] = m

    # (6) 성능 평가
    pred_p50 = models["p50"].predict(X_test)
    mae = np.abs(y_test - pred_p50).mean()
    rmse = np.sqrt(((y_test - pred_p50) ** 2).mean())
    print("--- 성능 (Test, 중앙값 예측) ---")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")

    # (7) 모델 저장
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    for name, m in models.items():
        joblib.dump(m, MODELS_DIR / f"forecast_{name}.pkl")
    with open(MODELS_DIR / "forecast_feature_cols.json", "w", encoding="utf-8") as f:
        json.dump({"feature_cols": feature_cols}, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {MODELS_DIR}/forecast_p10.pkl, forecast_p50.pkl, forecast_p90.pkl, forecast_feature_cols.json")

    # (8) 학습 직후 확인: 샘플 상권 3개 예측
    print("\n--- 샘플 상권 3개 다음 분기 전망 ---")
    test_df = test.reset_index(drop=True)
    # 서로 다른 (region_id, sector_code) 3개 선택
    uniq = test_df.groupby(["region_id", "sector_code"]).agg({"year": "max", "quarter": "max"}).reset_index()
    uniq = uniq.head(3)
    for _, r in uniq.iterrows():
        rid, sid = r["region_id"], r["sector_code"]
        row = test_df[(test_df["region_id"] == rid) & (test_df["sector_code"] == sid)].sort_values(["year", "quarter"]).iloc[-1:]
        if row.empty:
            continue
        X_row = row[feature_cols].fillna(0)
        p10 = models["p10"].predict(X_row)[0]
        p50 = models["p50"].predict(X_row)[0]
        p90 = models["p90"].predict(X_row)[0]
        # log -> 원화 백만원 (exp(log_sales)-1 ≈ sales, 단위 1원 -> 백만원 = /1e6)
        def log_to_million(log_sales: float) -> float:
            return max(0.0, np.expm1(log_sales) / 1e6)
        m10 = log_to_million(p10)
        m50 = log_to_million(p50)
        m90 = log_to_million(p90)
        current = row["log_sales"].iloc[0]
        current_m = log_to_million(current)
        growth_pct = (m50 - current_m) / current_m * 100.0 if current_m > 0 else 0.0
        region_name = row["region_name"].iloc[0] if "region_name" in row.columns else rid
        sector_name = row["sector_name"].iloc[0] if "sector_name" in row.columns else sid
        print(f"  [{region_name} × {sector_name}]")
        print(f"    다음 분기 예상 매출(중앙): {m50:.1f}백만원 ({m50/100:.1f}억원)")
        print(f"    보수~낙관: {m10:.1f} ~ {m90:.1f} 백만원")
        print(f"    전분기 대비 변화율: {growth_pct:+.1f}%")
        print()


if __name__ == "__main__":
    main()
