"""
API 예측용 (region_id, sector_code)별 최신 분기 1행을 forecast_latest_inputs.csv로 저장.

train_forecast_model.py와 동일한 전처리·분할·target encoding 후,
train+test를 합쳐 (region_id, sector_code)별 최신 행만 저장.

선행: sales_panel.csv, models/forecast_feature_cols.json (피처 목록 일치)
출력: data/processed/forecast_latest_inputs.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.datasets import temporal_train_test_split
from src.models.rank_dataset import add_rank_features, add_target_encoding

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FORECAST_TARGET_COL = "target_next"


def main() -> None:
    panel_path = DATA_PROCESSED / "sales_panel.csv"
    if not panel_path.exists():
        raise FileNotFoundError(f"sales_panel.csv 없음: {panel_path}")

    with open(MODELS_DIR / "forecast_feature_cols.json", encoding="utf-8") as f:
        feature_cols = json.load(f)["feature_cols"]

    panel = pd.read_csv(panel_path)
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
    df = df.dropna(subset=[FORECAST_TARGET_COL])
    df = df.loc[df["sales"].fillna(0) >= 1_000_000.0].copy()

    train_df, test_df = temporal_train_test_split(df, test_start_year=2023, test_start_quarter=1)
    train_df, test_df = add_target_encoding(
        train_df, test_df, target_col=FORECAST_TARGET_COL, encode_cols=["region_id", "sector_code"]
    )
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined = combined.sort_values(["region_id", "sector_code", "year", "quarter"])
    latest = combined.groupby(["region_id", "sector_code"], as_index=False).last()
    out_cols = ["region_id", "sector_code", "region_name", "sector_name", "year_quarter"] + feature_cols + ["log_sales"]
    out_cols = [c for c in out_cols if c in latest.columns]
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    latest[out_cols].to_csv(DATA_PROCESSED / "forecast_latest_inputs.csv", index=False)
    print(f"저장: {DATA_PROCESSED / 'forecast_latest_inputs.csv'} (행 수: {len(latest)})")


if __name__ == "__main__":
    main()
