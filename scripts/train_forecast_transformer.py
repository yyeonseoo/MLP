"""
전망 모델 — Transformer: 시퀀스(과거 seq_len 분기) → 다음 분기 log_sales 예측.

- 데이터/분할/피처: train_forecast_model.py와 동일 (2023~ test, 동일 feature_cols)
- 시퀀스 구축 후 Transformer Encoder 학습, MAE/RMSE 평가 및 저장
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.datasets import temporal_train_test_split
from src.models.rank_dataset import (
    add_rank_features,
    add_target_encoding,
)
from src.models.forecast_transformer import (
    build_sequence_arrays,
    get_transformer_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FORECAST_TARGET_COL = "target_next"

SEQ_LEN = 4
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DROPOUT = 0.1
EPOCHS = 30
BATCH_SIZE = 512
LR = 1e-3


def _ensure_target_and_features(panel: pd.DataFrame) -> pd.DataFrame:
    """train_forecast_model과 동일."""
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
    """train_forecast_model과 동일."""
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
    df = df.loc[df["sales"].fillna(0) >= 1_000_000.0].copy()
    if df.empty:
        raise ValueError("조건을 만족하는 행이 없습니다.")

    train_df, test_df = temporal_train_test_split(df, test_start_year=2023, test_start_quarter=1)
    train_df, test_df = add_target_encoding(
        train_df, test_df,
        target_col=FORECAST_TARGET_COL,
        encode_cols=["region_id", "sector_code"],
    )
    # 시퀀스는 "예측 시점" 기준으로 train/test가 나뉘므로, 전체를 합친 뒤 build_sequence_arrays에서 분할
    full = pd.concat([train_df, test_df], ignore_index=True)
    full = full.sort_values(["region_id", "sector_code", "year", "quarter"])
    feature_cols = get_forecast_feature_columns(train_df)
    if not feature_cols:
        raise ValueError("전망 피처가 없습니다.")

    out = build_sequence_arrays(
        full, feature_cols, FORECAST_TARGET_COL,
        seq_len=SEQ_LEN, test_start_year=2023, test_start_quarter=1,
        return_test_current_log_sales=True,
    )
    X_train, y_train, X_test, y_test = out[0], out[1], out[2], out[3]
    current_log_sales_test = out[4]
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("시퀀스 train 또는 test 샘플이 0개입니다.")

    print(f"Train sequences: {X_train.shape[0]}, Test sequences: {X_test.shape[0]}")
    print(f"Seq length: {SEQ_LEN}, Features: {len(feature_cols)}")

    import torch
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tr = torch.from_numpy(X_train)
    y_tr = torch.from_numpy(y_train).unsqueeze(1)
    X_te = torch.from_numpy(X_test)
    y_te = torch.from_numpy(y_test)

    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    model = get_transformer_model(
        seq_len=SEQ_LEN,
        num_features=len(feature_cols),
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb.squeeze(-1))
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} loss={total_loss/len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(X_te), BATCH_SIZE):
            batch = X_te[i : i + BATCH_SIZE].to(device)
            preds.append(model(batch).cpu().numpy())
        pred = np.concatenate(preds, axis=0)

    mae = float(np.abs(y_test - pred).mean())
    rmse = float(np.sqrt(((y_test - pred) ** 2).mean()))
    # 분류: 다음 분기 매출 > 당분기 매출 여부
    y_true_class = (y_test > current_log_sales_test).astype(int)
    pred_class = (pred > current_log_sales_test).astype(int)
    from sklearn.metrics import f1_score
    accuracy = float((y_true_class == pred_class).mean())
    f1 = float(f1_score(y_true_class, pred_class, zero_division=0))
    print("\n--- Transformer 성능 (Test) ---")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1:       {f1:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "accuracy": accuracy,
        "f1": f1,
        "eval_set": "test",
        "model": "transformer",
        "seq_len": SEQ_LEN,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(MODELS_DIR / "forecast_transformer_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    torch.save(model.state_dict(), MODELS_DIR / "forecast_transformer.pt")
    with open(MODELS_DIR / "forecast_transformer_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "feature_cols": feature_cols,
            "seq_len": SEQ_LEN,
            "d_model": D_MODEL,
            "nhead": NHEAD,
            "num_layers": NUM_LAYERS,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {MODELS_DIR}/forecast_transformer_metrics.json, forecast_transformer.pt")


if __name__ == "__main__":
    main()
