"""
시계열 Transformer: (region, sector)별 과거 seq_len 분기 피처 시퀀스 → 다음 분기 log_sales 예측.

- 입력: (batch, seq_len, num_features)
- 출력: (batch,) 스칼라 예측
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def build_sequence_arrays(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seq_len: int,
    test_start_year: int = 2023,
    test_start_quarter: int = 1,
    return_test_current_log_sales: bool = False,
) -> Tuple:
    """
    그룹별 시퀀스: [row i-seq_len .. row i-1] (seq_len개) → row i의 target_next (다음 분기 로그매출).
    예측 시점 = row i의 다음 분기 → test 여부: next_year >= test_start_year and next_q >= test_start_quarter.

    Returns
    -------
    X_train, y_train, X_test, y_test
    또는 return_test_current_log_sales=True 시 (..., current_log_sales_test)
    """
    df = df.sort_values(["region_id", "sector_code", "year", "quarter"]).copy()
    F = len(feature_cols)
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    current_log_sales_test_list: List[float] = []

    for _, grp in df.groupby(["region_id", "sector_code"], sort=False):
        grp = grp.reset_index(drop=True)
        if len(grp) <= seq_len:
            continue
        mat = grp[feature_cols].fillna(0).values.astype(np.float32)
        if "log_sales" not in grp.columns:
            log_sales = np.log1p(grp["sales"].fillna(0).values)
        else:
            log_sales = grp["log_sales"].values
        for i in range(seq_len, len(grp)):
            y_val = grp[target_col].iloc[i]
            if not np.isfinite(y_val):
                continue
            X_seq = mat[i - seq_len : i]  # (seq_len, F)
            year_cur = int(grp["year"].iloc[i])
            quarter_cur = int(grp["quarter"].iloc[i])
            next_year = year_cur + (1 if quarter_cur == 4 else 0)
            next_q = 1 if quarter_cur == 4 else quarter_cur + 1
            is_test = (next_year > test_start_year) or (
                next_year == test_start_year and next_q >= test_start_quarter
            )
            if is_test:
                X_test_list.append(X_seq)
                y_test_list.append(y_val)
                if return_test_current_log_sales:
                    current_log_sales_test_list.append(float(log_sales[i]))
            else:
                X_train_list.append(X_seq)
                y_train_list.append(y_val)

    F = len(feature_cols)
    if not X_train_list or not X_test_list:
        out = (
            np.zeros((0, seq_len, F), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros((0, seq_len, F), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )
        if return_test_current_log_sales:
            return out + (np.zeros(0, dtype=np.float32),)
        return out
    out = (
        np.stack(X_train_list, axis=0),
        np.array(y_train_list, dtype=np.float32),
        np.stack(X_test_list, axis=0),
        np.array(y_test_list, dtype=np.float32),
    )
    if return_test_current_log_sales:
        return out + (np.array(current_log_sales_test_list, dtype=np.float32),)
    return out


def get_transformer_model(seq_len: int, num_features: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
    """PyTorch Transformer Encoder 기반 회귀 모델 반환."""
    import torch
    import torch.nn as nn

    class TransformerForecast(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(num_features, d_model)
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, T, F)
            h = self.input_proj(x)  # (B, T, d_model)
            h = self.transformer(h)  # (B, T, d_model)
            h = h[:, -1, :]  # (B, d_model) — 마지막 시점만
            return self.fc(h).squeeze(-1)  # (B,)

    return TransformerForecast()
