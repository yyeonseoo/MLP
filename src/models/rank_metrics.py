"""
추천/랭킹 평가 지표: NDCG@K, Precision@K, HitRate@K.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def dcg_at_k(relevances: List[float], k: int) -> float:
    """DCG@k. relevances: 상위 k개 예측 순서대로의 실제 relevance(예: 로그매출)."""
    relevances = np.asarray(relevances[:k], dtype=float)
    if relevances.size == 0:
        return 0.0
    gains = np.power(2, relevances) - 1
    discounts = np.log2(np.arange(2, gains.size + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 20,
) -> float:
    """
    NDCG@k. 한 그룹(상권·분기) 내에서 예측 순서와 실제 순서의 일치도.

    y_true, y_pred: 해당 그룹의 실제·예측 점수 (같은 길이).
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0
    order = np.argsort(-np.asarray(y_pred))
    rel_pred = np.asarray(y_true)[order]
    dcg = dcg_at_k(rel_pred.tolist(), k)
    rel_best = np.sort(np.asarray(y_true))[::-1]
    idcg = dcg_at_k(rel_best.tolist(), k)
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def precision_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 20,
    top_pct: float = 0.2,
) -> float:
    """
    Precision@K: 예측 상위 K개 중 "실제 상위 20%"에 들어간 비율.
    top_pct: 실제로 "좋은" 것으로 볼 상위 비율 (기본 20%).
    """
    if len(y_true) < k:
        return 0.0
    n = len(y_true)
    n_top = max(1, int(n * top_pct))
    thresh = np.partition(y_true, -n_top)[-n_top]
    pred_order = np.argsort(-np.asarray(y_pred))[:k]
    hits = np.sum(np.asarray(y_true)[pred_order] >= thresh)
    return hits / float(k)


def hit_rate_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 20,
    top_pct: float = 0.2,
) -> float:
    """
    HitRate@K: 실제 상위 top_pct 안에 든 항목 중, 예측 상위 K개에 포함된 비율.
    """
    n = len(y_true)
    n_top = max(1, int(n * top_pct))
    thresh = np.partition(y_true, -n_top)[-n_top]
    actual_top_set = set(np.where(np.asarray(y_true) >= thresh)[0])
    pred_top_k = set(np.argsort(-np.asarray(y_pred))[:k])
    if len(actual_top_set) == 0:
        return 0.0
    hits = len(actual_top_set & pred_top_k)
    return hits / float(len(actual_top_set))


def evaluate_rank_groups(
    df: pd.DataFrame,
    group_key: str,
    actual_col: str,
    pred_col: str,
    k: int = 20,
) -> dict:
    """
    그룹별 NDCG@k, Precision@k, HitRate@k 평균.
    """
    ndcgs, precs, hits = [], [], []
    for _, g in df.groupby(group_key):
        if len(g) < 2:
            continue
        y_true = g[actual_col].values
        y_pred = g[pred_col].values
        ndcgs.append(ndcg_at_k(y_true, y_pred, k=k))
        precs.append(precision_at_k(y_true, y_pred, k=k))
        hits.append(hit_rate_at_k(y_true, y_pred, k=k))
    return {
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"precision@{k}": float(np.mean(precs)) if precs else 0.0,
        f"hit_rate@{k}": float(np.mean(hits)) if hits else 0.0,
    }
