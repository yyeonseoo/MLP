"""
추천 로직: 상권별 다음 분기 TOP-K 업종.

- Regressor 방식: 예측 점수(성장률/로그매출)로 정렬
- Ranker 방식: Learning-to-Rank 모델로 점수 예측 후 정렬 (서비스 권장)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.growth_models import FeatureConfig, build_feature_matrix
from src.models.rank_dataset import get_rank_feature_columns


@dataclass
class RecommendConfig:
    """
    추천 로직 설정.
    - top_k: 추천 개수
    - feature_cfg: Regressor 사용 시 피처 설정
    """

    top_k: int = 20
    feature_cfg: FeatureConfig = FeatureConfig()


# 기본 랭커 모델 경로
DEFAULT_RANKER_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "ranker_lgbm.txt"
RANKER_FEATURE_COLS_JSON = Path(__file__).resolve().parents[1] / "data" / "processed" / "ranker_feature_cols.json"


def load_ranker(
    path: Optional[Path] = None,
) -> Tuple[object, List[str]]:
    """
    LightGBM Ranker 로드.
    반환: (booster, feature_cols). feature_cols는 ranker_feature_cols.json에서 로드.
    """
    path = Path(path or DEFAULT_RANKER_PATH)
    if not path.exists():
        raise FileNotFoundError(f"랭커 모델 없음: {path}. train_ranker_lgbm.py 실행 후 사용.")
    try:
        from lightgbm import Booster
        booster = Booster(model_file=str(path))
    except Exception:
        raise
    # 피처 순서 (저장된 JSON)
    cols_path = path.parent / "ranker_feature_cols.json"
    if cols_path.exists():
        with open(cols_path, encoding="utf-8") as f:
            feature_cols = json.load(f)
    else:
        feature_cols = []
    return booster, feature_cols


def recommend_top_n_ranker(
    ranker,
    df_region_quarter: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    한 상권·한 분기의 업종 행에 대해 랭커로 점수 예측 후 TOP-K 반환.

    Parameters
    ----------
    ranker: LightGBM Booster 또는 LGBMRanker (predict 지원)
    df_region_quarter: 해당 (region_id, year_quarter)의 행. 랭킹 피처 컬럼 포함.
    feature_cols: 사용할 피처. None이면 get_rank_feature_columns(df) 사용.
    top_k: 추천 개수.

    Returns
    -------
    DataFrame: region_id, sector_code, sector_name, score, is_lipstick 등 + rank
    """
    if df_region_quarter.empty:
        return df_region_quarter

    if feature_cols is None:
        feature_cols = get_rank_feature_columns(df_region_quarter)
    missing = [c for c in feature_cols if c not in df_region_quarter.columns]
    if missing:
        for c in missing:
            df_region_quarter[c] = 0.0

    X = df_region_quarter[feature_cols].fillna(0)
    if hasattr(ranker, "predict"):
        scores = ranker.predict(X)
    else:
        scores = ranker.predict(X)

    out = df_region_quarter.copy()
    out["score"] = scores
    out_cols = ["region_id", "sector_code", "score"]
    if "sector_name" in out.columns:
        out_cols.append("sector_name")
    if "is_lipstick" in out.columns:
        out_cols.append("is_lipstick")
    if "is_luxury" in out.columns:
        out_cols.append("is_luxury")
    out = out[out_cols].sort_values("score", ascending=False).head(top_k)
    out["rank"] = range(1, len(out) + 1)
    return out


def explain_recommendation(
    ranker,
    row: pd.Series,
    feature_cols: List[str],
    top_n: int = 3,
) -> List[str]:
    """
    추천 이유 요약 (피처 기여 상위 N개). SHAP 없이 importance·값으로 간이 설명.
    """
    reasons: List[str] = []
    imp = None
    if hasattr(ranker, "feature_importances_"):
        imp = ranker.feature_importances_
    elif hasattr(ranker, "feature_importance") and callable(ranker.feature_importance):
        imp = ranker.feature_importance(importance_type="gain")
    if imp is not None and len(imp) >= len(feature_cols):
        names = getattr(ranker, "feature_name_", None) or feature_cols
        if len(names) != len(imp):
            names = feature_cols
        idx = np.argsort(-np.asarray(imp))[:top_n]
        for i in idx:
            if i >= len(names):
                continue
            name = names[i] if i < len(names) else feature_cols[i] if i < len(feature_cols) else f"feat_{i}"
            val = row.get(name, 0)
            reasons.append(f"{name}={val:.2f}")
    if not reasons:
        reasons.append("최근 매출·거래 추세 반영")
    return reasons


def recommend_top_n_for_region(
    model,
    df_current: pd.DataFrame,
    region_id: str,
    cfg: Optional[RecommendConfig] = None,
) -> pd.DataFrame:
    """
    특정 상권(region_id)에 대해 TOP-N 성장 예측 업종을 추천 (Regressor용).

    model.predict로 점수 예측 후 정렬. 랭커는 recommend_top_n_ranker 사용 권장.
    """
    if cfg is None:
        cfg = RecommendConfig()

    df_region = df_current[df_current["region_id"] == region_id].copy()
    if df_region.empty:
        return df_region

    if cfg.feature_cfg.target_col not in df_region.columns:
        df_region[cfg.feature_cfg.target_col] = 0.0

    X, _, feature_cols = build_feature_matrix(df_region, cfg.feature_cfg)
    preds = model.predict(X)
    df_region = df_region.loc[X.index].copy()
    df_region["predicted_growth"] = preds

    out_cols: List[str] = ["region_id", "sector_code"]
    if "sector_name" in df_region.columns:
        out_cols.append("sector_name")
    if "is_lipstick" in df_region.columns:
        out_cols.append("is_lipstick")
    if "is_luxury" in df_region.columns:
        out_cols.append("is_luxury")
    out_cols.append("predicted_growth")

    result = df_region[out_cols].sort_values("predicted_growth", ascending=False)
    return result.head(cfg.top_k)
