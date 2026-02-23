from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from src.models.growth_models import FeatureConfig, build_feature_matrix


@dataclass
class RecommendConfig:
    """
    추천 로직 설정.

    기본 전략:
      - 입력: 특정 시점(또는 최근 분기)의 상권×업종 행들
      - 점수: model이 예측한 다음 분기 성장률 (target_next)
      - 출력: 상위 N개 업종, 립스틱 여부 포함
    """

    top_k: int = 20
    feature_cfg: FeatureConfig = FeatureConfig()


def recommend_top_n_for_region(
    model,
    df_current: pd.DataFrame,
    region_id: str,
    cfg: Optional[RecommendConfig] = None,
) -> pd.DataFrame:
    """
    특정 상권(region_id)에 대해 TOP-N 성장 예측 업종을 추천.

    Parameters
    ----------
    model:
        학습된 회귀 모델 (sklearn/LightGBM 등, .predict 지원).
    df_current:
        상권×업종×분기 패널 또는 growth_dataset에서
        "현재 시점" 행들만 추려온 DataFrame.
        (예: 가장 최신 year_quarter만 필터링한 상태)
    region_id:
        추천 대상 상권 코드.

    Returns
    -------
    DataFrame:
        columns: ["region_id", "sector_code", "sector_name", "predicted_growth", "is_lipstick", ...]
    """
    if cfg is None:
        cfg = RecommendConfig()

    df_region = df_current[df_current["region_id"] == region_id].copy()
    if df_region.empty:
        return df_region

    # target_next는 필요 없으므로, 임시로 0 채우고 feature만 사용
    if cfg.feature_cfg.target_col not in df_region.columns:
        df_region[cfg.feature_cfg.target_col] = 0.0

    X, _, feature_cols = build_feature_matrix(df_region, cfg.feature_cfg)

    preds = model.predict(X)
    df_region = df_region.loc[X.index].copy()
    df_region["predicted_growth"] = preds

    # 립스틱 여부 컬럼이 이미 패널에 있다고 가정 (is_lipstick)
    out_cols: List[str] = [
        "region_id",
        "sector_code",
    ]
    if "sector_name" in df_region.columns:
        out_cols.append("sector_name")
    if "is_lipstick" in df_region.columns:
        out_cols.append("is_lipstick")

    out_cols.append("predicted_growth")

    result = df_region[out_cols].sort_values("predicted_growth", ascending=False)
    return result.head(cfg.top_k)

