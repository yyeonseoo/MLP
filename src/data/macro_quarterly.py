from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class MacroConfig:
    """
    분기별 거시지표 CSV 포맷 정의.

    권장 포맷 (예: data/processed/macro_quarterly.csv)
    --------------------------------------------------
    - year           : 연도 (int)
    - quarter        : 분기 (1~4, int)
    - cpi            : 소비자물가지수 (지수)
    - policy_rate    : 기준금리 (%)
    - unemployment   : 실업률 (%)
    - ccsi           : 소비자심리지수
    - real_wage      : 실질임금 지수 (선택)
    """

    year_col: str = "year"
    quarter_col: str = "quarter"
    cpi_col: str = "cpi"
    policy_rate_col: str = "policy_rate"
    unemployment_col: str = "unemployment"
    ccsi_col: str = "ccsi"
    real_wage_col: str | None = "real_wage"


def load_macro_quarterly(
    path: str | Path,
    config: MacroConfig | None = None,
) -> pd.DataFrame:
    """
    이미 분기 단위로 집계된 거시지표 CSV를 로드한다.

    실제 엑셀(xlsx) 파일은 별도 노트북/스크립트에서 처리하여
    data/processed/macro_quarterly.csv 형태로 만드는 것을 권장한다.
    """
    if config is None:
        config = MacroConfig()

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(p)

    # 기본 컬럼 존재 여부 확인
    required = {config.year_col, config.quarter_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Macro quarterly CSV missing required columns: {missing}")

    df = df.copy()
    df["year"] = df[config.year_col].astype(int)
    df["quarter"] = df[config.quarter_col].astype(int)
    df["year_quarter"] = df["year"].astype(str) + "Q" + df["quarter"].astype(str)

    return df


def add_macro_derivatives(
    df: pd.DataFrame,
    config: MacroConfig | None = None,
    shock_ccsi_threshold: float = 100.0,
    shock_cpi_yoy_quantile: float = 0.75,
) -> pd.DataFrame:
    """
    파생 변수 (YoY, 변화량, 충격 더미) 계산.

    - cpi_yoy        : 전년 동분기 대비 CPI 변화율
    - ccsi_diff      : 전분기 대비 소비자심리지수 차이
    - policy_rate_diff: 전분기 대비 기준금리 변화
    - macro_shock    : (ccsi < threshold) 또는 (cpi_yoy 상위 quantile 이상) 이면 1
    """
    if config is None:
        config = MacroConfig()

    df = df.sort_values(["year", "quarter"]).copy()

    # 전년 동분기 CPI
    df["cpi_yoy"] = np.nan
    if config.cpi_col in df.columns:
        s = df.sort_values(["year", "quarter"])[config.cpi_col]
        df["cpi_yoy"] = s.pct_change(periods=4).values
        # 위 연산은 다소 복잡해질 수 있으므로, 간단히 groupby/year shift 방식으로도 가능

    # CCSI 변화량
    if config.ccsi_col in df.columns:
        df["ccsi_diff"] = df[config.ccsi_col].diff()

    # 기준금리 변화량
    if config.policy_rate_col in df.columns:
        df["policy_rate_diff"] = df[config.policy_rate_col].diff()

    # 충격 더미
    shock = np.zeros(len(df), dtype=int)

    cond_ccsi = (
        df[config.ccsi_col] < shock_ccsi_threshold
        if config.ccsi_col in df.columns
        else False
    )

    if "cpi_yoy" in df.columns:
        valid_cpi = df["cpi_yoy"].dropna()
        if not valid_cpi.empty:
            cpi_cut = valid_cpi.quantile(shock_cpi_yoy_quantile)
            cond_cpi = df["cpi_yoy"] >= cpi_cut
        else:
            cond_cpi = False
    else:
        cond_cpi = False

    shock[np.where(cond_ccsi | cond_cpi)[0]] = 1
    df["macro_shock"] = shock

    return df

