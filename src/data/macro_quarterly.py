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

    # 숫자 컬럼 강제 (문자열로 읽혀서 diff 등이 깨지는 것 방지)
    for c in [config.cpi_col, config.policy_rate_col, config.unemployment_col, config.ccsi_col]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _parse_shock_quarters(
    shock_periods: (
        list[tuple[int, int]]  # [(year, quarter), ...]
        | list[str]           # ["2020Q1", "2020Q2", ...]
        | None
    ),
) -> set[str]:
    """shock_periods를 year_quarter 문자열 집합으로 변환."""
    if not shock_periods:
        return set()
    out: set[str] = set()
    for x in shock_periods:
        if isinstance(x, str):
            out.add(x.strip())
        elif isinstance(x, (list, tuple)) and len(x) >= 2:
            out.add(f"{int(x[0])}Q{int(x[1])}")
    return out


def _zscore(series: pd.Series) -> pd.Series:
    """Z-score; std=0 또는 결측은 0 처리."""
    out = pd.Series(0.0, index=series.index)
    valid = series.dropna()
    if len(valid) < 2:
        return out
    mu, sigma = valid.mean(), valid.std()
    if sigma is None or sigma == 0:
        return out
    out.loc[valid.index] = (valid - mu) / sigma
    return out


def add_macro_derivatives(
    df: pd.DataFrame,
    config: MacroConfig | None = None,
    shock_ccsi_threshold: float = 100.0,
    shock_cpi_yoy_quantile: float = 0.75,
    *,
    use_shock_score: bool = True,
    shock_score_top_quantile: float = 0.75,
    use_quantile_shock: bool = True,
    ccsi_low_quantile: float = 0.25,
    cpi_yoy_high_quantile: float = 0.75,
    policy_rate_diff_high_quantile: float = 0.75,
    shock_periods: (
        list[tuple[int, int]] | list[str] | None
    ) = None,
) -> pd.DataFrame:
    """
    파생 변수 (YoY, 변화량, 충격 점수·더미) 계산.

    - cpi_yoy        : 전년 동분기 대비 CPI 변화율
    - ccsi_diff      : 전분기 대비 소비자심리지수 차이
    - policy_rate_diff: 전분기 대비 기준금리 변화
    - shock_score    : 연속형 충격 점수 (Z-score 복합). 높을수록 거시 스트레스 큼.
    - macro_shock    : 충격기(1) vs 비충격기(0). 기준은 아래 우선순위.

    충격기 기준 (우선순위):
    1. shock_periods 가 주어지면: 해당 분기만 충격기(1). (고정 구간 실험용)
    2. use_shock_score=True(기본): shock_score 상위 (1 - shock_score_top_quantile) 비율만 충격기.
       예: shock_score_top_quantile=0.75 → 상위 25% 분기만 shock=1. 시계열이 블록으로 갈리지 않음.
    3. use_shock_score=False, use_quantile_shock=True: 구식 OR 조건 (CCSI≤Q25 OR CPI YoY≥Q75 OR Δ금리≥Q75).
    4. use_quantile_shock=False: (CCSI < threshold) OR (CPI YoY ≥ quantile).
    """
    if config is None:
        config = MacroConfig()

    # 정렬: 반드시 year, quarter 숫자 기준 (문자열 year_quarter 정렬 시 2020Q10 등 꼬일 수 있음)
    df = df.sort_values(["year", "quarter"]).reset_index(drop=True)

    # 전년 동분기 CPI
    df["cpi_yoy"] = np.nan
    if config.cpi_col in df.columns:
        s = df.sort_values(["year", "quarter"])[config.cpi_col]
        df["cpi_yoy"] = s.pct_change(periods=4).values

    # CCSI 변화량 (정렬된 순서 기준 diff)
    if config.ccsi_col in df.columns:
        df["ccsi_diff"] = df[config.ccsi_col].astype(float).diff()

    # 기준금리 변화량 (정렬된 순서 기준 diff)
    if config.policy_rate_col in df.columns:
        df["policy_rate_diff"] = df[config.policy_rate_col].astype(float).diff()

    # 실업률 변화량 (사용자용 지표용)
    if config.unemployment_col in df.columns:
        df["unemployment_diff"] = df[config.unemployment_col].astype(float).diff()

    # ----- 연속형 충격 점수 (Z-score 복합): 높을수록 스트레스 큼 -----
    shock_score = pd.Series(0.0, index=df.index)
    if config.ccsi_col in df.columns:
        z_ccsi = _zscore(df[config.ccsi_col])
        shock_score = shock_score + (-z_ccsi)
    if "cpi_yoy" in df.columns:
        z_cpi = _zscore(df["cpi_yoy"].fillna(0.0))
        shock_score = shock_score + z_cpi
    if "policy_rate_diff" in df.columns:
        z_rate = _zscore(df["policy_rate_diff"].fillna(0.0))
        shock_score = shock_score + z_rate
    df["shock_score"] = shock_score.values

    shock = np.zeros(len(df), dtype=int)
    shock_quarters = _parse_shock_quarters(shock_periods)

    if shock_quarters:
        yq = df["year"].astype(str) + "Q" + df["quarter"].astype(str)
        shock = np.where(yq.isin(shock_quarters), 1, 0).astype(int)
    elif use_shock_score:
        q = df["shock_score"].quantile(shock_score_top_quantile)
        shock = (df["shock_score"].values >= q).astype(int)
    elif use_quantile_shock:
        cond_ccsi = np.zeros(len(df), dtype=bool)
        cond_cpi = np.zeros(len(df), dtype=bool)
        cond_rate = np.zeros(len(df), dtype=bool)
        if config.ccsi_col in df.columns:
            v = df[config.ccsi_col].dropna()
            if len(v) > 0:
                cut = v.quantile(ccsi_low_quantile)
                cond_ccsi = (df[config.ccsi_col] <= cut).values

        if "cpi_yoy" in df.columns:
            v = df["cpi_yoy"].dropna()
            if len(v) > 0:
                cut = v.quantile(cpi_yoy_high_quantile)
                cond_cpi = (df["cpi_yoy"].fillna(-np.inf) >= cut).values

        if "policy_rate_diff" in df.columns:
            v = df["policy_rate_diff"].dropna()
            if len(v) > 0:
                cut = v.quantile(policy_rate_diff_high_quantile)
                cond_rate = (df["policy_rate_diff"].fillna(-np.inf) >= cut).values

        shock = (cond_ccsi | cond_cpi | cond_rate).astype(int)
    else:
        # 기존 방식 (threshold + cpi quantile)
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


def compute_shock_score_ui(
    df: pd.DataFrame,
    ccsi_diff_col: str = "ccsi_diff",
    cpi_yoy_col: str = "cpi_yoy",
    policy_rate_diff_col: str = "policy_rate_diff",
    unemployment_diff_col: str | None = "unemployment_diff",
    *,
    scale_01: str = "percentile",
) -> pd.DataFrame:
    """
    사용자용 거시 위험도 지표 (Event-aware Stress Index).

    - 악화 방향만 반영: 심리 하락, 물가 상승, 금리 인상, 실업 증가 → 각각 max(0, z) 또는 max(0, -z).
    - Z-score는 전체 기간 기준. 합산 후 percentile로 0~1 리스케일 (단조 추세 완화).
    - 반환: shock_score_ui (0~1), shock_score_ui_100 (0~100), contrib_ccsi, contrib_cpi, contrib_rate, contrib_unemp.

    Parameters
    ----------
    df : 이미 add_macro_derivatives 등으로 ccsi_diff, cpi_yoy, policy_rate_diff(, unemployment_diff) 있는 DataFrame.
    scale_01 : "percentile" (기본, 지난 기간 대비 상대) | "minmax" (합산값 min-max 스케일).
    """
    out = df.copy()
    n = len(out)

    def _z(s: pd.Series) -> pd.Series:
        return _zscore(s.fillna(0.0))

    # 악화 방향만: 심리 하락 = 위험 → max(0, -z(ccsi_diff))
    contrib_ccsi = pd.Series(0.0, index=out.index)
    if ccsi_diff_col in out.columns:
        z_ccsi = _z(out[ccsi_diff_col])
        contrib_ccsi = np.maximum(0.0, -z_ccsi.values)  # 하락이면 z 음수 → -z 양수

    # 물가 상승 = 위험
    contrib_cpi = pd.Series(0.0, index=out.index)
    if cpi_yoy_col in out.columns:
        z_cpi = _z(out[cpi_yoy_col])
        contrib_cpi = np.maximum(0.0, z_cpi.values)

    # 금리 인상 = 위험
    contrib_rate = pd.Series(0.0, index=out.index)
    if policy_rate_diff_col in out.columns:
        z_rate = _z(out[policy_rate_diff_col])
        contrib_rate = np.maximum(0.0, z_rate.values)

    # 실업 증가 = 위험
    contrib_unemp = pd.Series(0.0, index=out.index)
    if unemployment_diff_col and unemployment_diff_col in out.columns:
        z_unemp = _z(out[unemployment_diff_col])
        contrib_unemp = np.maximum(0.0, z_unemp.values)

    total = pd.Series(
        np.asarray(contrib_ccsi) + np.asarray(contrib_cpi) + np.asarray(contrib_rate) + np.asarray(contrib_unemp),
        index=out.index,
    )

    if scale_01 == "percentile":
        # 지난 기간 대비 백분위 → 단조 추세 완화
        out["shock_score_ui"] = total.rank(pct=True, method="average").values
    else:
        # min-max
        tmin, tmax = total.min(), total.max()
        if tmax > tmin:
            out["shock_score_ui"] = ((total - tmin) / (tmax - tmin)).values
        else:
            out["shock_score_ui"] = 0.5

    out["shock_score_ui_100"] = (out["shock_score_ui"] * 100).round(1)
    out["contrib_ccsi"] = np.asarray(contrib_ccsi)
    out["contrib_cpi"] = np.asarray(contrib_cpi)
    out["contrib_rate"] = np.asarray(contrib_rate)
    out["contrib_unemp"] = np.asarray(contrib_unemp)
    return out

