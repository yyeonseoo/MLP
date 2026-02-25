"""
매출 전망 API — 모델 pkl + forecast_latest_inputs.csv 기반 예측.

실행: 프로젝트 루트에서
  PYTHONPATH=. uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

app = FastAPI(
    title="매출 전망 API",
    description="지역(상권)·업종 입력 시 다음 분기 매출 예측 (p10/p50/p90, 증감률) 반환",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 시작 시 한 번 로드
_models = None
_feature_cols = None
_latest_df = None
_macro_df = None
_lipstick_df = None
_sales_panel_df = None


def _load_artifacts():
    global _models, _feature_cols, _latest_df
    if _models is not None and _latest_df is not None:
        return
    if not (MODELS_DIR / "forecast_feature_cols.json").exists():
        return
    if not (DATA_PROCESSED / "forecast_latest_inputs.csv").exists():
        return
    try:
        import json
        with open(MODELS_DIR / "forecast_feature_cols.json", encoding="utf-8") as f:
            _feature_cols = json.load(f)["feature_cols"]
        _models = {
            "p10": joblib.load(MODELS_DIR / "forecast_p10.pkl"),
            "p50": joblib.load(MODELS_DIR / "forecast_p50.pkl"),
            "p90": joblib.load(MODELS_DIR / "forecast_p90.pkl"),
        }
        _latest_df = pd.read_csv(DATA_PROCESSED / "forecast_latest_inputs.csv")
    except Exception:
        _models = None
        _feature_cols = None
        _latest_df = None


def _log_to_million(log_sales: float) -> float:
    return max(0.0, float(np.expm1(log_sales)) / 1e6)


def _load_macro():
    """거시 시계열 (macro_quarterly.csv). 파일 없거나 오류 시 빈 DataFrame."""
    global _macro_df
    if _macro_df is not None:
        return
    path = DATA_PROCESSED / "macro_quarterly.csv"
    try:
        if path.exists():
            _macro_df = pd.read_csv(path)
        else:
            _macro_df = pd.DataFrame()
    except Exception:
        _macro_df = pd.DataFrame()


def _load_lipstick():
    """립스틱 지역×분기 (lipstick_region_quarter.csv)."""
    global _lipstick_df
    if _lipstick_df is not None:
        return
    path = DATA_PROCESSED / "lipstick_region_quarter.csv"
    if path.exists():
        _lipstick_df = pd.read_csv(path)
    else:
        _lipstick_df = pd.DataFrame()


def _load_sales_panel():
    """매출 패널 (sales_panel.csv). API 호출 시에만 로드."""
    global _sales_panel_df
    if _sales_panel_df is not None:
        return
    path = DATA_PROCESSED / "sales_panel.csv"
    if path.exists():
        _sales_panel_df = pd.read_csv(path)
    else:
        _sales_panel_df = pd.DataFrame()


@app.on_event("startup")
def startup():
    if (MODELS_DIR / "forecast_p50.pkl").exists() and (DATA_PROCESSED / "forecast_latest_inputs.csv").exists():
        try:
            _load_artifacts()
        except Exception:
            pass


@app.get("/api/forecast/options")
def get_forecast_options():
    """드롭다운용 지역·업종 목록 + 유효 (지역×업종) 조합."""
    _load_artifacts()
    if _latest_df is None:
        raise HTTPException(status_code=503, detail="forecast_latest_inputs.csv 또는 모델이 없습니다.")
    regions = (
        _latest_df[["region_id", "region_name"]]
        .drop_duplicates()
        .sort_values("region_name")
        .to_dict("records")
    )
    sectors = (
        _latest_df[["sector_code", "sector_name"]]
        .drop_duplicates()
        .sort_values("sector_name")
        .to_dict("records")
    )
    combinations = (
        _latest_df[["region_id", "region_name", "sector_code", "sector_name"]]
        .drop_duplicates()
        .to_dict("records")
    )
    return {"regions": regions, "sectors": sectors, "combinations": combinations}


@app.get("/api/forecast")
def get_forecast(
    region_id: str = Query(..., description="상권 코드"),
    sector_code: str = Query(..., description="업종 코드"),
):
    """지역·업종에 대한 다음 분기 매출 예측 (p10/p50/p90, 증감률)."""
    _load_artifacts()
    if _latest_df is None or _models is None:
        raise HTTPException(status_code=503, detail="모델 또는 forecast_latest_inputs.csv가 없습니다.")
    row = _latest_df[
        (_latest_df["region_id"].astype(str) == str(region_id))
        & (_latest_df["sector_code"].astype(str) == str(sector_code))
    ]
    if row.empty:
        raise HTTPException(status_code=404, detail="해당 지역·업종 조합에 대한 전망 데이터가 없습니다.")
    row = row.iloc[0:1]
    X = row[_feature_cols].fillna(0)
    p10 = float(_models["p10"].predict(X)[0])
    p50 = float(_models["p50"].predict(X)[0])
    p90 = float(_models["p90"].predict(X)[0])
    m10 = _log_to_million(p10)
    m50 = _log_to_million(p50)
    m90 = _log_to_million(p90)
    base_quarter = str(row["year_quarter"].iloc[0])
    region_name = str(row["region_name"].iloc[0])
    sector_name = str(row["sector_name"].iloc[0])
    log_sales = row["log_sales"].iloc[0]
    current_m = _log_to_million(log_sales)
    growth_pct = float((m50 - current_m) / current_m * 100.0) if current_m > 0 else 0.0
    return {
        "base_quarter": base_quarter,
        "region_name": region_name,
        "sector_name": sector_name,
        "p10": round(m10, 1),
        "p50": round(m50, 1),
        "p90": round(m90, 1),
        "growth_pct": round(growth_pct, 1),
    }


@app.get("/api/dashboard/macro")
def get_dashboard_macro():
    """거시 시계열 (CPI, 금리, CCSI, shock_score). macro_quarterly.csv 기반."""
    try:
        _load_macro()
        if _macro_df is None or _macro_df.empty:
            return []
        cols = ["year_quarter", "year", "quarter", "cpi", "policy_rate", "ccsi", "cpi_yoy", "shock_score", "macro_shock"]
        cols = [c for c in cols if c in _macro_df.columns]
        if not cols:
            return []
        out = _macro_df[cols].copy()
        sort_by = [c for c in ["year", "quarter"] if c in out.columns]
        if sort_by:
            out = out.sort_values(sort_by)
        elif "year_quarter" in out.columns:
            out = out.sort_values("year_quarter")
        out = out.where(pd.notna(out), None)
        records = out.to_dict("records")
        # numpy 타입 → Python 기본 타입 (JSON 직렬화 안전)
        return [
            {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in row.items()}
            for row in records
        ]
    except Exception:
        import traceback
        traceback.print_exc()
        return []


@app.get("/api/dashboard/top_sectors")
def get_dashboard_top_sectors(limit: int = Query(30, ge=1, le=100, description="상위 N개")):
    """다음 분기 예상 성장률 상위 N개 (region×sector). forecast_latest_inputs + pkl 예측."""
    _load_artifacts()
    if _latest_df is None or _models is None:
        raise HTTPException(status_code=503, detail="모델 또는 forecast_latest_inputs.csv가 없습니다.")
    X = _latest_df[_feature_cols].fillna(0)
    p50_pred = _models["p50"].predict(X)
    current_m = _latest_df["log_sales"].map(_log_to_million)
    next_m = np.vectorize(_log_to_million)(p50_pred)
    growth_pct = np.where(current_m > 0, (next_m - current_m) / current_m * 100.0, 0.0)
    df = _latest_df[["region_id", "region_name", "sector_code", "sector_name", "year_quarter"]].copy()
    df["growth_pct"] = growth_pct
    df["p50_million"] = next_m
    df = df.sort_values("growth_pct", ascending=False).head(int(limit))
    df = df.reset_index(drop=True)
    df["rank"] = df.index + 1
    return df.to_dict("records")


@app.get("/api/dashboard/lipstick_series")
def get_dashboard_lipstick_series():
    """분기별 립스틱 비중(매출 가중) + macro_shock. lipstick_region_quarter + macro_quarterly."""
    _load_lipstick()
    _load_macro()
    if _lipstick_df is None or _lipstick_df.empty:
        return []
    agg = (
        _lipstick_df.groupby("year_quarter")
        .agg(sales_lipstick=("sales_lipstick", "sum"), sales_total=("sales_total", "sum"))
        .assign(lipstick_share=lambda x: x["sales_lipstick"] / x["sales_total"].replace(0, np.nan))
        .reset_index()
    )
    if _macro_df is not None and not _macro_df.empty and "macro_shock" in _macro_df.columns:
        agg = agg.merge(
            _macro_df[["year_quarter", "macro_shock"]].drop_duplicates(),
            on="year_quarter",
            how="left",
        )
    else:
        agg["macro_shock"] = 0
    agg = agg.where(pd.notna(agg), None)
    return agg[["year_quarter", "lipstick_share", "macro_shock"]].to_dict("records")


@app.get("/api/dashboard/growth_comparison")
def get_dashboard_growth_comparison():
    """분기별 립스틱 vs 논립스틱 매출 성장률 중앙값. sales_panel 기반."""
    _load_sales_panel()
    if _sales_panel_df is None or _sales_panel_df.empty or "sales_growth_qoq" not in _sales_panel_df.columns:
        return []
    df = _sales_panel_df.dropna(subset=["sales_growth_qoq"]).copy()
    df["sales_growth_qoq"] = np.clip(df["sales_growth_qoq"], -0.5, 0.5)
    if "is_lipstick" not in df.columns:
        return []
    med = df.groupby(["year_quarter", "is_lipstick"])["sales_growth_qoq"].median().unstack(fill_value=None).reset_index()
    rename_map = {c: "lipstick_median" if c in (True, 1) else "non_lipstick_median" for c in med.columns if c != "year_quarter"}
    med = med.rename(columns=rename_map)
    cols = [c for c in ["year_quarter", "lipstick_median", "non_lipstick_median"] if c in med.columns]
    return med[cols].where(pd.notna(med[cols]), None).to_dict("records")


@app.get("/api/dashboard/sensitivity_ranking")
def get_dashboard_sensitivity_ranking(limit: int = Query(20, ge=1, le=50)):
    """상권별 충격기-비충격기 립스틱 비중 차이 상위 N. lipstick_region_quarter + macro."""
    _load_lipstick()
    _load_macro()
    if _lipstick_df is None or _lipstick_df.empty or _macro_df is None or _macro_df.empty or "macro_shock" not in _macro_df.columns:
        return []
    merged = _lipstick_df.merge(
        _macro_df[["year_quarter", "macro_shock"]].drop_duplicates(),
        on="year_quarter",
        how="left",
    )
    shock_mean = merged[merged["macro_shock"] == 1].groupby(["region_id", "region_name"])["lipstick_share"].mean()
    non_mean = merged[merged["macro_shock"] == 0].groupby(["region_id", "region_name"])["lipstick_share"].mean()
    diff = (shock_mean - non_mean).reset_index(name="shock_diff")
    diff = diff.dropna(subset=["shock_diff"]).sort_values("shock_diff", ascending=False).head(int(limit))
    diff["shock_diff"] = diff["shock_diff"].round(6)
    return diff.to_dict("records")


@app.get("/api/dashboard/sales_growth_hist")
def get_dashboard_sales_growth_hist(bins: int = Query(50, ge=10, le=100), clip_lo: float = -0.5, clip_hi: float = 0.5):
    """성장률(sales_growth_qoq) 히스토그램용 빈·카운트. sales_panel 기반."""
    _load_sales_panel()
    if _sales_panel_df is None or _sales_panel_df.empty or "sales_growth_qoq" not in _sales_panel_df.columns:
        return {"bins": [], "counts": []}
    s = _sales_panel_df["sales_growth_qoq"].dropna()
    s = s[(s >= clip_lo) & (s <= clip_hi)]
    if s.empty:
        return {"bins": [], "counts": []}
    counts, bin_edges = np.histogram(s, bins=bins, range=(clip_lo, clip_hi))
    bins_center = [(float(bin_edges[i]) + float(bin_edges[i + 1])) / 2 for i in range(len(bin_edges) - 1)]
    return {"bins": bins_center, "counts": counts.tolist()}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/health")
def api_health():
    """프록시용 헬스체크 (프론트에서 /api/health 호출)."""
    return {"status": "ok"}
