from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


@dataclass
class PanelRegressionConfig:
    """
    립스틱 효과 검증용 패널 회귀 설정.

    기본 모형 (예시)
    ---------------
    log_sales ~ macro_shock * is_lipstick
                + C(region_id)  (행정동/상권 고정효과)
                + C(year_quarter) (시점 고정효과)
    """

    dependent: str = "log_sales"
    shock_var: str = "macro_shock"
    lipstick_var: str = "is_lipstick"
    region_fe: str = "region_id"
    time_fe: str = "year_quarter"


def prepare_regression_data(df: pd.DataFrame, sales_col: str = "sales") -> pd.DataFrame:
    """
    회귀용 파생변수 (log_sales 등)를 추가한다.
    """
    out = df.copy()
    out["log_sales"] = np.log(out[sales_col].replace({0: np.nan}))
    return out.dropna(subset=["log_sales"])


def fit_lipstick_panel_regression(
    df: pd.DataFrame,
    cfg: PanelRegressionConfig | None = None,
):
    """
    립스틱 효과 검증용 패널 회귀를 적합하고 결과를 반환한다.

    Returns
    -------
    result : statsmodels RegressionResults
        - 상호작용항 계수(cfg.shock_var:cfg.lipstick_var)가 β₃에 해당
    """
    if cfg is None:
        cfg = PanelRegressionConfig()

    df = prepare_regression_data(df)

    # formula 구성
    # 예: log_sales ~ macro_shock * is_lipstick + C(region_id) + C(year_quarter)
    interaction = f"{cfg.shock_var}*{cfg.lipstick_var}"
    fe_terms = []
    if cfg.region_fe:
        fe_terms.append(f"C({cfg.region_fe})")
    if cfg.time_fe:
        fe_terms.append(f"C({cfg.time_fe})")

    rhs_terms = " + ".join([interaction] + fe_terms)
    formula = f"{cfg.dependent} ~ {rhs_terms}"

    model = smf.ols(formula=formula, data=df)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df[cfg.region_fe]})

    return result

