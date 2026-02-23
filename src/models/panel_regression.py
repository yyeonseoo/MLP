from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


@dataclass
class PanelRegressionConfig:
    """
    립스틱 효과 검증용 패널 회귀 설정.

    2분류 모형 (기본)
    -----------------
    log_sales ~ macro_shock * is_lipstick + FE

    3분류 모형 (use_three_way=True)
    --------------------------------
    log(Sales) = β₀ + β₁·Shock + β₂·Lipstick + β₃·Luxury
                 + β₄·(Shock×Lipstick) + β₅·(Shock×Luxury) + FE + ε
    - β₄ > 0 → 립스틱 효과
    - β₅ < 0 → 럭셔리 감소
    """

    dependent: str = "log_sales"
    shock_var: str = "macro_shock"
    lipstick_var: str = "is_lipstick"
    luxury_var: str = "is_luxury"
    region_fe: str = "region_id"
    time_fe: str = "year_quarter"
    use_three_way: bool = True


def prepare_regression_data(df: pd.DataFrame, sales_col: str = "sales") -> pd.DataFrame:
    """회귀용 파생변수 (log_sales 등) 추가."""
    out = df.copy()
    out["log_sales"] = np.log(out[sales_col].replace({0: np.nan}))
    return out.dropna(subset=["log_sales"])


def fit_lipstick_panel_regression(
    df: pd.DataFrame,
    cfg: PanelRegressionConfig | None = None,
):
    """
    립스틱 효과 검증용 패널 회귀.

    use_three_way=True: Shock×Lipstick + Shock×Luxury 동시 포함 (β₄, β₅).
    use_three_way=False: Shock×Lipstick 만 (기존 2분류).
    """
    if cfg is None:
        cfg = PanelRegressionConfig()

    df = prepare_regression_data(df)

    if cfg.use_three_way and cfg.luxury_var in df.columns:
        # 3분류: β₁·Shock + β₂·Lipstick + β₃·Luxury + β₄·(Shock×Lipstick) + β₅·(Shock×Luxury)
        interaction = (
            f"{cfg.shock_var} + {cfg.lipstick_var} + {cfg.luxury_var} + "
            f"{cfg.shock_var}:{cfg.lipstick_var} + {cfg.shock_var}:{cfg.luxury_var}"
        )
    else:
        interaction = f"{cfg.shock_var} * {cfg.lipstick_var}"

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
