from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pandas as pd

# 코로나 거시 충격 구간 (기간 지정 시 사용). 2020Q1 ~ 2022Q1
COVID_SHOCK_QUARTERS: list[str] = (
    [f"{y}Q{q}" for y in (2020, 2021) for q in (1, 2, 3, 4)] + ["2022Q1"]
)

from src.data.macro_quarterly import MacroConfig, add_macro_derivatives, load_macro_quarterly
from src.data.seoul_sales import (
    SalesConfig,
    add_growth_features,
    compute_lipstick_index_relative_growth,
    compute_lipstick_share_by_region_quarter,
    load_seoul_sales,
)
from src.data.shock_index import compute_shock_index_from_sales, merge_shock_index_into_macro
from src.data.trade_area_change import TradeAreaChangeConfig, load_trade_change_by_area


@dataclass
class PanelConfig:
    """
    패널 데이터 생성 전체 설정.

    Attributes
    ----------
    sales_paths:
        2020~2024년 서울시 상권 추정매출 CSV 경로 리스트.
    macro_quarterly_path:
        분기 단위 거시지표 CSV 경로.
    trade_change_area_path:
        상권변화지표-상권 CSV 경로 (선택).
    """

    sales_paths: Sequence[str]
    macro_quarterly_path: str
    trade_change_area_path: str | None = None
    sales_config: SalesConfig = field(default_factory=SalesConfig)
    macro_config: MacroConfig = field(default_factory=MacroConfig)
    trade_change_config: TradeAreaChangeConfig = field(default_factory=TradeAreaChangeConfig)
    # 충격기 구간. None이면 quantile 기반, 지정하면 해당 분기만 충격기(1). 예: COVID_SHOCK_QUARTERS
    shock_periods: list[tuple[int, int]] | list[str] | None = None
    # True면 매출 기반 충격지수(분기합계→업종별 증감→상대 증감→Z-score)로 shock_score/macro_shock 덮어씀
    use_sales_shock_index: bool = True


def build_panel_dataset(cfg: PanelConfig) -> dict[str, pd.DataFrame]:
    """
    상권 매출 + 거시지표 + 립스틱 지수/성장률을 포함한 핵심 테이블들을 생성한다.

    Returns
    -------
    dict with keys:
      - sales_panel : 상권 x 업종 x 분기 패널 (성장률 포함)
      - macro_quarterly : 분기별 거시지표 (파생변수 포함)
      - lipstick_region_quarter : 상권 x 분기 립스틱 비중 / 지수
    """
    # 1) 매출 데이터
    sales = load_seoul_sales(cfg.sales_paths, config=cfg.sales_config)
    sales = add_growth_features(sales)

    # 2) 거시지표
    macro = load_macro_quarterly(cfg.macro_quarterly_path, config=cfg.macro_config)
    macro = add_macro_derivatives(
        macro, config=cfg.macro_config, shock_periods=cfg.shock_periods
    )

    # 2-1) 매출 기반 거시경제 충격지수 (분기별 매출 합계 → 업종별 증감률 → 전체 대비 상대 증감 → Z-score)
    if cfg.use_sales_shock_index:
        try:
            shock_quarterly = compute_shock_index_from_sales(sales)
            macro = merge_shock_index_into_macro(macro, shock_quarterly, binary_quantile=0.75)
        except Exception as e:
            import warnings
            warnings.warn(f"매출 기반 충격지수 계산 실패, 거시 변수 기준 유지: {e}")

    # 3) 립스틱 지수 (상권 x 분기)
    lipstick_share = compute_lipstick_share_by_region_quarter(sales)
    lipstick_idx = compute_lipstick_index_relative_growth(lipstick_share)

    # 4) 매출 패널과 거시지표 병합 (연-분기 기준)
    sales_panel = sales.merge(
        macro,
        on=["year", "quarter", "year_quarter"],
        how="left",
        validate="m:1",
    )

    # 5) 상권변화지표(상권 단위) 병합 (선택)
    trade_change_area = None
    if cfg.trade_change_area_path is not None:
        trade_change_area = load_trade_change_by_area(
            cfg.trade_change_area_path,
            config=cfg.trade_change_config,
        )
        sales_panel = sales_panel.merge(
            trade_change_area[
                [
                    "year",
                    "quarter",
                    "year_quarter",
                    "region_id",
                    "change_code",
                    "change_name",
                    "oper_months_avg",
                    "close_months_avg",
                    "seoul_oper_months_avg",
                    "seoul_close_months_avg",
                ]
            ],
            on=["year", "quarter", "year_quarter", "region_id"],
            how="left",
            validate="m:1",
        )

    outputs: dict[str, pd.DataFrame] = {
        "sales_panel": sales_panel,
        "macro_quarterly": macro,
        "lipstick_region_quarter": lipstick_idx,
    }
    if trade_change_area is not None:
        outputs["trade_change_area"] = trade_change_area

    return outputs


def save_panel_outputs(
    outputs: dict[str, pd.DataFrame],
    out_dir: str | Path,
) -> None:
    """
    패널 생성 결과를 CSV로 저장한다.

    Parameters
    ----------
    outputs:
        build_panel_dataset() 반환 딕셔너리.
    out_dir:
        저장 디렉토리 (예: data/processed).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, df in outputs.items():
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False)

