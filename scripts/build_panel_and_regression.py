from __future__ import annotations

"""
서울시 상권 매출 + 거시지표 패널 생성 및
립스틱 효과 패널 회귀를 한 번에 실행하는 예제 스크립트.

사용 전 준비
-----------
1) 서울시 상권분석서비스(추정매출-상권) CSV를 다음 위치로 복사:

   data/raw/seoul_sales/
     ├─ 서울시_상권분석서비스(추정매출-상권)_2020년.csv
     ├─ 서울시_상권분석서비스(추정매출-상권)_2021년.csv
     ├─ 서울시_상권분석서비스(추정매출-상권)_2022년.csv
     ├─ 서울시_상권분석서비스(추정매출-상권)_2023년.csv
     └─ 서울시_상권분석서비스(추정매출-상권)_2024년.csv

2) 거시지표(소비자물가지수, 기준금리, 실업률, 소비자심리지수 등)를
   Jupyter 노트북 등을 이용해 분기 단위 CSV로 가공:

   data/processed/macro_quarterly.csv
     - year, quarter, cpi, policy_rate, unemployment, ccsi, real_wage ...

3) 실행

   python scripts/build_panel_and_regression.py
"""

from pathlib import Path

import pandas as pd

from src.data.panel_dataset import PanelConfig, build_panel_dataset, save_panel_outputs
from src.models.panel_regression import PanelRegressionConfig, fit_lipstick_panel_regression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    sales_dir = DATA_RAW / "seoul_sales"
    trade_change_dir = DATA_RAW / "seoul_trade_change"
    macro_quarterly_path = DATA_PROCESSED / "macro_quarterly.csv"

    all_sales: list[Path] = []
    for base in (sales_dir, DATA_RAW):
        if not base.exists():
            continue
        for year in ("2020", "2021", "2022", "2023", "2024"):
            all_sales.extend(base.glob(f"*{year}*.csv"))
    sales_paths = sorted(
        set(p for p in all_sales if "상권변화" not in p.name and "행정동" not in p.name),
        key=lambda p: p.name,
    )
    if not sales_paths:
        raise FileNotFoundError(
            f"상권 매출 CSV가 없습니다. data/raw/ 또는 data/raw/seoul_sales/ 확인하세요."
        )
    if not macro_quarterly_path.exists():
        raise FileNotFoundError(
            f"거시지표 macro_quarterly.csv 가 없습니다: {macro_quarterly_path}"
        )

    # 상권변화지표-상권 (선택)
    trade_change_area_path = None
    candidate = trade_change_dir / "서울시_상권분석서비스(상권변화지표-상권).csv"
    if candidate.exists():
        trade_change_area_path = str(candidate)
    else:
        fallback = DATA_RAW / "서울시_상권분석서비스(상권변화지표-상권).csv"
        if fallback.exists():
            trade_change_area_path = str(fallback)

    # 1) 패널 데이터 생성 (충격기 = Z-score 기반 shock_score 상위 25%. 고정 구간 쓰려면 shock_periods=COVID_SHOCK_QUARTERS)
    cfg = PanelConfig(
        sales_paths=[str(p) for p in sales_paths],
        macro_quarterly_path=str(macro_quarterly_path),
        trade_change_area_path=trade_change_area_path,
        shock_periods=None,
    )
    outputs = build_panel_dataset(cfg)

    # 2) CSV로 저장
    save_panel_outputs(outputs, DATA_PROCESSED)

    sales_panel: pd.DataFrame = outputs["sales_panel"]

    # 3) 립스틱 효과 패널 회귀 (3분류: Shock×Lipstick, Shock×Luxury)
    reg_cfg = PanelRegressionConfig(use_three_way=True)
    result = fit_lipstick_panel_regression(sales_panel, cfg=reg_cfg)

    print(result.summary())
    print()
    print("=== 핵심 계수 (β₄: Shock×Lipstick, β₅: Shock×Luxury) ===")
    term_sl = f"{reg_cfg.shock_var}:{reg_cfg.lipstick_var}"
    term_slux = f"{reg_cfg.shock_var}:{reg_cfg.luxury_var}"
    if term_sl in result.params:
        print(f"  {term_sl} (β₄, 립스틱 효과 기대 > 0) = {result.params[term_sl]:.4f}")
    if term_slux in result.params:
        print(f"  {term_slux} (β₅, 럭셔리 감소 기대 < 0) = {result.params[term_slux]:.4f}")


if __name__ == "__main__":
    main()

