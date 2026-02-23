from __future__ import annotations

"""
립스틱 효과 EDA 및 학습용 데이터셋 생성 스크립트.

내용
----
1) 립스틱 지수(lipstick_share, lipstick_index_rel)가
   거시 충격기(macro_shock=1)에서 실제로 상승하는지 확인
   - 충격기 vs 비충격기 평균
   - boxplot
   - t-test

2) sales_growth_qoq 분포 확인
   - 분포 요약 통계
   - 히스토그램
   - 이상치 컷(예: 1%, 99%) 제안

3) 거시 변수 상관관계
   - cpi_yoy vs sales_growth_qoq
   - ccsi_diff vs lipstick_share
   - shock 더미별 평균 비교

4) 학습용 데이터셋 분리
   - X_t 로부터 y_{t+1}=sales_growth_qoq(t+1)를 예측하는 데이터셋 생성
   - 시간 기준(train: 2020~2022, test: 2023~)
   - data/processed/ 아래에 train/test CSV로 저장
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

from src.models.datasets import (
    make_next_quarter_growth_dataset,
    suggest_growth_outlier_bounds,
    temporal_train_test_split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "eda"


def load_processed_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sales_panel_path = DATA_PROCESSED / "sales_panel.csv"
    macro_path = DATA_PROCESSED / "macro_quarterly.csv"
    lipstick_path = DATA_PROCESSED / "lipstick_region_quarter.csv"

    if not sales_panel_path.exists():
        raise FileNotFoundError(
            f"sales_panel.csv 가 없습니다. 먼저 scripts/build_panel_and_regression.py 를 실행하세요: {sales_panel_path}"
        )
    if not macro_path.exists() or not lipstick_path.exists():
        raise FileNotFoundError(
            f"macro_quarterly.csv 또는 lipstick_region_quarter.csv 가 없습니다. build_panel_and_regression 스크립트를 먼저 실행하세요."
        )

    sales_panel = pd.read_csv(sales_panel_path)
    macro = pd.read_csv(macro_path)
    lipstick = pd.read_csv(lipstick_path)

    return sales_panel, macro, lipstick


def eda_lipstick_vs_shock(macro: pd.DataFrame, lipstick: pd.DataFrame) -> None:
    """
    립스틱 지수가 거시 충격기에 어떻게 변하는지 분석.
    """
    df = lipstick.merge(
        macro[["year", "quarter", "year_quarter", "macro_shock"]],
        on=["year", "quarter", "year_quarter"],
        how="left",
        validate="m:1",
    )

    print("\n[1] 립스틱 지수 - 충격기 vs 비충격기 평균")
    summary = df.groupby("macro_shock")["lipstick_share"].agg(["mean", "std", "count"])
    print(summary)

    # t-test (Welch)
    shock = df.loc[df["macro_shock"] == 1, "lipstick_share"].dropna()
    non_shock = df.loc[df["macro_shock"] == 0, "lipstick_share"].dropna()
    if len(shock) > 5 and len(non_shock) > 5:
        t_stat, p_val = stats.ttest_ind(shock, non_shock, equal_var=False)
        print("\nWelch t-test (shock=1 vs shock=0, lipstick_share)")
        print(f"t-stat = {t_stat:.4f}, p-value = {p_val:.4g}")
    else:
        print("\n샘플 수가 부족하여 t-test를 수행하지 않았습니다.")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.boxplot(
        data=df,
        x="macro_shock",
        y="lipstick_share",
    )
    plt.xlabel("Macro shock (0=비충격, 1=충격)")
    plt.ylabel("Lipstick share (상권 내 립스틱 업종 매출 비중)")
    plt.title("Shock 여부에 따른 립스틱 비중 분포")
    plt.tight_layout()
    out_path = OUTPUTS_DIR / "lipstick_share_by_shock.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"boxplot 저장: {out_path}")


def eda_sales_growth_distribution(sales_panel: pd.DataFrame) -> tuple[float, float]:
    """
    sales_growth_qoq 분포 및 이상치 컷 제안.
    """
    print("\n[2] sales_growth_qoq 분포 요약")
    growth = sales_panel["sales_growth_qoq"].replace([float("inf"), float("-inf")], pd.NA)
    desc = growth.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
    print(desc)

    lower, upper = suggest_growth_outlier_bounds(growth, lower_q=0.01, upper_q=0.99)
    print(f"\n이상치 컷 제안 (1%~99% 분위수 기준): [{lower:.3f}, {upper:.3f}]")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    sns.histplot(growth.dropna(), bins=100, kde=True)
    plt.xlabel("sales_growth_qoq")
    plt.title("분기별 매출 성장률 분포 (원자료)")
    plt.tight_layout()
    out_path = OUTPUTS_DIR / "sales_growth_qoq_hist.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"히스토그램 저장: {out_path}")

    # log1p 변환 분포도 참고용
    valid = growth.dropna()
    shifted = 1.0 + valid
    shifted = shifted[shifted > 0]
    log_growth = np.log(shifted)

    plt.figure(figsize=(7, 4))
    sns.histplot(log_growth, bins=100, kde=True)
    plt.xlabel("log(1 + sales_growth_qoq)")
    plt.title("분기별 매출 성장률 분포 (log1p 변환)")
    plt.tight_layout()
    out_path2 = OUTPUTS_DIR / "sales_growth_qoq_log_hist.png"
    plt.savefig(out_path2, dpi=150)
    plt.close()
    print(f"log 분포 히스토그램 저장: {out_path2}")

    return lower, upper


def eda_macro_correlations(sales_panel: pd.DataFrame, macro: pd.DataFrame, lipstick: pd.DataFrame) -> None:
    """
    거시 변수와 성장률/립스틱 지수의 상관관계 확인.
    """
    print("\n[3] 거시 변수와 성장률/립스틱 지수 상관관계")

    # cpi_yoy vs sales_growth_qoq
    cols = ["cpi_yoy", "ccsi", "ccsi_diff", "policy_rate", "policy_rate_diff"]
    cols = [c for c in cols if c in sales_panel.columns]
    if "sales_growth_qoq" in sales_panel.columns:
        corr_mat = sales_panel[["sales_growth_qoq"] + cols].corr()
        print("\n상관계수 (sales_growth_qoq vs macro variables):")
        print(corr_mat["sales_growth_qoq"].sort_values(ascending=False))

    # ccsi_diff vs lipstick_share (상권 x 분기)
    lip_macro = lipstick.merge(
        macro[["year", "quarter", "year_quarter", "ccsi_diff", "macro_shock"]],
        on=["year", "quarter", "year_quarter"],
        how="left",
        validate="m:1",
    )
    if "ccsi_diff" in lip_macro.columns:
        corr_lip = lip_macro[["lipstick_share", "ccsi_diff"]].corr().iloc[0, 1]
        print(f"\n상관계수 (lipstick_share vs ccsi_diff): {corr_lip:.4f}")

    # shock 더미별 평균
    if "macro_shock" in sales_panel.columns:
        grp = sales_panel.groupby("macro_shock")["sales_growth_qoq"].agg(["mean", "std", "count"])
        print("\nShock 더미별 sales_growth_qoq 평균/표준편차/표본수:")
        print(grp)


def build_and_save_datasets(
    sales_panel: pd.DataFrame,
    lower_cut: float,
    upper_cut: float,
) -> None:
    """
    다음 분기 성장률 예측용 데이터셋 생성 및 train/test 분리.
    """
    print("\n[4] 다음 분기 성장률 예측용 데이터셋 생성")

    ds = make_next_quarter_growth_dataset(sales_panel)

    # 이상치 제거 (제안 컷 적용)
    ds_filtered = ds[
        (ds["sales_growth_qoq"] >= lower_cut)
        & (ds["sales_growth_qoq"] <= upper_cut)
        & (ds["target_next"] >= lower_cut)
        & (ds["target_next"] <= upper_cut)
    ].copy()

    print(f"전체 샘플 수: {len(ds)}, 이상치 제거 후: {len(ds_filtered)}")

    # 시간 기준 train/test split (예: 2023Q1 이상을 test로 사용)
    train, test = temporal_train_test_split(ds_filtered, test_start_year=2023, test_start_quarter=1)
    print(f"train 샘플 수: {len(train)}, test 샘플 수: {len(test)}")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    train_path = DATA_PROCESSED / "growth_dataset_train.csv"
    test_path = DATA_PROCESSED / "growth_dataset_test.csv"

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"train 저장: {train_path}")
    print(f"test 저장: {test_path}")


def main() -> None:
    sales_panel, macro, lipstick = load_processed_tables()

    eda_lipstick_vs_shock(macro, lipstick)
    lower, upper = eda_sales_growth_distribution(sales_panel)
    eda_macro_correlations(sales_panel, macro, lipstick)
    build_and_save_datasets(sales_panel, lower_cut=lower, upper_cut=upper)


if __name__ == "__main__":
    main()

