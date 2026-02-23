"""
논문/발표용 핵심 시각화 5종 생성.

생성 차트
---------
1. 립스틱 비중 시계열 + 충격 구간(macro_shock=1) 음영
2. 충격기 vs 비충격기 립스틱 비중 박스플롯 (평균/중앙값 표시)
3. 립스틱 vs 논립스틱 평균 성장률 시계열 (2선)
4. RandomForest Feature Importance (상위 10개)
5. 예측 vs 실제 산점도 (45도 선)

실행 전 데이터 필요
------------------
- data/processed/macro_quarterly.csv
- data/processed/lipstick_region_quarter.csv
- data/processed/sales_panel.csv
- data/processed/growth_dataset_train.csv
- data/processed/growth_dataset_test.csv
"""

from __future__ import annotations

from pathlib import Path

# 폰트 설정은 pyplot 사용 전에 적용 (outputs/paper 한글 표시)
from src.config.mpl_font import setup_mpl_font
setup_mpl_font()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.models.growth_models import (
    FeatureConfig,
    build_feature_matrix,
    compute_topk_recall,
)
from sklearn.ensemble import RandomForestRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "paper"


def load_data():
    macro = pd.read_csv(DATA_PROCESSED / "macro_quarterly.csv")
    lipstick = pd.read_csv(DATA_PROCESSED / "lipstick_region_quarter.csv")
    sales_panel = pd.read_csv(DATA_PROCESSED / "sales_panel.csv")
    train_df = pd.read_csv(DATA_PROCESSED / "growth_dataset_train.csv")
    test_df = pd.read_csv(DATA_PROCESSED / "growth_dataset_test.csv")
    return macro, lipstick, sales_panel, train_df, test_df


# ---------------------------------------------------------------------------
# 01b 연속형 충격 점수 시계열 + 상위 25% threshold
# ---------------------------------------------------------------------------
def plot_shock_score_timeseries(macro: pd.DataFrame) -> None:
    """
    shock_score 시계열 + 75% 분위수 기준선 + macro_shock=1 구간 음영.
    (Z-score 기반이면 블록이 아닌 구간별 변동으로 나옴)
    """
    if "shock_score" not in macro.columns:
        print("  shock_score 없음 → 01b 스킵 (패널을 Z-score 기준으로 재생성 후 다시 실행)")
        return
    by_q = (
        macro[["year_quarter", "year", "quarter", "shock_score", "macro_shock"]]
        .sort_values(["year", "quarter"])
        .reset_index(drop=True)
    )
    q75 = by_q["shock_score"].quantile(0.75)
    x = np.arange(len(by_q))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, by_q["shock_score"], color="C0", linewidth=1.5, label="shock_score")
    ax.axhline(q75, color="red", linestyle="--", alpha=0.8, label=f"75% 분위 ({q75:.2f})")
    shock = by_q["macro_shock"].values
    i = 0
    while i < len(shock):
        if shock[i] == 1:
            start = i
            while i < len(shock) and shock[i] == 1:
                i += 1
            ax.axvspan(start - 0.5, i - 0.5, alpha=0.2, color="red")
        else:
            i += 1
    ax.set_xticks(x)
    ax.set_xticklabels(by_q["year_quarter"], rotation=45, ha="right")
    ax.set_xlabel("분기")
    ax.set_ylabel("shock_score (Z-score 복합)")
    ax.set_title("01b 연속형 충격 점수 시계열 (상위 25% = 충격기 음영)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "01b_shock_score_timeseries.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"저장: {OUTPUTS_DIR / '01b_shock_score_timeseries.png'}")


# ---------------------------------------------------------------------------
# ① 립스틱 비중 시계열 + 충격 구간 표시
# ---------------------------------------------------------------------------
def plot_lipstick_share_timeseries(
    macro: pd.DataFrame,
    lipstick: pd.DataFrame,
    sales_panel: pd.DataFrame | None = None,
) -> None:
    """
    x: year_quarter, y: lipstick_share.
    sales_panel이 있으면 전 서울 합산 기준( sum(lipstick sales)/sum(sales) ),
    없으면 상권별 비중의 평균. macro_shock=1 구간 음영.
    """
    if sales_panel is not None and "is_lipstick" in sales_panel.columns and "sales" in sales_panel.columns:
        # 전 서울 합산 기준: 분기별 lipstick_share = sum(sales|is_lipstick) / sum(sales)
        tmp = sales_panel.copy()
        tmp["_sales_lipstick"] = np.where(tmp["is_lipstick"].astype(bool), tmp["sales"], 0.0)
        agg = tmp.groupby(["year", "quarter"], as_index=False).agg(
            sales_total=("sales", "sum"),
            sales_lipstick=("_sales_lipstick", "sum"),
        )
        agg["lipstick_share"] = np.where(
            agg["sales_total"] > 0,
            agg["sales_lipstick"] / agg["sales_total"],
            np.nan,
        )
        agg["year_quarter"] = agg["year"].astype(str) + "Q" + agg["quarter"].astype(str)
        by_q = agg[["year", "quarter", "year_quarter", "lipstick_share"]].copy()
        by_q = by_q.sort_values(["year", "quarter"]).reset_index(drop=True)
        share_col = "lipstick_share"
        ylabel = "립스틱 비중 (전 서울 합산)"
    else:
        # fallback: 상권별 비중의 분기별 평균 (year, quarter 기준 정렬)
        by_q = lipstick.groupby(["year", "quarter"], as_index=False).agg(
            lipstick_share=("lipstick_share", "mean"),
        )
        by_q["year_quarter"] = by_q["year"].astype(str) + "Q" + by_q["quarter"].astype(str)
        by_q = by_q.sort_values(["year", "quarter"]).reset_index(drop=True)
        share_col = "lipstick_share"
        ylabel = "립스틱 비중 (상권 평균)"

    by_q = by_q.merge(
        macro[["year_quarter", "year", "quarter", "macro_shock"]],
        on="year_quarter",
        how="left",
    )
    by_q = by_q.sort_values(["year", "quarter"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(by_q))
    ax.plot(x, by_q[share_col], color="C0", linewidth=2, label="립스틱 비중")

    # macro_shock=1 구간 음영
    shock = by_q["macro_shock"].values
    i = 0
    while i < len(shock):
        if shock[i] == 1:
            start = i
            while i < len(shock) and shock[i] == 1:
                i += 1
            ax.axvspan(start - 0.5, i - 0.5, alpha=0.25, color="red", label="충격기" if start == 0 or shock[start - 1] != 1 else None)
        else:
            i += 1
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    ax.set_xticks(x)
    ax.set_xticklabels(by_q["year_quarter"], rotation=45, ha="right")
    ax.set_xlabel("분기 (year_quarter)")
    ax.set_ylabel(ylabel)
    ax.set_title("① 립스틱 비중 시계열 및 거시 충격 구간")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "01_lipstick_share_timeseries_shock.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"저장: {OUTPUTS_DIR / '01_lipstick_share_timeseries_shock.png'}")


# ---------------------------------------------------------------------------
# ② 충격기 vs 비충격기 립스틱 비중 박스플롯
# ---------------------------------------------------------------------------
def plot_lipstick_share_boxplot(macro: pd.DataFrame, lipstick: pd.DataFrame) -> None:
    """
    x: shock 여부, y: lipstick_share, 평균/중앙값 표시.
    """
    df = lipstick.merge(
        macro[["year", "quarter", "year_quarter", "macro_shock"]],
        on=["year", "quarter", "year_quarter"],
        how="left",
    )
    df["shock_label"] = df["macro_shock"].map({0: "비충격기", 1: "충격기"})

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.boxplot(data=df, x="shock_label", y="lipstick_share", order=["비충격기", "충격기"], ax=ax)

    # 평균/중앙값 표시
    for i, label in enumerate(["비충격기", "충격기"]):
        sub = df.loc[df["shock_label"] == label, "lipstick_share"].dropna()
        if len(sub) > 0:
            med = sub.median()
            mean = sub.mean()
            ax.scatter(i, med, color="darkred", s=60, zorder=5, marker="D", label="중앙값" if i == 0 else None)
            ax.scatter(i, mean, color="orange", s=60, zorder=5, marker="*", label="평균" if i == 0 else None)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.set_xlabel("거시 충격 여부")
    ax.set_ylabel("립스틱 비중")
    ax.set_title("② 충격기 vs 비충격기 립스틱 비중 분포")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "02_lipstick_share_boxplot_shock.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"저장: {OUTPUTS_DIR / '02_lipstick_share_boxplot_shock.png'}")


# ---------------------------------------------------------------------------
# ③ 립스틱 vs 논립스틱 평균 성장률 시계열
# ---------------------------------------------------------------------------
def plot_growth_lipstick_vs_non(sales_panel: pd.DataFrame) -> None:
    """
    분기별로 립스틱 vs 논립스틱 성장률 2선 그래프.
    평균은 outlier에 취약하므로 중앙값(median) 사용.
    """
    growth = sales_panel.replace([np.inf, -np.inf], np.nan).dropna(subset=["sales_growth_qoq"])
    by_q = growth.groupby(["year_quarter", "is_lipstick"], as_index=False).agg(
        sales_growth_median=("sales_growth_qoq", "median"),
    )
    # 정렬용 year, quarter
    q_order = growth[["year_quarter", "year", "quarter"]].drop_duplicates().sort_values(["year", "quarter"])
    by_q = by_q.merge(q_order, on="year_quarter", how="left")

    lip = by_q.loc[by_q["is_lipstick"] == 1].sort_values(["year", "quarter"])
    non = by_q.loc[by_q["is_lipstick"] == 0].sort_values(["year", "quarter"])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lip["year_quarter"], lip["sales_growth_median"], marker="o", label="립스틱 업종", color="C0")
    ax.plot(non["year_quarter"], non["sales_growth_median"], marker="s", label="논립스틱 업종", color="C1")
    ax.set_xlabel("분기")
    ax.set_ylabel("분기 대비 성장률 중앙값 (sales_growth_qoq)")
    ax.set_title("③ 립스틱 vs 논립스틱 업종 성장률 비교 (분기별 중앙값)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "03_growth_lipstick_vs_nonlipstick.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"저장: {OUTPUTS_DIR / '03_growth_lipstick_vs_nonlipstick.png'}")


# ---------------------------------------------------------------------------
# ③b 립스틱 vs 럭셔리 vs 필수 3그룹 성장률 시계열
# ---------------------------------------------------------------------------
def plot_growth_three_groups(sales_panel: pd.DataFrame) -> None:
    """
    3단계 분류: Lipstick(저가 감정 사치) vs Luxury(경기 민감 고가) vs Necessity(필수).
    분기별 중앙값 성장률 3선 그래프. sector_group 필요 (패널 재생성 후).
    """
    if "sector_group" not in sales_panel.columns:
        print("sector_group 없음 → 03b 스킵 (패널 재생성 후 다시 실행)")
        return

    growth = sales_panel.replace([np.inf, -np.inf], np.nan).dropna(subset=["sales_growth_qoq"])
    growth = growth.copy()
    # sector_group: lipstick | luxury | necessity (3분류)
    growth["group_3"] = growth["sector_group"].replace(
        {"lipstick": "Lipstick", "luxury": "Luxury", "necessity": "Necessity"}
    )

    by_q = growth.groupby(["year_quarter", "group_3"], as_index=False).agg(
        sales_growth_median=("sales_growth_qoq", "median"),
    )
    q_order = growth[["year_quarter", "year", "quarter"]].drop_duplicates().sort_values(["year", "quarter"])
    by_q = by_q.merge(q_order, on="year_quarter", how="left")

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, label in enumerate(["Lipstick", "Luxury", "Necessity"]):
        sub = by_q.loc[by_q["group_3"] == label].sort_values(["year", "quarter"])
        if len(sub) == 0:
            continue
        ax.plot(sub["year_quarter"], sub["sales_growth_median"], marker="o" if i == 0 else "s" if i == 1 else "^", label=label, color=f"C{i}")

    ax.set_xlabel("분기")
    ax.set_ylabel("분기 대비 성장률 중앙값")
    ax.set_title("③b 립스틱 vs 럭셔리 vs 필수 업종 성장률 비교 (3단계 분류)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "03b_growth_lipstick_luxury_necessity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"저장: {OUTPUTS_DIR / '03b_growth_lipstick_luxury_necessity.png'}")


# ---------------------------------------------------------------------------
# ④ Feature Importance (RF) & ⑤ 예측 vs 실제
# ---------------------------------------------------------------------------
def train_rf_and_plot(train_df: pd.DataFrame, test_df: pd.DataFrame):
    cfg = FeatureConfig()
    X_train, y_train, feature_cols = build_feature_matrix(train_df, cfg)
    test_subset = test_df.copy()
    for c in feature_cols + [cfg.target_col]:
        if c not in test_subset.columns:
            test_subset[c] = 0.0
    X_test, y_test, _ = build_feature_matrix(
        test_subset[feature_cols + [cfg.target_col]], cfg
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ④ Feature Importance (상위 10개)
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
    top10 = imp.tail(10)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(range(len(top10)), top10.values, color="steelblue", alpha=0.8)
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10.index, fontsize=9)
    ax.set_xlabel("Feature Importance")
    ax.set_title("④ RandomForest Feature Importance (상위 10)")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "04_feature_importance_rf.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"저장: {OUTPUTS_DIR / '04_feature_importance_rf.png'}")

    # ⑤ 예측 vs 실제 산점도
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_test, y_pred, alpha=0.3, s=10, c="C0")
    lims = [
        min(y_test.min(), y_pred.min()),
        max(y_test.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, "k--", lw=2, label="y=x (완벽 예측)")
    ax.set_xlabel("실제 (target_next)")
    ax.set_ylabel("예측값")
    ax.set_title("⑤ 예측 vs 실제 (RandomForest)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "05_predicted_vs_actual_rf.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"저장: {OUTPUTS_DIR / '05_predicted_vs_actual_rf.png'}")

    return model, y_pred, feature_cols


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    macro, lipstick, sales_panel, train_df, test_df = load_data()

    print("01b 연속형 충격 점수 시계열 (shock_score + threshold)")
    plot_shock_score_timeseries(macro)

    print("① 립스틱 비중 시계열 + 충격 구간 (전 서울 합산 기준)")
    plot_lipstick_share_timeseries(macro, lipstick, sales_panel)

    print("② 충격기 vs 비충격기 박스플롯")
    plot_lipstick_share_boxplot(macro, lipstick)

    print("③ 립스틱 vs 논립스틱 성장률 비교")
    plot_growth_lipstick_vs_non(sales_panel)

    print("③b 립스틱 vs 럭셔리 vs 필수 3그룹 성장률")
    plot_growth_three_groups(sales_panel)

    print("④⑤ RF 학습 및 Feature Importance / 예측 vs 실제")
    train_rf_and_plot(train_df, test_df)

    print("\n논문용 핵심 시각화 5개 생성 완료:", OUTPUTS_DIR)


if __name__ == "__main__":
    main()
