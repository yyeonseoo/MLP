"""
거시 충격(macro_shock, shock_score) 및 사용자용 위험도(shock_score_ui) 진단.

- (a) shock_score, shock_score_ui_100 테이블 출력
- (b) 요인별 기여도(contrib) 막대
- (c) 원시지표 4개 미니 시계열 (cpi_yoy, policy_rate_diff, ccsi_diff, unemployment_diff)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config.mpl_font import setup_mpl_font
from src.data.macro_quarterly import (
    add_macro_derivatives,
    compute_shock_score_ui,
    load_macro_quarterly,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "macro_shock"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

setup_mpl_font()


def main() -> None:
    path = DATA_PROCESSED / "macro_quarterly.csv"
    if not path.exists():
        print(f"파일 없음: {path}")
        return

    macro = load_macro_quarterly(path)
    macro = add_macro_derivatives(macro, shock_periods=None)
    macro = compute_shock_score_ui(macro, scale_01="percentile")
    macro = macro.sort_values(["year", "quarter"]).reset_index(drop=True)

    # ---- (a) shock_score / shock_score_ui_100 테이블 ----
    print("=== (a) shock_score vs 사용자용 위험도(0~100) ===")
    cols = [
        "year_quarter",
        "policy_rate",
        "policy_rate_diff",
        "ccsi",
        "cpi_yoy",
        "shock_score",
        "shock_score_ui_100",
        "macro_shock",
    ]
    available = [c for c in cols if c in macro.columns]
    print(macro[available].to_string(index=False))
    print()
    n = len(macro)
    n_shock = macro["macro_shock"].sum()
    print(f"총 분기: {n}, macro_shock=1: {n_shock} ({100 * n_shock / n:.1f}%)")
    if "shock_score_ui_100" in macro.columns:
        print(f"shock_score_ui_100 범위: {macro['shock_score_ui_100'].min():.1f} ~ {macro['shock_score_ui_100'].max():.1f}")
    print()

    # ---- (b) 요인별 기여도 막대 (분기별) ----
    contrib_cols = [c for c in ["contrib_ccsi", "contrib_cpi", "contrib_rate", "contrib_unemp"] if c in macro.columns]
    if contrib_cols:
        fig, ax = plt.subplots(figsize=(10, 4))
        x = range(len(macro))
        width = 0.7
        bottom = pd.Series(0.0, index=macro.index)
        colors = ["#3b82f6", "#ef4444", "#eab308", "#22c55e"]
        for i, col in enumerate(contrib_cols):
            label = col.replace("contrib_", "").upper()
            if col == "contrib_ccsi":
                label = "ΔCCSI(하락)"
            elif col == "contrib_cpi":
                label = "CPI YoY"
            elif col == "contrib_rate":
                label = "Δ금리"
            elif col == "contrib_unemp":
                label = "Δ실업"
            ax.bar(x, macro[col].values, width=width, bottom=bottom.values, label=label, color=colors[i % len(colors)], alpha=0.85)
            bottom = bottom + macro[col].values
        ax.set_xticks(x)
        ax.set_xticklabels(macro["year_quarter"].values, rotation=45, ha="right")
        ax.set_ylabel("기여도 (악화 방향 Z-score)")
        ax.set_title("(b) 사용자용 거시 위험도 요인별 기여도")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUTS_DIR / "contrib_bars.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"저장: {OUTPUTS_DIR / 'contrib_bars.png'}")

    # ---- (c) 원시지표 4개 미니 시계열 ----
    series_cols = []
    if "cpi_yoy" in macro.columns:
        series_cols.append(("cpi_yoy", "CPI 전년비", "#22c55e"))
    if "policy_rate_diff" in macro.columns:
        series_cols.append(("policy_rate_diff", "Δ기준금리", "#3b82f6"))
    if "ccsi_diff" in macro.columns:
        series_cols.append(("ccsi_diff", "ΔCCSI", "#f59e0b"))
    if "unemployment_diff" in macro.columns:
        series_cols.append(("unemployment_diff", "Δ실업률", "#ef4444"))

    if series_cols:
        n_axes = len(series_cols)
        fig, axes = plt.subplots(n_axes, 1, figsize=(10, 2 * n_axes), sharex=True)
        if n_axes == 1:
            axes = [axes]
        x = macro["year_quarter"].values
        for ax, (col, title, color) in zip(axes, series_cols):
            ax.plot(x, macro[col].values, marker="o", markersize=3, color=color, label=title)
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("분기")
        axes[0].set_title("(c) 원시 지표 시계열")
        for ax in axes:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(OUTPUTS_DIR / "raw_indicators_mini.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"저장: {OUTPUTS_DIR / 'raw_indicators_mini.png'}")

    # ---- 사용자용 위험도 시계열 (0~100) ----
    if "shock_score_ui_100" in macro.columns:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.fill_between(range(len(macro)), macro["shock_score_ui_100"].values, alpha=0.3, color="#dc2626")
        ax.plot(range(len(macro)), macro["shock_score_ui_100"].values, marker="o", markersize=4, color="#dc2626", label="거시 위험도 (0~100)")
        ax.set_xticks(range(len(macro)))
        ax.set_xticklabels(macro["year_quarter"].values, rotation=45, ha="right")
        ax.set_ylabel("위험도 (0~100)")
        ax.set_title("사용자용 거시 위험도 (percentile 기반)")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUTS_DIR / "shock_score_ui_100.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"저장: {OUTPUTS_DIR / 'shock_score_ui_100.png'}")

    # ---- 기존 진단: OR 조건 등 ----
    if "policy_rate_diff" in macro.columns and "cpi_yoy" in macro.columns:
        m = macro.copy()
        q_ccsi = m["ccsi"].quantile(0.25)
        q_cpi = m["cpi_yoy"].quantile(0.75)
        q_dr = m["policy_rate_diff"].dropna().quantile(0.75)
        m["cond_ccsi"] = (m["ccsi"] <= q_ccsi).astype(int)
        m["cond_cpi"] = (m["cpi_yoy"].fillna(-1e9) >= q_cpi).astype(int)
        m["cond_dr"] = (m["policy_rate_diff"].fillna(-1e9) >= q_dr).astype(int)
        print("\n=== OR 조건별 True 개수 (참고) ===")
        print("cond_ccsi (CCSI≤Q25):", m["cond_ccsi"].sum(), "| cond_cpi (CPI YoY≥Q75):", m["cond_cpi"].sum(), "| cond_dr (Δ금리≥Q75):", m["cond_dr"].sum())


if __name__ == "__main__":
    main()
