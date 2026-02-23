"""
전처리 검증 스크립트 (전처리 QA).

패널 생성 후 실행하여 아래 5가지를 점검한다.
- 2-1. 패널 연속성 (분기 누락 시 성장률 왜곡 여부)
- 2-2. sales_prev 작은/0 샘플 비중
- 2-3. 타깃 누수(shift) 검증
- 2-4. 거시 조인 정상 여부
- 2-5. 립스틱 비중 계산 방식 안내

실행: PYTHONPATH=. python scripts/validate_preprocessing.py
선행: scripts/build_panel_and_regression.py (sales_panel.csv 필요)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def load_panel() -> pd.DataFrame:
    p = DATA_PROCESSED / "sales_panel.csv"
    if not p.exists():
        raise FileNotFoundError(f"sales_panel.csv 없음. 먼저 build_panel_and_regression.py 실행: {p}")
    return pd.read_csv(p)


def load_macro() -> pd.DataFrame | None:
    p = DATA_PROCESSED / "macro_quarterly.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def load_growth_dataset():
    train_path = DATA_PROCESSED / "growth_dataset_train.csv"
    test_path = DATA_PROCESSED / "growth_dataset_test.csv"
    if not train_path.exists() or not test_path.exists():
        return None
    return pd.concat([pd.read_csv(train_path), pd.read_csv(test_path)], ignore_index=True)


def check_continuity(df: pd.DataFrame) -> dict:
    """2-1. region_id×sector_code마다 분기가 연속인지. 연속비율 90% 이상 권장, 80% 미만이면 성장률 의심."""
    group_keys = ["region_id", "sector_code"]
    df = df.sort_values(group_keys + ["year", "quarter"])
    df = df.copy()
    df["_t"] = df["year"] * 4 + (df["quarter"] - 1)
    df["_t_prev"] = df.groupby(group_keys, sort=False)["_t"].shift(1)
    df["_contiguous"] = (df["_t"] - df["_t_prev"]) == 1

    total = df["_t_prev"].notna().sum()
    if total == 0:
        return {"ok": False, "ratio": 0.0, "message": "분기 이전 행 없음(연속성 계산 불가)"}
    contiguous = df.loc[df["_t_prev"].notna(), "_contiguous"].sum()
    ratio = contiguous / total
    ok = ratio >= 0.90
    return {
        "ok": ok,
        "ratio": ratio,
        "contiguous_count": int(contiguous),
        "total_with_prev": int(total),
        "message": f"분기 연속 비율 {ratio:.2%} (권장 ≥90%, 80% 미만이면 성장률 왜곡 의심)",
    }


def check_sales_prev(df: pd.DataFrame) -> dict:
    """2-2. sales_prev == 0 및 매우 작은 값 비중."""
    if "sales_prev" not in df.columns:
        return {"ok": False, "message": "sales_prev 컬럼 없음"}
    sp = df["sales_prev"].dropna()
    n = len(sp)
    n_zero = (df["sales_prev"] == 0).sum()
    n_under_100k = (df["sales_prev"].fillna(-1) < 100_000).sum()
    pct_zero = n_zero / len(df) * 100 if len(df) else 0
    pct_under = n_under_100k / len(df) * 100 if len(df) else 0
    ok = pct_zero < 0.1 and pct_under < 50  # 임계치는 조정 가능
    return {
        "ok": ok,
        "n_zero": int(n_zero),
        "n_under_100k": int(n_under_100k),
        "pct_zero": pct_zero,
        "pct_under_100k": pct_under,
        "message": f"sales_prev==0: {n_zero}행({pct_zero:.2f}%), <10만: {n_under_100k}행({pct_under:.1f}%)",
    }


def check_leakage(ds: pd.DataFrame) -> dict | None:
    """2-3. target_next와 현재 분기 성장률 상관. 지나치게 높으면 shift 오류(누수) 의심."""
    if ds is None or "target_next" not in ds.columns:
        return None
    for col in ["sales_growth_log", "sales_growth_qoq"]:
        if col not in ds.columns:
            continue
        valid = ds[[col, "target_next"]].dropna()
        if len(valid) < 10:
            continue
        corr = valid[col].corr(valid["target_next"])
        return {
            "ok": abs(corr) < 0.85,
            "corr": corr,
            "col": col,
            "message": f"corr({col}, target_next) = {corr:.4f} (0.9 근처면 누수 의심)",
        }
    return None


def check_macro_join(df: pd.DataFrame, macro: pd.DataFrame | None) -> dict | None:
    """2-4. 조인 후 macro_shock 결측률, 분기별 동일값 여부."""
    if macro is None or "macro_shock" not in df.columns:
        return None
    miss = df["macro_shock"].isna().sum()
    pct_miss = miss / len(df) * 100 if len(df) else 0
    # 분기별로 상권/업종 무관하게 macro_shock 한 값인지
    by_q = df.groupby(["year", "quarter"])["macro_shock"].nunique()
    all_same = (by_q == 1).all()
    return {
        "ok": pct_miss < 1 and all_same,
        "pct_missing": pct_miss,
        "same_per_quarter": bool(all_same),
        "message": f"macro_shock 결측 {pct_miss:.2f}%, 분기별 동일값 여부: {all_same}",
    }


def main() -> None:
    print("=" * 60)
    print("전처리 검증 (Preprocessing QA)")
    print("=" * 60)

    df = load_panel()
    macro = load_macro()
    ds = load_growth_dataset()

    # 2-1
    print("\n[2-1] 패널 연속성 (분기 누락 시 성장률 왜곡)")
    r1 = check_continuity(df)
    print(f"  {r1['message']}")
    print(f"  → {'PASS' if r1['ok'] else 'WARN'}")

    # 2-2
    print("\n[2-2] sales_prev 작은/0 샘플")
    r2 = check_sales_prev(df)
    print(f"  {r2['message']}")
    print(f"  → {'PASS' if r2['ok'] else 'WARN'}")

    # 2-3
    print("\n[2-3] 타깃 누수(shift) 검증")
    r3 = check_leakage(ds)
    if r3:
        print(f"  {r3['message']}")
        print(f"  → {'PASS' if r3['ok'] else 'WARN'}")
    else:
        print("  growth_dataset 미생성 또는 컬럼 없음 → 스킵 (EDA 후 재실행)")

    # 2-4
    print("\n[2-4] 거시 조인 정상 여부")
    r4 = check_macro_join(df, macro)
    if r4:
        print(f"  {r4['message']}")
        print(f"  → {'PASS' if r4['ok'] else 'WARN'}")
    else:
        print("  macro 또는 macro_shock 없음 → 스킵")

    # 2-5
    print("\n[2-5] 립스틱 비중 계산")
    print("  논문 설득력: 전 서울 합산 sum(sales_lipstick)/sum(sales_total) 또는 매출 가중 평균 사용.")
    print("  시각화 ①는 sales_panel 기반 전 서울 합산으로 계산됨 (plot_paper_visualizations.py).")

    print("\n" + "=" * 60)
    print("검증 완료. WARN이면 전처리/패널 연속성·필터·조인 점검 권장.")
    print("=" * 60)


if __name__ == "__main__":
    main()
