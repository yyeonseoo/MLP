# 전망 모델 비교 — 총 정리

> 동일 태스크(다음 분기 log_sales 예측), 동일 시간 분할(2023~ test) 기준.  
> 회귀 지표(MAE, RMSE) + 분류 지표(정확도, F1) 포함.

---

## 1. 태스크 정의

| 항목 | 내용 |
|------|------|
| **타깃** | 다음 분기 로그 매출 `log(sales+1)` (회귀) |
| **분류 보조** | "다음 분기 매출 > 당분기 매출" 여부(이진) → Accuracy, F1 |
| **입력** | 상권×업종×분기 패널, 동일 피처(거시·lag·rolling·quarter·target encoding 등) |
| **Train** | 2022년 이하 |
| **Test** | 2023년 1분기 이후 |
| **회귀 지표** | MAE, RMSE (낮을수록 좋음) |
| **분류 지표** | Accuracy, F1 (높을수록 좋음) |

---

## 2. 모델별 성능 (전체)

| 순서 | 모델 | MAE | RMSE | Accuracy | F1 | 비고 |
|------|------|-----|------|----------|-----|------|
| — | **LightGBM** | 0.2731 | 0.4972 | 0.6306 | 0.6419 | 단일 시점 회귀 (baseline) |
| 1 | **CatBoost** | 0.2761 | 0.4990 | 0.6212 | 0.6379 | GBDT |
| 2 | **앙상블 (LGBM+CatBoost)** | **0.2728** | **0.4963** | 0.6285 | **0.6431** | 두 예측 평균 — 회귀 최고 |
| 3 | **XGBoost** | 0.2746 | 0.4986 | 0.6266 | 0.6421 | GBDT |
| 4 | **HistGradientBoosting** | 0.2747 | 0.4982 | 0.6282 | 0.6401 | sklearn 히스토그램 GBDT |
| 5 | **Transformer** | 0.6311 | 0.7984 | 0.5701 | 0.4142 | 과거 4분기 시퀀스 |

- GBDT/앙상블: `models/forecast_compare_results.json`  
- Transformer: `models/forecast_transformer_metrics.json`

---

## 3. 성능 순위

### 3-1. 회귀 (MAE 기준, 낮을수록 좋음)

| 순위 | 모델 | MAE | RMSE |
|------|------|-----|------|
| 1 | **앙상블 (LGBM + CatBoost)** | **0.2728** | **0.4963** |
| 2 | LightGBM | 0.2731 | 0.4972 |
| 3 | XGBoost | 0.2746 | 0.4986 |
| 4 | HistGradientBoosting | 0.2747 | 0.4982 |
| 5 | CatBoost | 0.2761 | 0.4990 |
| 6 | Transformer | 0.6311 | 0.7984 |

### 3-2. 분류 (Accuracy / F1, 높을수록 좋음)

| 순위 | 모델 | Accuracy | F1 |
|------|------|----------|-----|
| 1 | **앙상블 (LGBM+CatBoost)** | 0.6285 | **0.6431** |
| 2 | LightGBM | **0.6306** | 0.6419 |
| 3 | XGBoost | 0.6266 | 0.6421 |
| 4 | HistGradientBoosting | 0.6282 | 0.6401 |
| 5 | CatBoost | 0.6212 | 0.6379 |
| 6 | Transformer | 0.5701 | 0.4142 |

- **정확도·F1**은 회귀 예측값을 "다음 분기 매출 > 당분기 매출" 여부로 이진 분류했을 때의 결과임.  
- **F1 최고**: 앙상블(0.6431). **정확도 최고**: LightGBM(0.6306).  
- Transformer는 회귀·분류 모두 다른 모델보다 낮음.

---

## 4. 요약

- **LightGBM보다 좋은 모델(회귀)**: **앙상블 (LightGBM + CatBoost)** — MAE/RMSE 최소.
- **정확도·F1**: GBDT/앙상블이 0.62~0.64 수준, Transformer는 0.57 / 0.41로 낮음.
- **권장**: 배포 시 **앙상블(LGBM+CatBoost)** 또는 LightGBM 단독 사용. Transformer는 현재 설정에서 비권장.

---

## 5. 재현 방법

```bash
# LightGBM, CatBoost, 앙상블, XGBoost, HistGradientBoosting (MAE/RMSE + Accuracy/F1)
PYTHONPATH=. python scripts/train_forecast_compare_models.py

# Transformer (별도 실행)
PYTHONPATH=. python scripts/train_forecast_transformer.py
```

- 비교 결과: `models/forecast_compare_results.json`  
- Transformer: `models/forecast_transformer_metrics.json`

---

## 6. 모델 구성 요약

| 모델 | 스크립트 | 입력 형태 |
|------|----------|-----------|
| LightGBM | `train_forecast_compare_models.py` | 1행 피처 벡터 |
| CatBoost | 동일 | 동일 |
| 앙상블 | 동일 (LGBM + CatBoost 예측 평균) | — |
| XGBoost | 동일 | 동일 |
| HistGradientBoosting | 동일 (sklearn) | 동일 |
| Transformer | `train_forecast_transformer.py` | 과거 4분기 시퀀스 (seq_len=4) |
