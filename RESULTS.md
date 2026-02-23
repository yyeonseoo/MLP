# 거시경제 충격 하에서 립스틱 효과 분석 — 실행 결과 정리

> 최종 실행일: 2026-02-23

---

## 1. 파이프라인 실행 개요

| 단계 | 스크립트 | 상태 | 비고 |
|------|----------|------|------|
| 거시지표 생성 | `scripts/build_macro_quarterly.py` | ✅ 완료 | 2020Q1~2024Q4 placeholder CSV 생성 |
| 패널 빌드 | `scripts/build_panel_and_regression.py` | ✅ 패널 완료 | 회귀는 대용량으로 인해 별도 실행 권장 |
| EDA | `scripts/eda_lipstick_effect.py` | ✅ 완료 | 4개 섹션 모두 실행 |
| 모델 학습 | `scripts/train_growth_models.py` | ✅ 완료 | Linear, RandomForest 실행 (LightGBM 미실행) |

---

## 2. 데이터 생성 결과

### 2-1. 거시지표 (macro_quarterly.csv)

- **경로**: `data/processed/macro_quarterly.csv`
- **내용**: 2020Q1~2024Q4 분기별 `year`, `quarter`, `cpi`, `policy_rate`, `unemployment`, `ccsi`
- **비고**: 실제 CCSI/CPI xlsx 활용 시 `scripts/build_macro_quarterly.py` 수정 후 재생성 필요

### 2-2. 상권×업종×분기 패널 (sales_panel.csv)

- **경로**: `data/processed/sales_panel.csv`
- **행 수**: 약 40만 행 (상권×업종×분기)
- **주요 컬럼**: `region_id`, `sector_code`, `sector_name`, `year`, `quarter`, `sales`, `transactions`, `sales_growth_qoq`, `is_lipstick`, `macro_shock`, `cpi_yoy`, `ccsi`, `ccsi_diff`, `change_code`, `oper_months_avg`, `close_months_avg` 등
- **병합 데이터**: 거시지표, 상권변화지표(상권 단위) 포함

### 2-3. 립스틱 비중/지수 (lipstick_region_quarter.csv)

- **경로**: `data/processed/lipstick_region_quarter.csv`
- **단위**: 상권 × 분기
- **컬럼**: `lipstick_share`, `lipstick_index_rel`, `sales_total`, `sales_lipstick`, `sales_non_lipstick` 등

### 2-4. 학습용 데이터셋 (growth_dataset_train / test)

- **경로**:
  - `data/processed/growth_dataset_train.csv` — 222,299행 (2020~2022)
  - `data/processed/growth_dataset_test.csv` — 146,112행 (2023~)
- **타깃**: `target_next` = 다음 분기 `sales_growth_qoq`
- **이상치 제거**: 1%~99% 분위수 외 제거 (`[-0.850, 6.590]`) 후 368,411행 유지

---

## 3. EDA 결과

### 3-1. 립스틱 지수 vs 거시 충격 (macro_shock)

| macro_shock | lipstick_share 평균 | 표준편차 | 표본 수 |
|-------------|---------------------|----------|---------|
| 0 (비충격) | 0.1586 | 0.2014 | 15,709 |
| 1 (충격) | 0.1599 | 0.2046 | 15,673 |

- **Welch t-test (립스틱 비중)**  
  - t-stat = 0.5789  
  - p-value = 0.5627  
- **해석**: 충격기/비충격기 간 립스틱 비중 차이는 통계적으로 유의하지 않음. (placeholder 거시지표 사용 시 한계 있음)

### 3-2. sales_growth_qoq 분포

| 지표 | 값 |
|------|-----|
| count | 407,164 |
| mean | 2.86 |
| std | 649.58 |
| min | -1.00 |
| 1% | -0.85 |
| 50% | 0.0005 |
| 95% | 1.27 |
| 99% | 6.59 |
| max | 371,900.74 |

- **이상치 컷 제안 (1%~99% 분위수)**: [-0.850, 6.590]
- **비고**: 극단적 양의 이상치(max 371,900) 존재. 학습 시 위 컷 적용 권장

### 3-3. 거시 변수와 sales_growth_qoq 상관관계

| 변수 | 상관계수 |
|------|----------|
| sales_growth_qoq | 1.0000 |
| policy_rate_diff | 0.0026 |
| policy_rate | 0.0012 |
| ccsi | 0.0009 |
| cpi_yoy | -0.0004 |
| ccsi_diff | -0.0008 |

- **lipstick_share vs ccsi_diff**: 0.0016
- **해석**: 표본에서 거시 변수와 성장률 간 선형 상관은 거의 없음. placeholder 거시지표, 복잡한 비선형 관계 가능성 고려 필요

### 3-4. Shock 더미별 sales_growth_qoq

| macro_shock | 평균 | 표준편차 | 표본 수 |
|-------------|------|----------|---------|
| 0 (비충격) | 3.56 | 859.54 | 218,200 |
| 1 (충격) | 2.06 | 236.82 | 188,964 |

- 충격기에서 평균 성장률·분산 모두 감소 (극단치 영향 가능)

### 3-5. EDA 시각화 파일

- `outputs/eda/lipstick_share_by_shock.png` — shock별 립스틱 비중 boxplot
- `outputs/eda/sales_growth_qoq_hist.png` — 성장률 히스토그램
- `outputs/eda/sales_growth_qoq_log_hist.png` — log(1 + growth) 히스토그램

---

## 4. 예측 모델 성능

### 4-1. 베이스라인 모델

| 모델 | RMSE | MAE | Top-20 Recall |
|------|------|-----|---------------|
| LinearRegression | 325.67 | 299.47 | 0.0000 |
| RandomForest | 0.5314 | 0.2812 | 0.0071 |
| LightGBM | (미실행) | — | — |

- **LinearRegression**: 극단치와 비선형성으로 RMSE가 매우 큼
- **RandomForest**: RMSE 0.53, MAE 0.28 수준으로 상대적으로 양호
- **Top-20 Recall**: year_quarter별 실제 상위 20 vs 예측 상위 20 교집합 비율 평균 — 현재 낮음 (0.71%)

### 4-2. Feature 구성

- **제외**: `region_id`, `sector_code`, `year`, `quarter`, `sales`, `transactions`, `sales_growth_qoq`, `sales_prev`, `transactions_prev`
- **포함**: `txn_growth_qoq`, `macro_shock`, `cpi_yoy`, `ccsi`, `ccsi_diff`, `policy_rate`, `policy_rate_diff`, `is_lipstick`, `oper_months_avg`, `close_months_avg` 등 수치형 컬럼

---

## 5. 패널 회귀 (립스틱 효과 β₃)

- **모형**:  
  `log_sales ~ macro_shock * is_lipstick + C(region_id) + C(year_quarter)`  
- **β₃ 해석**: `macro_shock:is_lipstick` 계수 > 0이면 충격기에 립스틱 업종이 상대적으로 강세
- **실행 비고**: 40만 행 + 상권/시점 고정효과로 인해 실행 시간이 길어 이번 문서 작성 시점에는 미실행. 별도로 아래 명령 실행 권장:

```bash
PYTHONPATH=. python scripts/build_panel_and_regression.py
```

---

## 6. 추천 시스템

- **함수**: `src/recommend/recommender.py` — `recommend_top_n_for_region(model, df_current, region_id, top_k=20)`
- **용도**: 특정 상권에 대해 다음 분기 성장률 예측 상위 20개 업종 추천
- **실제 서비스**: 학습된 모델 저장/로드 및 FastAPI 연동 필요

---

## 7. 실행 순서 요약

```bash
cd /Users/LEEJIWOO/Desktop/new_ml_project

# 1) 거시지표 생성 (처음 1회)
python scripts/build_macro_quarterly.py

# 2) 패널 + 회귀 (실제 CCSI/CPI xlsx 연동 시 build_macro 먼저 수정)
PYTHONPATH=. python scripts/build_panel_and_regression.py

# 3) EDA + 학습용 train/test 생성
PYTHONPATH=. python scripts/eda_lipstick_effect.py

# 4) 성장률 예측 모델 학습
PYTHONPATH=. python scripts/train_growth_models.py
```

---

## 8. 다음 단계 권장

1. **거시지표**: CCSI, CPI xlsx 실제 데이터로 `build_macro_quarterly.py` 수정 후 재생성
2. **립스틱 효과 검증**: 패널 회귀 β₃ 추정 및 해석 (데이터·모형 안정화 후)
3. **모델 튜닝**: LightGBM(libomp 설치), 하이퍼파라미터, feature engineering으로 Top-20 Recall 개선
4. **서비스화**: FastAPI 서버 + React 대시보드 구현
