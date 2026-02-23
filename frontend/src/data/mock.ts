/**
 * 목업 데이터 (실제 API 연동 시 교체)
 * Python 파이프라인 출력과 동일한 구조로 맞춤.
 *
 * 충격기(macro_shock=1): 코로나 거시 충격 구간 2020Q1~2022Q1 기준.
 * (Python에서는 shock_periods=COVID_SHOCK_QUARTERS 로 동일 구간 사용)
 */

export interface MacroQuarter {
  year_quarter: string;
  year: number;
  quarter: number;
  cpi: number;
  policy_rate: number;
  ccsi: number;
  cpi_yoy: number;
  ccsi_diff: number;
  policy_rate_diff: number;
  macro_shock: number;
  /** 연속형 충격 점수 (0~1, 높을수록 거시 스트레스 큼). Z-score 기반 복합 지표 목업 */
  shock_score: number;
}

export interface LipstickSeriesPoint {
  year_quarter: string;
  lipstick_share: number;
  macro_shock: number;
}

export interface GrowthComparisonPoint {
  year_quarter: string;
  lipstick_median: number;
  non_lipstick_median: number;
}

export interface SensitivityRankItem {
  region_name: string;
  region_id: string;
  shock_diff: number; // 충격기 평균 lipstick_share - 비충격기 평균
}

export interface Top20Item {
  sector_name: string;
  sector_code: string;
  predicted_growth: number;
  is_lipstick: boolean;
  rank: number;
  rationale?: string;
}

// 시계열 (립스틱 비중 + 충격 구간)
export const lipstickSeries: LipstickSeriesPoint[] = [
  { year_quarter: "2020Q1", lipstick_share: 0.152, macro_shock: 1 },
  { year_quarter: "2020Q2", lipstick_share: 0.155, macro_shock: 1 },
  { year_quarter: "2020Q3", lipstick_share: 0.157, macro_shock: 1 },
  { year_quarter: "2020Q4", lipstick_share: 0.158, macro_shock: 1 },
  { year_quarter: "2021Q1", lipstick_share: 0.159, macro_shock: 1 },
  { year_quarter: "2021Q2", lipstick_share: 0.160, macro_shock: 1 },
  { year_quarter: "2021Q3", lipstick_share: 0.158, macro_shock: 1 },
  { year_quarter: "2021Q4", lipstick_share: 0.161, macro_shock: 1 },
  { year_quarter: "2022Q1", lipstick_share: 0.162, macro_shock: 1 },
  { year_quarter: "2022Q2", lipstick_share: 0.159, macro_shock: 0 },
  { year_quarter: "2022Q3", lipstick_share: 0.160, macro_shock: 0 },
  { year_quarter: "2022Q4", lipstick_share: 0.158, macro_shock: 0 },
  { year_quarter: "2023Q1", lipstick_share: 0.161, macro_shock: 0 },
  { year_quarter: "2023Q2", lipstick_share: 0.159, macro_shock: 0 },
  { year_quarter: "2023Q3", lipstick_share: 0.160, macro_shock: 0 },
  { year_quarter: "2023Q4", lipstick_share: 0.162, macro_shock: 0 },
  { year_quarter: "2024Q1", lipstick_share: 0.161, macro_shock: 0 },
  { year_quarter: "2024Q2", lipstick_share: 0.163, macro_shock: 0 },
];

// 립스틱 vs 논립스틱 성장률 비교
export const growthComparison: GrowthComparisonPoint[] = lipstickSeries.map((p, i) => ({
  year_quarter: p.year_quarter,
  lipstick_median: 0.02 + (i % 3) * 0.01,
  non_lipstick_median: 0.01 + (i % 4) * 0.005,
}));

// 충격 민감도 행정동(상권) 랭킹 Top 20
export const sensitivityRanking: SensitivityRankItem[] = [
  { region_id: "1", region_name: "강남역 상권", shock_diff: 0.042 },
  { region_id: "2", region_name: "홍대입구역 상권", shock_diff: 0.038 },
  { region_id: "3", region_name: "잠실역 상권", shock_diff: 0.035 },
  { region_id: "4", region_name: "신촌역 상권", shock_diff: 0.031 },
  { region_id: "5", region_name: "이태원역 상권", shock_diff: 0.028 },
  { region_id: "6", region_name: "건대입구역 상권", shock_diff: 0.026 },
  { region_id: "7", region_name: "신림역 상권", shock_diff: 0.024 },
  { region_id: "8", region_name: "을지로3가역 상권", shock_diff: 0.022 },
  { region_id: "9", region_name: "왕십리역 상권", shock_diff: 0.020 },
  { region_id: "10", region_name: "구로디지털단지역 상권", shock_diff: 0.018 },
  { region_id: "11", region_name: "사당역 상권", shock_diff: 0.016 },
  { region_id: "12", region_name: "노원역 상권", shock_diff: 0.015 },
  { region_id: "13", region_name: "김포공항역 상권", shock_diff: 0.014 },
  { region_id: "14", region_name: "역삼역 상권", shock_diff: 0.013 },
  { region_id: "15", region_name: "선릉역 상권", shock_diff: 0.012 },
  { region_id: "16", region_name: "서울대입구역 상권", shock_diff: 0.011 },
  { region_id: "17", region_name: "영등포역 상권", shock_diff: 0.010 },
  { region_id: "18", region_name: "신도림역 상권", shock_diff: 0.009 },
  { region_id: "19", region_name: "천호역 상권", shock_diff: 0.008 },
  { region_id: "20", region_name: "수유역 상권", shock_diff: 0.007 },
];

// TOP-20 추천 (다음 분기)
export const top20Recommendations: Top20Item[] = [
  { sector_name: "화장품", sector_code: "S1", predicted_growth: 0.082, is_lipstick: true, rank: 1, rationale: "거래건수 성장률·거시 충격 구간에서 상대 강세" },
  { sector_name: "네일숍", sector_code: "S2", predicted_growth: 0.078, is_lipstick: true, rank: 2, rationale: "소규모 감정 소비 수요 안정" },
  { sector_name: "커피-음료", sector_code: "S3", predicted_growth: 0.075, is_lipstick: true, rank: 3, rationale: "일상 소비·프리미엄 음료 수요 견조" },
  { sector_name: "피부관리실", sector_code: "S4", predicted_growth: 0.072, is_lipstick: true, rank: 4, rationale: "셀프 케어·경기 둔화기 자기관리 수요" },
  { sector_name: "제과점", sector_code: "S5", predicted_growth: 0.068, is_lipstick: true, rank: 5, rationale: "소액 선물·감정 소비 트렌드 지속" },
  { sector_name: "미용실", sector_code: "S6", predicted_growth: 0.065, is_lipstick: true, rank: 6, rationale: "필수 미용 수요와 소규모 지출 선호" },
  { sector_name: "편의점", sector_code: "S7", predicted_growth: 0.062, is_lipstick: false, rank: 7, rationale: "편의성·단일 구매 수요 확대" },
  { sector_name: "슈퍼마켓", sector_code: "S8", predicted_growth: 0.058, is_lipstick: false, rank: 8, rationale: "필수 식료품·가격 민감 수요" },
  { sector_name: "한식음식점", sector_code: "S9", predicted_growth: 0.055, is_lipstick: false, rank: 9, rationale: "외식 수요 회복·가성비 선호" },
  { sector_name: "의약품", sector_code: "S10", predicted_growth: 0.052, is_lipstick: false, rank: 10, rationale: "고령화·건강 관심으로 수요 안정" },
  { sector_name: "분식전문점", sector_code: "S11", predicted_growth: 0.050, is_lipstick: false, rank: 11, rationale: "저가 외식·일상 소비 수요" },
  { sector_name: "치킨전문점", sector_code: "S12", predicted_growth: 0.048, is_lipstick: false, rank: 12, rationale: "배달·모임 수요와 가격 경쟁력" },
  { sector_name: "일반교습학원", sector_code: "S13", predicted_growth: 0.046, is_lipstick: false, rank: 13, rationale: "교육 지출 우선순위 유지" },
  { sector_name: "세탁소", sector_code: "S14", predicted_growth: 0.044, is_lipstick: false, rank: 14, rationale: "필수 서비스·단가 소폭 상승 반영" },
  { sector_name: "문구", sector_code: "S15", predicted_growth: 0.042, is_lipstick: false, rank: 15, rationale: "학기·사무용품 수요 꾸준" },
  { sector_name: "서적", sector_code: "S16", predicted_growth: 0.040, is_lipstick: false, rank: 16, rationale: "취미·자기계발 소비 지속" },
  { sector_name: "애완동물", sector_code: "S17", predicted_growth: 0.038, is_lipstick: false, rank: 17, rationale: "반려동물 관련 지출 증가 추세" },
  { sector_name: "노래방", sector_code: "S18", predicted_growth: 0.036, is_lipstick: false, rank: 18, rationale: "소규모 여가·모임 수요" },
  { sector_name: "가방", sector_code: "S19", predicted_growth: 0.034, is_lipstick: false, rank: 19, rationale: "실속 구매·교체 수요" },
  { sector_name: "안경", sector_code: "S20", predicted_growth: 0.032, is_lipstick: false, rank: 20, rationale: "필수 의료·교체 수요 안정" },
];

// 거시 지표 (CPI, 금리, CCSI). shock_score: 연속형 충격 점수(0~1) 목업
export const macroQuarterly: MacroQuarter[] = [
  { year_quarter: "2020Q1", year: 2020, quarter: 1, cpi: 102, policy_rate: 1.5, ccsi: 90.5, cpi_yoy: 0, ccsi_diff: 0, policy_rate_diff: 0, macro_shock: 1, shock_score: 0.92 },
  { year_quarter: "2020Q2", year: 2020, quarter: 2, cpi: 104.04, policy_rate: 1.5, ccsi: 91, cpi_yoy: 0.02, ccsi_diff: 0.5, policy_rate_diff: 0, macro_shock: 1, shock_score: 0.88 },
  { year_quarter: "2020Q3", year: 2020, quarter: 3, cpi: 106.12, policy_rate: 1.5, ccsi: 91.5, cpi_yoy: 0.02, ccsi_diff: 0.5, policy_rate_diff: 0, macro_shock: 1, shock_score: 0.85 },
  { year_quarter: "2020Q4", year: 2020, quarter: 4, cpi: 108.24, policy_rate: 1.5, ccsi: 92, cpi_yoy: 0.02, ccsi_diff: 0.5, policy_rate_diff: 0, macro_shock: 1, shock_score: 0.82 },
  { year_quarter: "2021Q1", year: 2021, quarter: 1, cpi: 110.41, policy_rate: 1.75, ccsi: 92.5, cpi_yoy: 0.082, ccsi_diff: 0.5, policy_rate_diff: 0.25, macro_shock: 1, shock_score: 0.78 },
  { year_quarter: "2022Q1", year: 2022, quarter: 1, cpi: 119.51, policy_rate: 2, ccsi: 99.5, cpi_yoy: 0.082, ccsi_diff: 5.5, policy_rate_diff: 0.25, macro_shock: 1, shock_score: 0.72 },
  { year_quarter: "2022Q2", year: 2022, quarter: 2, cpi: 121.9, policy_rate: 2, ccsi: 100, cpi_yoy: 0.082, ccsi_diff: 0.5, policy_rate_diff: 0, macro_shock: 0, shock_score: 0.38 },
  { year_quarter: "2023Q1", year: 2023, quarter: 1, cpi: 129.36, policy_rate: 2.25, ccsi: 101.5, cpi_yoy: 0.082, ccsi_diff: 0.5, policy_rate_diff: 0.25, macro_shock: 0, shock_score: 0.28 },
  { year_quarter: "2023Q4", year: 2023, quarter: 4, cpi: 137.28, policy_rate: 2.25, ccsi: 103, cpi_yoy: 0.082, ccsi_diff: 0.5, policy_rate_diff: 0, macro_shock: 0, shock_score: 0.22 },
  { year_quarter: "2024Q1", year: 2024, quarter: 1, cpi: 140.02, policy_rate: 2.5, ccsi: 103.5, cpi_yoy: 0.082, ccsi_diff: 0.5, policy_rate_diff: 0.25, macro_shock: 0, shock_score: 0.20 },
  { year_quarter: "2024Q4", year: 2024, quarter: 4, cpi: 148.59, policy_rate: 2.5, ccsi: 105, cpi_yoy: 0.082, ccsi_diff: 0.5, policy_rate_diff: 0, macro_shock: 0, shock_score: 0.18 },
];
