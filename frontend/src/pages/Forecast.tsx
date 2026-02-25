import { useState, useEffect } from "react";
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from "recharts";

import { getMacroData } from "../apiMacroCache";

export default function Forecast() {
  const [macroData, setMacroData] = useState<{ year_quarter: string; shock_score: number | null }[]>([]);
  const [macroLoad, setMacroLoad] = useState<"idle" | "loading" | "ok" | "error">("idle");

  useEffect(() => {
    setMacroLoad("loading");
    getMacroData()
      .then((arr) => {
        setMacroData(arr.map((r) => ({ year_quarter: r.year_quarter, shock_score: r.shock_score ?? null })));
        setMacroLoad("ok");
      })
      .catch((e) => {
        console.error("[Forecast] macro:", e);
        setMacroLoad("error");
      });
  }, []);

  return (
    <>
      <h1 style={{ marginTop: 0 }}>매출 전망</h1>

      <div className="card">
        <h2>모델 개요</h2>
        <p style={{ lineHeight: 1.7, color: "#d4d4d8" }}>
          다음 분기 매출을 <strong>log(매출+1)</strong> 타깃으로 예측하는 시계열 회귀 모델입니다.
          피처는 직전 매출·4분기 rolling 평균/표준편차, 거래건수 성장률, 계절(quarter), 거시지표(CPI·금리·소비자심리 등), 상권·업종 target encoding입니다.
          학습은 2022년 이하 Train / 2023년 이후 Test로 시간 기준 분할했으며, 중앙값 예측 기준 <strong>MAE 0.27, RMSE 0.50</strong>입니다.
          불확실성 구간은 Quantile 회귀로 <strong>p10(보수)·p50(중앙)·p90(낙관)</strong> 세 가지를 제공합니다.
        </p>
      </div>

      <div className="card">
        <h2>시각화 결과</h2>
        <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
          <div>
            {macroLoad === "loading" && <p style={{ color: "#a1a1aa" }}>거시 충격 시계열 로딩 중…</p>}
            {macroLoad === "error" && <p style={{ color: "#f87171" }}>거시 시계열을 불러오지 못했습니다.</p>}
            {macroLoad === "ok" && macroData.length > 0 && (
              <ResponsiveContainer width="100%" height={280}>
                <AreaChart data={macroData} margin={{ top: 8, right: 8, left: 8, bottom: 24 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="year_quarter" tick={{ fontSize: 11 }} />
                  <YAxis domain={["auto", "auto"]} tick={{ fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ background: "#27272a", border: "1px solid #3f3f46" }}
                    formatter={(value: number) => [value != null ? Number(value).toFixed(3) : "—", "충격 점수"]}
                  />
                  <Area type="monotone" dataKey="shock_score" name="충격 점수" stroke="#dc2626" fill="#dc2626" fillOpacity={0.4} />
                </AreaChart>
              </ResponsiveContainer>
            )}
            <p style={{ marginTop: 8, fontSize: "0.9rem", color: "#a1a1aa" }}>
              거시 충격 시계열: shock_score와 macro_shock=1 구간(빨간 음영). 거시 스트레스가 높았던 구간을 확인할 수 있습니다.
            </p>
          </div>
          <div>
            <img src="/notebook/predicted_vs_actual.png" alt="예측 vs 실제" style={{ maxWidth: "100%", height: "auto", borderRadius: 8 }} />
            <p style={{ marginTop: 8, fontSize: "0.9rem", color: "#a1a1aa" }}>
              예측 vs 실제: Test set에서 중앙값(p50) 예측과 실제 다음 분기 log 매출. y=x에 가까울수록 예측이 정확합니다.
            </p>
          </div>
        </div>
      </div>
    </>
  );
}
