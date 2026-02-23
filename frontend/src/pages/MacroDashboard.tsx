import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
} from "recharts";
import { macroQuarterly } from "../data/mock";

// 최근 분기 = 현재 경기 국면
const latest = macroQuarterly[macroQuarterly.length - 1];
const latestShockScore = latest?.shock_score ?? 0;

export default function MacroDashboard() {
  return (
    <>
      <h1 style={{ marginTop: 0 }}>거시경제 대시보드</h1>

      <div className="card" style={{ display: "flex", flexWrap: "wrap", gap: "1rem", alignItems: "center" }}>
        <h2 style={{ margin: 0, width: "100%" }}>현재 경기 국면</h2>
        <div style={{ padding: "1rem", background: "#27272a", borderRadius: 8, minWidth: 140 }}>
          <div style={{ fontSize: "0.85rem", color: "#a1a1aa" }}>기준 분기</div>
          <div style={{ fontSize: "1.25rem", fontWeight: 700 }}>{latest?.year_quarter ?? "—"}</div>
        </div>
        <div style={{ padding: "1rem", background: "#27272a", borderRadius: 8, minWidth: 140 }}>
          <div style={{ fontSize: "0.85rem", color: "#a1a1aa" }}>충격 여부</div>
          <span className={`badge ${latest?.macro_shock === 1 ? "shock" : "normal"}`}>
            {latest?.macro_shock === 1 ? "충격기" : "비충격기"}
          </span>
        </div>
        <div style={{ padding: "1rem", background: "#27272a", borderRadius: 8, minWidth: 140 }}>
          <div style={{ fontSize: "0.85rem", color: "#a1a1aa" }}>충격 점수</div>
          <div style={{ fontSize: "1.25rem", fontWeight: 700 }}>
            {(latestShockScore * 100).toFixed(0)}%
          </div>
          <div style={{ fontSize: "0.75rem", color: "#71717a" }}>현재 분기 (0=약함, 100=강함)</div>
        </div>
      </div>

      <div className="card">
        <h2>CPI(소비자물가지수) / 금리 / 소비자심리지수(CCSI) 추이</h2>
        <p style={{ margin: "0 0 0.75rem", fontSize: "0.85rem", color: "#71717a" }}>
          CPI·CCSI: 왼쪽 축(%) / 기준금리: 오른쪽 축(%p, 독립 스케일)
        </p>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={macroQuarterly} margin={{ top: 8, right: 56, left: 48, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="year_quarter" tick={{ fontSize: 11 }} />
            <YAxis yAxisId="left" domain={["auto", "auto"]} tick={{ fontSize: 11 }} label={{ value: "CPI / CCSI", angle: -90, position: "insideLeft", style: { fontSize: 10 } }} />
            <YAxis yAxisId="right" orientation="right" domain={[0, 5]} tick={{ fontSize: 11 }} label={{ value: "기준금리(%)", angle: 90, position: "insideRight", style: { fontSize: 10 } }} />
            <Tooltip
              contentStyle={{ background: "#27272a", border: "1px solid #3f3f46" }}
              formatter={(value: number, name: string) => [
                name === "CCSI" ? value.toFixed(1) : value.toFixed(2),
                name,
              ]}
            />
            <Legend />
            <Line yAxisId="left" type="monotone" dataKey="cpi" name="CPI" stroke="#22c55e" strokeWidth={2} dot={{ r: 2 }} />
            <Line yAxisId="left" type="monotone" dataKey="ccsi" name="CCSI" stroke="#f59e0b" strokeWidth={2} dot={{ r: 2 }} />
            <Line yAxisId="right" type="monotone" dataKey="policy_rate" name="기준금리(%)" stroke="#3b82f6" strokeWidth={2} dot={{ r: 2 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="card">
        <h2>충격 점수 시각화 (분기별 shock_score)</h2>
        <p style={{ margin: "0 0 0.75rem", fontSize: "0.85rem", color: "#71717a" }}>
          0에 가까우면 거시 스트레스 약함, 1에 가까우면 강함 (Z-score 기반 복합 지표)
        </p>
        <ResponsiveContainer width="100%" height={220}>
          <AreaChart
            data={[...macroQuarterly].sort((a, b) => b.shock_score - a.shock_score)}
            margin={{ top: 8, right: 8, left: 8, bottom: 8 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="year_quarter" tick={{ fontSize: 11 }} />
            <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} tickFormatter={(v) => v.toFixed(2)} />
            <Tooltip
              contentStyle={{ background: "#27272a", border: "1px solid #3f3f46" }}
              formatter={(value: number) => [(value * 100).toFixed(1) + "%", "충격 점수"]}
            />
            <Area
              type="monotone"
              dataKey="shock_score"
              name="충격 점수"
              stroke="#dc2626"
              fill="#dc2626"
              fillOpacity={0.4}
            />
          </AreaChart>
        </ResponsiveContainer>
        <p style={{ margin: "0.5rem 0 0", fontSize: "0.85rem", color: "#a1a1aa" }}>
          <strong>근거:</strong> CCSI(소비자심리지수)·CPI 전년비·기준금리 변동을 분기별 Z-score로 표준화한 뒤 합산한 복합 지표. 높을수록 거시 스트레스(충격)가 큰 구간이며, 상위 25% 분기를 충격기(macro_shock=1)로 분류하는 데 사용.
        </p>
      </div>
    </>
  );
}
