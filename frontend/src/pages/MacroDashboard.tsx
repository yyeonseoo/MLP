import { useState, useEffect } from "react";
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
import { getMacroData, type MacroRow } from "../apiMacroCache";

export default function MacroDashboard() {
  const [macroData, setMacroData] = useState<MacroRow[]>([]);
  const [status, setStatus] = useState<"idle" | "loading" | "ok" | "error">("idle");

  useEffect(() => {
    setStatus("loading");
    getMacroData()
      .then((arr) => {
        setMacroData(arr);
        setStatus("ok");
      })
      .catch((e) => {
        console.error("[MacroDashboard] macro:", e);
        setStatus("error");
      });
  }, []);

  const latest = macroData.length > 0 ? macroData[macroData.length - 1] : null;
  const latestShockScore = latest?.shock_score ?? 0;

  return (
    <>
      <h1 style={{ marginTop: 0 }}>거시경제 대시보드</h1>
      {status === "loading" && <p style={{ color: "#a1a1aa" }}>데이터 로딩 중…</p>}
      {status === "error" && <p style={{ color: "#f87171" }}>거시 데이터를 불러오지 못했습니다.</p>}

      {status === "ok" && (
        <>
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
            {typeof latestShockScore === "number" ? latestShockScore.toFixed(2) : "—"}
          </div>
          <div style={{ fontSize: "0.75rem", color: "#71717a" }}>Z-score 기반 (높을수록 스트레스)</div>
        </div>
      </div>

      <div className="card">
        <h2>CPI(소비자물가지수) / 금리 / 소비자심리지수(CCSI) 추이</h2>
        <p style={{ margin: "0 0 0.75rem", fontSize: "0.85rem", color: "#71717a" }}>
          CPI·CCSI: 왼쪽 축 / 기준금리: 오른쪽 축(%)
        </p>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={macroData} margin={{ top: 8, right: 56, left: 48, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="year_quarter" tick={{ fontSize: 11 }} />
            <YAxis yAxisId="left" domain={["auto", "auto"]} tick={{ fontSize: 11 }} label={{ value: "CPI / CCSI", angle: -90, position: "insideLeft", style: { fontSize: 10 } }} />
            <YAxis yAxisId="right" orientation="right" domain={[0, 5]} tick={{ fontSize: 11 }} label={{ value: "기준금리(%)", angle: 90, position: "insideRight", style: { fontSize: 10 } }} />
            <Tooltip
              contentStyle={{ background: "#27272a", border: "1px solid #3f3f46" }}
              formatter={(value: number, name: string) => [
                value != null ? (name === "CCSI" ? Number(value).toFixed(1) : Number(value).toFixed(2)) : "—",
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
            data={[...macroData].sort((a, b) => (b.shock_score ?? 0) - (a.shock_score ?? 0))}
            margin={{ top: 8, right: 8, left: 8, bottom: 8 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="year_quarter" tick={{ fontSize: 11 }} />
            <YAxis domain={["auto", "auto"]} tick={{ fontSize: 11 }} tickFormatter={(v) => Number(v).toFixed(2)} />
            <Tooltip
              contentStyle={{ background: "#27272a", border: "1px solid #3f3f46" }}
              formatter={(value: number) => [value != null ? Number(value).toFixed(3) : "—", "충격 점수"]}
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
      )}
    </>
  );
}
