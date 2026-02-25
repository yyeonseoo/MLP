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

const DATA_SOURCES = [
  {
    file: "소비자심리지수(CCSI).xlsx",
    sheet: "Sheet1",
    columns: "0열=기간(YYYY-MM), 1열=CCSI 값. 7행부터 데이터",
    aggregation: "월별 → 분기 평균",
    indicator: "ccsi (소비자심리지수)",
  },
  {
    file: "소비자물가지수_20260223183741.xlsx",
    sheet: "데이터",
    columns: "0열=시점(연도), 1열=항목, 2열=총지수. '연간 (2023=100)' 행에서 연도별 CPI",
    aggregation: "연도별 → 분기별 동일값",
    indicator: "cpi (소비자물가지수)",
  },
  {
    file: "성별_경제활동인구_총괄_20260223183912.xlsx",
    sheet: "데이터",
    columns: "row0=기간(YYYY.MM 또는 YYYY), row1=지표명, row2='계'. 8컬럼 블록 반복, 7번째=실업률(%)",
    aggregation: "월별 → 분기 평균",
    indicator: "unemployment (실업률)",
  },
];

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

      <div className="card">
        <h2>데이터 출처 및 컬럼 분석</h2>
        <p style={{ marginBottom: "1rem", fontSize: "0.9rem", color: "#a1a1aa" }}>
          아래 세 개 raw 엑셀 파일을 <code>scripts/build_macro_quarterly.py</code>로 읽어 분기별{" "}
          <code>macro_quarterly.csv</code>를 생성합니다. 기준금리(policy_rate)는 raw에 없어 스크립트 내 2020~2024 근사 시계열을 사용합니다.
        </p>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid #3f3f46" }}>
                <th style={{ textAlign: "left", padding: "0.5rem 0.75rem" }}>파일명</th>
                <th style={{ textAlign: "left", padding: "0.5rem 0.75rem" }}>시트</th>
                <th style={{ textAlign: "left", padding: "0.5rem 0.75rem" }}>구조·컬럼</th>
                <th style={{ textAlign: "left", padding: "0.5rem 0.75rem" }}>집계 방식</th>
                <th style={{ textAlign: "left", padding: "0.5rem 0.75rem" }}>대시보드 지표</th>
              </tr>
            </thead>
            <tbody>
              {DATA_SOURCES.map((row, i) => (
                <tr key={i} style={{ borderBottom: "1px solid #27272a" }}>
                  <td style={{ padding: "0.5rem 0.75rem", color: "#e4e4e7" }}>{row.file}</td>
                  <td style={{ padding: "0.5rem 0.75rem", color: "#a1a1aa" }}>{row.sheet}</td>
                  <td style={{ padding: "0.5rem 0.75rem", color: "#d4d4d8", maxWidth: 320 }}>{row.columns}</td>
                  <td style={{ padding: "0.5rem 0.75rem", color: "#a1a1aa" }}>{row.aggregation}</td>
                  <td style={{ padding: "0.5rem 0.75rem", color: "#a78bfa" }}>{row.indicator}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {status === "loading" && <p style={{ color: "#a1a1aa" }}>데이터 로딩 중…</p>}
      {status === "error" && <p style={{ color: "#f87171" }}>거시 데이터를 불러오지 못했습니다.</p>}
      {status === "ok" && macroData.length === 0 && (
        <p style={{ color: "#a1a1aa" }}>거시 데이터가 없습니다. macro_quarterly.csv를 확인하세요.</p>
      )}

      {status === "ok" && macroData.length > 0 && (
        <>
          <div className="card" style={{ display: "flex", flexWrap: "wrap", gap: "1rem", alignItems: "center" }}>
            <h2 style={{ margin: 0, width: "100%" }}>현재 경기 국면</h2>
            <div style={{ padding: "1rem", background: "#27272a", borderRadius: 8, minWidth: 120 }}>
              <div style={{ fontSize: "0.85rem", color: "#a1a1aa" }}>기준 분기</div>
              <div style={{ fontSize: "1.25rem", fontWeight: 700 }}>{latest?.year_quarter ?? "—"}</div>
            </div>
            <div style={{ padding: "1rem", background: "#27272a", borderRadius: 8, minWidth: 100 }}>
              <div style={{ fontSize: "0.85rem", color: "#a1a1aa" }}>CPI</div>
              <div style={{ fontSize: "1.1rem", fontWeight: 600 }}>{latest?.cpi != null ? Number(latest.cpi).toFixed(1) : "—"}</div>
            </div>
            <div style={{ padding: "1rem", background: "#27272a", borderRadius: 8, minWidth: 100 }}>
              <div style={{ fontSize: "0.85rem", color: "#a1a1aa" }}>기준금리(%)</div>
              <div style={{ fontSize: "1.1rem", fontWeight: 600 }}>{latest?.policy_rate != null ? Number(latest.policy_rate).toFixed(1) : "—"}</div>
            </div>
            <div style={{ padding: "1rem", background: "#27272a", borderRadius: 8, minWidth: 100 }}>
              <div style={{ fontSize: "0.85rem", color: "#a1a1aa" }}>CCSI</div>
              <div style={{ fontSize: "1.1rem", fontWeight: 600 }}>{latest?.ccsi != null ? Number(latest.ccsi).toFixed(1) : "—"}</div>
            </div>
            <div style={{ padding: "1rem", background: "#27272a", borderRadius: 8, minWidth: 100 }}>
              <div style={{ fontSize: "0.85rem", color: "#a1a1aa" }}>실업률(%)</div>
              <div style={{ fontSize: "1.1rem", fontWeight: 600 }}>{latest?.unemployment != null ? Number(latest.unemployment).toFixed(1) : "—"}</div>
            </div>
            <div style={{ padding: "1rem", background: "#27272a", borderRadius: 8, minWidth: 120 }}>
              <div style={{ fontSize: "0.85rem", color: "#a1a1aa" }}>충격 여부</div>
              <span className={`badge ${latest?.macro_shock === 1 ? "shock" : "normal"}`}>
                {latest?.macro_shock === 1 ? "충격기" : "비충격기"}
              </span>
            </div>
            <div style={{ padding: "1rem", background: "#27272a", borderRadius: 8, minWidth: 140 }}>
              <div style={{ fontSize: "0.85rem", color: "#a1a1aa" }}>충격 점수</div>
              <div style={{ fontSize: "1.1rem", fontWeight: 600 }}>
                {typeof latestShockScore === "number" ? Number(latestShockScore).toFixed(2) : "—"}
              </div>
              <div style={{ fontSize: "0.75rem", color: "#71717a" }}>Z-score 기반</div>
            </div>
          </div>

          <div className="card">
            <h2>CPI / 기준금리 / CCSI / 실업률 추이</h2>
            <p style={{ margin: "0 0 0.75rem", fontSize: "0.85rem", color: "#71717a" }}>
              CPI·CCSI·실업률: 왼쪽 축 / 기준금리: 오른쪽 축(%)
            </p>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={macroData} margin={{ top: 8, right: 56, left: 48, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year_quarter" tick={{ fontSize: 11 }} />
                <YAxis yAxisId="left" domain={["auto", "auto"]} tick={{ fontSize: 11 }} label={{ value: "CPI / CCSI / 실업률", angle: -90, position: "insideLeft", style: { fontSize: 10 } }} />
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
                <Line yAxisId="left" type="monotone" dataKey="unemployment" name="실업률(%)" stroke="#a78bfa" strokeWidth={2} dot={{ r: 2 }} />
                <Line yAxisId="right" type="monotone" dataKey="policy_rate" name="기준금리(%)" stroke="#3b82f6" strokeWidth={2} dot={{ r: 2 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="card">
            <h2>충격 점수 시각화 (분기별 shock_score)</h2>
            <p style={{ margin: "0 0 0.75rem", fontSize: "0.85rem", color: "#71717a" }}>
              Z-score 기반 복합 지표. 높을수록 거시 스트레스가 큰 구간이며, 상위 25% 분기를 충격기(macro_shock=1)로 분류하는 데 사용합니다.
            </p>
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={macroData} margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
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
          </div>
        </>
      )}
    </>
  );
}
