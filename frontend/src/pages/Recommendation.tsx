import { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from "recharts";
import { top20Recommendations } from "../data/mock";

export default function Recommendation() {
  const [filter, setFilter] = useState<"all" | "lipstick" | "non_lipstick">("all");

  const filtered =
    filter === "all"
      ? top20Recommendations
      : filter === "lipstick"
        ? top20Recommendations.filter((r) => r.is_lipstick)
        : top20Recommendations.filter((r) => !r.is_lipstick);

  const barData = filtered.map((r) => ({
    name: r.sector_name.length > 8 ? r.sector_name.slice(0, 8) + "…" : r.sector_name,
    fullName: r.sector_name,
    predicted_growth: r.predicted_growth,
    is_lipstick: r.is_lipstick,
    rank: r.rank,
    rationale: r.rationale,
  }));

  return (
    <>
      <h1 style={{ marginTop: 0 }}>추천 시스템</h1>

      <div className="card">
        <h2>다음 분기 TOP-20 업종 · 성장 점수</h2>
        <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1rem", flexWrap: "wrap" }}>
          <button
            onClick={() => setFilter("all")}
            style={{
              padding: "0.4rem 0.8rem",
              borderRadius: 8,
              border: "1px solid #3f3f46",
              background: filter === "all" ? "#3f3f46" : "transparent",
              color: "#e4e4e7",
              cursor: "pointer",
            }}
          >
            전체
          </button>
          <button
            onClick={() => setFilter("lipstick")}
            style={{
              padding: "0.4rem 0.8rem",
              borderRadius: 8,
              border: "1px solid #7c3aed",
              background: filter === "lipstick" ? "#5b21b6" : "transparent",
              color: "#c4b5fd",
              cursor: "pointer",
            }}
          >
            립스틱만
          </button>
          <button
            onClick={() => setFilter("non_lipstick")}
            style={{
              padding: "0.4rem 0.8rem",
              borderRadius: 8,
              border: "1px solid #b45309",
              background: filter === "non_lipstick" ? "#92400e" : "transparent",
              color: "#fcd34d",
              cursor: "pointer",
            }}
          >
            논립스틱만
          </button>
        </div>
        <ResponsiveContainer width="100%" height={380}>
          <BarChart data={barData} margin={{ top: 8, right: 8, left: 8, bottom: 24 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-45} textAnchor="end" height={60} />
            <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => `${(v * 100).toFixed(1)}%`} />
            <Tooltip
              contentStyle={{ background: "#27272a", border: "1px solid #3f3f46" }}
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const p = payload[0].payload;
                return (
                  <div style={{ padding: "0.5rem", minWidth: 220 }}>
                    <div style={{ fontWeight: 600, marginBottom: 4 }}>{p.fullName ?? p.name}</div>
                    <div style={{ color: "#a1a1aa", fontSize: "0.9rem" }}>예측 성장률 {(p.predicted_growth * 100).toFixed(2)}%</div>
                    {p.rationale && <div style={{ marginTop: 6, fontSize: "0.85rem", color: "#d4d4d8" }}>{p.rationale}</div>}
                  </div>
                );
              }}
            />
            <Legend />
            <Bar dataKey="predicted_growth" name="예측 성장률" radius={[4, 4, 0, 0]}>
              {barData.map((entry, index) => (
                <Cell key={index} fill={entry.is_lipstick ? "#a78bfa" : "#f59e0b"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <p style={{ margin: "0.5rem 0 0", fontSize: "0.85rem", color: "#a1a1aa" }}>
          보라: 립스틱 업종 · 주황: 논립스틱
        </p>
        <div style={{ marginTop: "1rem", overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid #3f3f46", color: "#a1a1aa" }}>
                <th style={{ textAlign: "left", padding: "0.5rem 0.75rem" }}>순위</th>
                <th style={{ textAlign: "left", padding: "0.5rem 0.75rem" }}>업종</th>
                <th style={{ textAlign: "right", padding: "0.5rem 0.75rem" }}>예측 성장률</th>
                <th style={{ textAlign: "left", padding: "0.5rem 0.75rem" }}>근거</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((r) => (
                <tr key={r.sector_code} style={{ borderBottom: "1px solid #27272a" }}>
                  <td style={{ padding: "0.5rem 0.75rem" }}>#{r.rank}</td>
                  <td style={{ padding: "0.5rem 0.75rem" }}>{r.sector_name}</td>
                  <td style={{ textAlign: "right", padding: "0.5rem 0.75rem" }}>{(r.predicted_growth * 100).toFixed(2)}%</td>
                  <td style={{ padding: "0.5rem 0.75rem", color: "#a1a1aa" }}>{r.rationale ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card">
        <h2>추천 근거 카드 (상위 6개)</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: "0.75rem" }}>
          {top20Recommendations.slice(0, 6).map((r) => (
            <div
              key={r.sector_code}
              style={{
                padding: "0.75rem",
                background: "#27272a",
                borderRadius: 8,
                borderLeft: `4px solid ${r.is_lipstick ? "#a78bfa" : "#f59e0b"}`,
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                <strong>{r.sector_name}</strong>
                <span className={r.is_lipstick ? "badge" : ""} style={r.is_lipstick ? { background: "#5b21b6", color: "#c4b5fd" } : {}}>
                  #{r.rank}
                </span>
              </div>
              <div style={{ fontSize: "0.9rem", color: "#a1a1aa" }}>
                예측 성장률 {(r.predicted_growth * 100).toFixed(2)}%
              </div>
              {r.rationale && (
                <div style={{ fontSize: "0.8rem", color: "#71717a", marginTop: 4 }}>{r.rationale}</div>
              )}
            </div>
          ))}
        </div>
      </div>
    </>
  );
}
