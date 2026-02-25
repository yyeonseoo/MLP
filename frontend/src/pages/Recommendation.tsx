import { useState, useEffect } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { API_BASE } from "../apiBase";

type TopSectorRow = {
  rank: number;
  region_id: string;
  region_name: string;
  sector_code: string;
  sector_name: string;
  year_quarter: string;
  growth_pct: number;
  p50_million: number;
};

const RATIONALE = "다음 분기 예상 성장률 상위 (전망 모델 기준)";

export default function Recommendation() {
  const [items, setItems] = useState<TopSectorRow[]>([]);
  const [status, setStatus] = useState<"idle" | "loading" | "ok" | "error">("idle");

  const fetchJson = async (url: string): Promise<unknown> => {
    const r = await fetch(API_BASE + url);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const ct = r.headers.get("content-type") ?? "";
    if (!ct.includes("application/json")) {
      const t = await r.text();
      throw new Error(`Non-JSON response: ${ct} / ${t.slice(0, 80)}`);
    }
    return r.json();
  };

  useEffect(() => {
    setStatus("loading");
    fetchJson("/api/dashboard/top_sectors?limit=30")
      .then((arr) => {
        setItems(Array.isArray(arr) ? (arr as TopSectorRow[]) : []);
        setStatus("ok");
      })
      .catch((e) => {
        console.error("[Recommendation] top_sectors:", e);
        setStatus("error");
      });
  }, []);

  const barData = items.map((r) => {
    const label = `${r.region_name} × ${r.sector_name}`;
    return {
      name: label.length > 14 ? label.slice(0, 14) + "…" : label,
      fullName: label,
      growth_pct: r.growth_pct,
      rank: r.rank,
      region_name: r.region_name,
      sector_name: r.sector_name,
    };
  });

  return (
    <>
      <h1 style={{ marginTop: 0 }}>추천 시스템</h1>

      <div className="card">
        <h2>다음 분기 TOP-30 상권×업종 · 예상 성장률</h2>
        {status === "loading" && <p style={{ color: "#a1a1aa" }}>로딩 중…</p>}
        {status === "error" && <p style={{ color: "#f87171" }}>데이터를 불러오지 못했습니다.</p>}
        {status === "ok" && items.length > 0 && (
          <>
        <ResponsiveContainer width="100%" height={380}>
          <BarChart data={barData} margin={{ top: 8, right: 8, left: 8, bottom: 24 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-45} textAnchor="end" height={60} />
            <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => `${Number(v).toFixed(1)}%`} />
            <Tooltip
              contentStyle={{ background: "#27272a", border: "1px solid #3f3f46" }}
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const p = payload[0].payload;
                return (
                  <div style={{ padding: "0.5rem", minWidth: 220 }}>
                    <div style={{ fontWeight: 600, marginBottom: 4 }}>{p.fullName ?? p.name}</div>
                    <div style={{ color: "#a1a1aa", fontSize: "0.9rem" }}>예측 성장률 {Number(p.growth_pct).toFixed(2)}%</div>
                    <div style={{ marginTop: 6, fontSize: "0.85rem", color: "#d4d4d8" }}>{RATIONALE}</div>
                  </div>
                );
              }}
            />
            <Legend />
            <Bar dataKey="growth_pct" name="예측 성장률(%)" fill="#7c3aed" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
        <div style={{ marginTop: "1rem", overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid #3f3f46", color: "#a1a1aa" }}>
                <th style={{ textAlign: "left", padding: "0.5rem 0.75rem" }}>순위</th>
                <th style={{ textAlign: "left", padding: "0.5rem 0.75rem" }}>상권 × 업종</th>
                <th style={{ textAlign: "right", padding: "0.5rem 0.75rem" }}>예측 성장률</th>
                <th style={{ textAlign: "left", padding: "0.5rem 0.75rem" }}>근거</th>
              </tr>
            </thead>
            <tbody>
              {items.map((r) => (
                <tr key={`${r.region_id}-${r.sector_code}`} style={{ borderBottom: "1px solid #27272a" }}>
                  <td style={{ padding: "0.5rem 0.75rem" }}>#{r.rank}</td>
                  <td style={{ padding: "0.5rem 0.75rem" }}>{r.region_name} × {r.sector_name}</td>
                  <td style={{ textAlign: "right", padding: "0.5rem 0.75rem" }}>{Number(r.growth_pct).toFixed(2)}%</td>
                  <td style={{ padding: "0.5rem 0.75rem", color: "#a1a1aa" }}>{RATIONALE}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div style={{ marginTop: "1rem" }}>
          <h3 style={{ margin: "0 0 0.5rem", fontSize: "1rem" }}>추천 근거 카드 (상위 6개)</h3>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: "0.75rem" }}>
            {items.slice(0, 6).map((r) => (
              <div
                key={`${r.region_id}-${r.sector_code}-card`}
                style={{
                  padding: "0.75rem",
                  background: "#27272a",
                  borderRadius: 8,
                  borderLeft: "4px solid #7c3aed",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                  <strong style={{ fontSize: "0.9rem" }}>{r.region_name} × {r.sector_name}</strong>
                  <span className="badge" style={{ background: "#5b21b6", color: "#c4b5fd" }}>#{r.rank}</span>
                </div>
                <div style={{ fontSize: "0.9rem", color: "#a1a1aa" }}>
                  예측 성장률 {Number(r.growth_pct).toFixed(2)}%
                </div>
                <div style={{ fontSize: "0.8rem", color: "#71717a", marginTop: 4 }}>{RATIONALE}</div>
              </div>
            ))}
          </div>
        </div>
          </>
        )}
      </div>
    </>
  );
}
