import { useState, useEffect, useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceArea,
  BarChart,
  Bar,
} from "recharts";

type LipstickPoint = { year_quarter: string; lipstick_share: number | null; macro_shock?: number | null };
type GrowthPoint = { year_quarter: string; lipstick_median?: number | null; non_lipstick_median?: number | null };
type SensitivityItem = { region_id: string; region_name: string; shock_diff: number };

export default function LipstickEffect() {
  const [lipstickSeries, setLipstickSeries] = useState<LipstickPoint[]>([]);
  const [growthComparison, setGrowthComparison] = useState<GrowthPoint[]>([]);
  const [sensitivityRanking, setSensitivityRanking] = useState<SensitivityItem[]>([]);
  const [lipstickStatus, setLipstickStatus] = useState<"idle" | "loading" | "ok" | "error">("idle");
  const [growthStatus, setGrowthStatus] = useState<"idle" | "loading" | "ok" | "error">("idle");
  const [sensitivityStatus, setSensitivityStatus] = useState<"idle" | "loading" | "ok" | "error">("idle");

  const fetchJson = async (url: string): Promise<unknown> => {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const ct = r.headers.get("content-type") ?? "";
    if (!ct.includes("application/json")) {
      const t = await r.text();
      throw new Error(`Non-JSON response: ${ct} / ${t.slice(0, 80)}`);
    }
    return r.json();
  };

  useEffect(() => {
    setLipstickStatus("loading");
    fetchJson("/api/dashboard/lipstick_series")
      .then((arr) => {
        setLipstickSeries(Array.isArray(arr) ? (arr as LipstickPoint[]) : []);
        setLipstickStatus("ok");
      })
      .catch((e) => {
        console.error("lipstick_series:", e);
        setLipstickStatus("error");
      });
  }, []);

  useEffect(() => {
    setGrowthStatus("loading");
    fetchJson("/api/dashboard/growth_comparison")
      .then((arr) => {
        setGrowthComparison(Array.isArray(arr) ? (arr as GrowthPoint[]) : []);
        setGrowthStatus("ok");
      })
      .catch((e) => {
        console.error("growth_comparison:", e);
        setGrowthStatus("error");
      });
  }, []);

  useEffect(() => {
    setSensitivityStatus("loading");
    fetchJson("/api/dashboard/sensitivity_ranking?limit=20")
      .then((arr) => {
        setSensitivityRanking(Array.isArray(arr) ? (arr as SensitivityItem[]) : []);
        setSensitivityStatus("ok");
      })
      .catch((e) => {
        console.error("sensitivity_ranking:", e);
        setSensitivityStatus("error");
      });
  }, []);

  const shockRanges = useMemo(() => {
    const out: [number, number][] = [];
    let start: number | null = null;
    lipstickSeries.forEach((p, i) => {
      if (p.macro_shock === 1) {
        if (start === null) start = i;
      } else {
        if (start !== null) {
          out.push([start, i - 1]);
          start = null;
        }
      }
    });
    if (start !== null) out.push([start!, lipstickSeries.length - 1]);
    return out;
  }, [lipstickSeries]);

  const rankingChartData = useMemo(
    () => [...sensitivityRanking].map((d) => ({ ...d, shock_diff: Number(d.shock_diff) })).sort((a, b) => b.shock_diff - a.shock_diff),
    [sensitivityRanking]
  );

  return (
    <>
      <h1 style={{ marginTop: 0 }}>립스틱 효과 분석</h1>

      <div className="card">
        <h2>립스틱 지수 시계열 · 충격 구간 음영</h2>
        {lipstickStatus === "loading" && <p style={{ color: "#a1a1aa" }}>로딩 중…</p>}
        {lipstickStatus === "error" && <p style={{ color: "#f87171" }}>립스틱 시계열을 불러오지 못했습니다.</p>}
        {lipstickStatus === "ok" && lipstickSeries.length > 0 && (
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={lipstickSeries} margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="year_quarter" tick={{ fontSize: 11 }} />
              <YAxis domain={["auto", "auto"]} tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: "#27272a", border: "1px solid #3f3f46" }}
                formatter={(value: number) => [value != null ? Number(value).toFixed(3) : "—", "립스틱 비중"]}
              />
              <Legend />
              {shockRanges.map(([a, b], i) => (
                <ReferenceArea
                  key={i}
                  x1={lipstickSeries[a]?.year_quarter}
                  x2={lipstickSeries[b]?.year_quarter}
                  strokeOpacity={0.3}
                  fill="#dc2626"
                  fillOpacity={0.2}
                />
              ))}
              <Line type="monotone" dataKey="lipstick_share" name="립스틱 비중" stroke="#a78bfa" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        )}
        <p style={{ margin: "0.5rem 0 0", fontSize: "0.85rem", color: "#a1a1aa" }}>
          빨간 음영: macro_shock=1 (충격기)
        </p>
      </div>

      <div className="card">
        <h2>립스틱 vs 논립스틱 성장률 비교</h2>
        <p style={{ margin: "0 0 0.75rem", fontSize: "0.9rem", color: "#71717a" }}>
          기준: 립스틱 = Core 6업종(네일숍·미용실·피부관리실·화장품·제과점·커피-음료), 논립스틱 = 그 외. 분기별 상권×업종 sales_growth_qoq 중앙값 (이상치 완화).
        </p>
        {growthStatus === "loading" && <p style={{ color: "#a1a1aa" }}>로딩 중…</p>}
        {growthStatus === "error" && <p style={{ color: "#f87171" }}>성장률 비교 데이터를 불러오지 못했습니다.</p>}
        {growthStatus === "ok" && growthComparison.length > 0 && (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={growthComparison} margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="year_quarter" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => `${(Number(v) * 100).toFixed(1)}%`} />
              <Tooltip
                contentStyle={{ background: "#27272a", border: "1px solid #3f3f46" }}
                formatter={(value: number) => [value != null ? (Number(value) * 100).toFixed(2) + "%" : "—", ""]}
              />
              <Legend />
              <Line type="monotone" dataKey="lipstick_median" name="립스틱 (중앙값)" stroke="#a78bfa" strokeWidth={2} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="non_lipstick_median" name="논립스틱 (중앙값)" stroke="#f59e0b" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      <div className="card">
        <h2>충격 민감도 행정동(상권) 랭킹 Top 20</h2>
        <p style={{ margin: "0 0 0.75rem", fontSize: "0.9rem", color: "#71717a" }}>
          기준: 충격기 평균 lipstick_share − 비충격기 평균 (상권별). 클수록 충격기에 립스틱 비중이 더 크게 올라간 상권.
        </p>
        {sensitivityStatus === "loading" && <p style={{ color: "#a1a1aa" }}>로딩 중…</p>}
        {sensitivityStatus === "error" && <p style={{ color: "#f87171" }}>민감도 랭킹을 불러오지 못했습니다.</p>}
        {sensitivityStatus === "ok" && rankingChartData.length > 0 && (
          <ResponsiveContainer width="100%" height={420}>
            <BarChart data={rankingChartData} layout="vertical" margin={{ top: 4, right: 24, left: 100, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 11 }} tickFormatter={(v) => Number(v).toFixed(3)} />
              <YAxis type="category" dataKey="region_name" width={96} tick={{ fontSize: 10 }} />
              <Tooltip
                contentStyle={{ background: "#27272a", border: "1px solid #3f3f46" }}
                formatter={(value: number) => [Number(value).toFixed(4), "차이"]}
              />
              <Bar dataKey="shock_diff" name="충격기−비충격기" fill="#a78bfa" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>
    </>
  );
}
