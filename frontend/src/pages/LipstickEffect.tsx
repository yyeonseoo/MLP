import { useMemo } from "react";
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
import {
  lipstickSeries,
  growthComparison,
  sensitivityRanking,
} from "../data/mock";

export default function LipstickEffect() {
  // 충격 구간 인덱스 (시계열에서 macro_shock=1인 연속 구간)
  const shockRanges: [number, number][] = [];
  let start: number | null = null;
  lipstickSeries.forEach((p, i) => {
    if (p.macro_shock === 1) {
      if (start === null) start = i;
    } else {
      if (start !== null) {
        shockRanges.push([start, i - 1]);
        start = null;
      }
    }
  });
  if (start !== null) shockRanges.push([start!, lipstickSeries.length - 1]);

  // 충격 민감도 랭킹: shock_diff 기준 내림차순 (큰 값 = 1등이 위로 가도록, reverse 의존 안 함)
  const rankingChartData = useMemo(
    () =>
      [...sensitivityRanking]
        .map((d) => ({ ...d, shock_diff: Number(d.shock_diff) }))
        .sort((a, b) => b.shock_diff - a.shock_diff),
    [sensitivityRanking]
  );

  return (
    <>
      <h1 style={{ marginTop: 0 }}>립스틱 효과 분석</h1>

      <div className="card">
        <h2>립스틱 지수 시계열 · 충격 구간 음영</h2>
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={lipstickSeries} margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="year_quarter" tick={{ fontSize: 11 }} />
            <YAxis domain={["auto", "auto"]} tick={{ fontSize: 11 }} />
            <Tooltip
              contentStyle={{ background: "#27272a", border: "1px solid #3f3f46" }}
              formatter={(value: number) => [value.toFixed(3), "립스틱 비중"]}
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
            <Line
              type="monotone"
              dataKey="lipstick_share"
              name="립스틱 비중"
              stroke="#a78bfa"
              strokeWidth={2}
              dot={{ r: 3 }}
            />
          </LineChart>
        </ResponsiveContainer>
        <p style={{ margin: "0.5rem 0 0", fontSize: "0.85rem", color: "#a1a1aa" }}>
          빨간 음영: macro_shock=1 (충격기)
        </p>
      </div>

      <div className="card">
        <h2>립스틱 vs 논립스틱 성장률 비교</h2>
        <p style={{ margin: "0 0 0.75rem", fontSize: "0.9rem", color: "#71717a" }}>
          기준: 립스틱 = Core 6업종(네일숍·미용실·피부관리실·화장품·제과점·커피-음료), 논립스틱 = 그 외. 분기별로 상권×업종 단위
          <strong> sales_growth_qoq</strong>의 <strong>중앙값</strong> (이상치 완화).
        </p>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={growthComparison} margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="year_quarter" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => `${(v * 100).toFixed(1)}%`} />
            <Tooltip
              contentStyle={{ background: "#27272a", border: "1px solid #3f3f46" }}
              formatter={(value: number) => [(value * 100).toFixed(2) + "%", ""]}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="lipstick_median"
              name="립스틱 (중앙값)"
              stroke="#a78bfa"
              strokeWidth={2}
              dot={{ r: 3 }}
            />
            <Line
              type="monotone"
              dataKey="non_lipstick_median"
              name="논립스틱 (중앙값)"
              stroke="#f59e0b"
              strokeWidth={2}
              dot={{ r: 3 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="card">
        <h2>충격 민감도 행정동(상권) 랭킹 Top 20</h2>
        <p style={{ margin: "0 0 0.75rem", fontSize: "0.9rem", color: "#71717a" }}>
          기준: 충격기 평균 lipstick_share − 비충격기 평균 (상권별). 클수록 충격기에 립스틱 비중이 더 크게 올라간 상권.
        </p>
        <ResponsiveContainer width="100%" height={420}>
          <BarChart
            data={rankingChartData}
            layout="vertical"
            margin={{ top: 4, right: 24, left: 100, bottom: 4 }}
          >
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
      </div>
    </>
  );
}
