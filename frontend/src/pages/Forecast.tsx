import { useState, useEffect, useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from "recharts";
// 예측/옵션은 프록시 사용(5173→8000). API_BASE 직접 호출 시 연결 실패 방지.
const FORECAST_API = "";

import { getMacroData } from "../apiMacroCache";

// 개발 시 API_BASE로 8000 직접 요청, 프로덕션은 같은 origin
type RegionOption = { region_id: string; region_name: string };
type SectorOption = { sector_code: string; sector_name: string };
type Combination = { region_id: string; region_name: string; sector_code: string; sector_name: string };
type ForecastResult = {
  base_quarter: string;
  region_name: string;
  sector_name: string;
  p10: number;
  p50: number;
  p90: number;
  growth_pct: number;
};

export default function Forecast() {
  const [regions, setRegions] = useState<RegionOption[]>([]);
  const [sectors, setSectors] = useState<SectorOption[]>([]);
  const [combinations, setCombinations] = useState<Combination[]>([]);
  const [regionId, setRegionId] = useState("");
  const [sectorCode, setSectorCode] = useState("");
  const [optionsLoadStatus, setOptionsLoadStatus] = useState<"idle" | "loading" | "ok" | "error">("idle");
  const [status, setStatus] = useState<"idle" | "loading" | "ok" | "error">("idle");
  const [result, setResult] = useState<ForecastResult | null>(null);
  const [errorMsg, setErrorMsg] = useState("");

  const [growthHist, setGrowthHist] = useState<{ bins: number[]; counts: number[] }>({ bins: [], counts: [] });
  const [growthHistLoad, setGrowthHistLoad] = useState<"idle" | "loading" | "ok" | "error">("idle");
  const [macroData, setMacroData] = useState<{ year_quarter: string; shock_score: number | null }[]>([]);
  const [macroLoad, setMacroLoad] = useState<"idle" | "loading" | "ok" | "error">("idle");

  useEffect(() => {
    setOptionsLoadStatus("loading");
    fetch(FORECAST_API + "/api/forecast/options")
      .then((r) => {
        console.log("OPTIONS status:", r.status);
        return r.json();
      })
      .then((data: { regions?: RegionOption[]; sectors?: SectorOption[]; combinations?: Combination[] }) => {
        console.log("OPTIONS json:", data);
        setRegions(data.regions ?? []);
        setSectors(data.sectors ?? []);
        const combos = data.combinations ?? [];
        setCombinations(combos);
        setOptionsLoadStatus("ok");
        if (combos.length > 0) {
          setRegionId(String(combos[0].region_id));
          setSectorCode(String(combos[0].sector_code));
        } else if (data.regions?.length && data.sectors?.length) {
          setRegionId(String(data.regions[0].region_id));
          setSectorCode(String(data.sectors[0].sector_code));
        }
      })
      .catch((e) => {
        console.error("[Forecast] forecast/options:", e);
        setOptionsLoadStatus("error");
      });
  }, []);

  useEffect(() => {
    setGrowthHistLoad("loading");
    fetch(FORECAST_API + "/api/dashboard/sales_growth_hist?bins=50")
      .then((r) => r.json())
      .then((d: { bins: number[]; counts: number[] }) => {
        setGrowthHist(d);
        setGrowthHistLoad("ok");
      })
      .catch((e) => {
        console.error("[Forecast] sales_growth_hist:", e);
        setGrowthHistLoad("error");
      });
  }, []);

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

  const validRegionOptions = useMemo(() => {
    const list = sectorCode
      ? combinations.filter((c) => String(c.sector_code) === String(sectorCode))
      : combinations;
    const seen = new Set<string>();
    return list
      .filter((c) => {
        const rid = String(c.region_id);
        if (seen.has(rid)) return false;
        seen.add(rid);
        return true;
      })
      .map((c) => ({ region_id: String(c.region_id), region_name: c.region_name }))
      .sort((a, b) => a.region_name.localeCompare(b.region_name));
  }, [combinations, sectorCode]);

  const validSectorOptions = useMemo(() => {
    const list = regionId
      ? combinations.filter((c) => String(c.region_id) === String(regionId))
      : combinations;
    const seen = new Set<string>();
    return list
      .filter((c) => {
        const sc = String(c.sector_code);
        if (seen.has(sc)) return false;
        seen.add(sc);
        return true;
      })
      .map((c) => ({ sector_code: String(c.sector_code), sector_name: c.sector_name }))
      .sort((a, b) => a.sector_name.localeCompare(b.sector_name));
  }, [combinations, regionId]);

  const isValidSelection = useMemo(
    () =>
      combinations.some(
        (c) => String(c.region_id) === String(regionId) && String(c.sector_code) === String(sectorCode)
      ),
    [combinations, regionId, sectorCode]
  );

  useEffect(() => {
    if (
      validRegionOptions.length &&
      !validRegionOptions.some((r) => String(r.region_id) === String(regionId))
    ) {
      setRegionId(validRegionOptions[0].region_id);
    }
  }, [validRegionOptions, regionId]);
  useEffect(() => {
    if (
      validSectorOptions.length &&
      !validSectorOptions.some((s) => String(s.sector_code) === String(sectorCode))
    ) {
      setSectorCode(validSectorOptions[0].sector_code);
    }
  }, [validSectorOptions, sectorCode]);

  const fetchForecast = async () => {
    setStatus("loading");
    setErrorMsg("");
    try {
      const params = new URLSearchParams({ region_id: regionId, sector_code: sectorCode });
      const url = FORECAST_API + `/api/forecast?${params}`;
      console.log("FORECAST url:", url);
      const res = await fetch(url);
      console.log("FORECAST status:", res.status);
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error((d as { detail?: string }).detail ?? `HTTP ${res.status}`);
      }
      const data: ForecastResult = await res.json();
      console.log("FORECAST json:", data);
      setResult(data);
      setStatus("ok");
    } catch (e) {
      console.error("[Forecast] forecast (region_id/sector_code):", e);
      setResult(null);
      setStatus("error");
      setErrorMsg(e instanceof Error ? e.message : "예측을 불러올 수 없습니다.");
    }
  };

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
            {growthHistLoad === "loading" && <p style={{ color: "#a1a1aa" }}>성장률 분포 로딩 중…</p>}
            {growthHistLoad === "error" && <p style={{ color: "#f87171" }}>성장률 분포를 불러오지 못했습니다.</p>}
            {growthHistLoad === "ok" && growthHist.bins.length > 0 && (
              <ResponsiveContainer width="100%" height={280}>
                <BarChart
                  data={growthHist.bins.map((bin, i) => ({ bin: Number(bin).toFixed(3), count: growthHist.counts[i] ?? 0 }))}
                  margin={{ top: 8, right: 8, left: 8, bottom: 24 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="bin" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip contentStyle={{ background: "#27272a", border: "1px solid #3f3f46" }} />
                  <Bar dataKey="count" name="건수" fill="#7c3aed" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
            <p style={{ marginTop: 8, fontSize: "0.9rem", color: "#a1a1aa" }}>
              성장률 분포: 전분기 대비 매출 성장률(sales_growth_qoq) 분포. 이상치 완화를 위해 ±0.5로 clip 후 히스토그램으로 표시했습니다.
            </p>
          </div>
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

      <div className="card">
        <h2>예측하기</h2>
        <p style={{ marginBottom: "1rem", fontSize: "0.9rem", color: "#a1a1aa" }}>
          지역(상권)과 업종을 선택한 뒤 예측 보기를 누르면 다음 분기 예상 매출(보수·중앙·낙관)과 전분기 대비 증감률을 볼 수 있습니다.
        </p>
        {optionsLoadStatus === "loading" && (
          <p style={{ color: "#a1a1aa", marginBottom: "1rem" }}>옵션 불러오는 중…</p>
        )}
        {optionsLoadStatus === "error" && (
          <p style={{ color: "#f87171", marginBottom: "1rem" }}>
            지역·업종 목록을 불러오지 못했습니다. 백엔드(8000)가 실행 중인지 확인해 주세요.
          </p>
        )}
        <div style={{ display: "flex", flexWrap: "wrap", gap: "1rem", alignItems: "flex-end", marginBottom: "1rem" }}>
          <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <span style={{ fontSize: "0.85rem", color: "#a1a1aa" }}>지역(상권)</span>
            <select
              value={regionId}
              onChange={(e) => setRegionId(e.target.value)}
              style={{
                padding: "0.5rem 0.75rem",
                borderRadius: 8,
                border: "1px solid #3f3f46",
                background: "#27272a",
                color: "#e4e4e7",
                minWidth: 200,
              }}
            >
              {validRegionOptions.map((r) => (
                <option key={r.region_id} value={String(r.region_id)}>{r.region_name}</option>
              ))}
            </select>
          </label>
          <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <span style={{ fontSize: "0.85rem", color: "#a1a1aa" }}>업종</span>
            <select
              value={sectorCode}
              onChange={(e) => setSectorCode(e.target.value)}
              style={{
                padding: "0.5rem 0.75rem",
                borderRadius: 8,
                border: "1px solid #3f3f46",
                background: "#27272a",
                color: "#e4e4e7",
                minWidth: 200,
              }}
            >
              {validSectorOptions.map((s) => (
                <option key={s.sector_code} value={String(s.sector_code)}>{s.sector_name}</option>
              ))}
            </select>
          </label>
          <button
            type="button"
            onClick={fetchForecast}
            disabled={status === "loading" || !isValidSelection}
            style={{
              padding: "0.5rem 1.25rem",
              borderRadius: 8,
              border: "none",
              background: "#7c3aed",
              color: "#fff",
              cursor: status === "loading" ? "wait" : "pointer",
              fontWeight: 600,
            }}
          >
            {status === "loading" ? "예측 중…" : "예측 보기"}
          </button>
        </div>
        {combinations.length === 0 && (regions.length > 0 || sectors.length > 0) && (
          <p style={{ fontSize: "0.85rem", color: "#fbbf24", marginTop: -8, marginBottom: 8 }}>
            예측 가능한 조합을 불러오지 못했습니다. 터미널에서 프로젝트 루트로 이동한 뒤 <code>PYTHONPATH=. uvicorn api.app:app --host 0.0.0.0 --port 8000</code> 으로 백엔드를 실행해 주세요.
          </p>
        )}
        {!isValidSelection && (regionId || sectorCode) && combinations.length > 0 && (
          <p style={{ fontSize: "0.85rem", color: "#a1a1aa", marginTop: -8, marginBottom: 8 }}>
            선택한 지역·업종 조합으로만 예측할 수 있습니다. 위 목록에서 유효한 조합을 선택해 주세요.
          </p>
        )}
        {status === "error" && (
          <p style={{ color: "#f87171", fontSize: "0.9rem" }}>
            {errorMsg}
            {errorMsg.includes("fetch") || errorMsg.includes("Failed") ? " API 서버가 꺼져 있을 수 있습니다. 터미널에서 PYTHONPATH=. uvicorn api.app:app --host 0.0.0.0 --port 8000 를 실행해 주세요." : ""}
          </p>
        )}
        {status === "ok" && result && (
          <div
            style={{
              padding: "1rem",
              background: "#27272a",
              borderRadius: 8,
              borderLeft: "4px solid #7c3aed",
            }}
          >
            <div style={{ fontSize: "0.85rem", color: "#a1a1aa", marginBottom: 6 }}>
              {result.region_name} × {result.sector_name} (기준: {result.base_quarter})
            </div>
            <p style={{ margin: "0 0 4px", fontSize: "1.1rem" }}>
              다음 분기 예상 매출(중앙): <strong>{(result.p50 / 100).toFixed(1)}억원</strong> ({result.p50.toFixed(1)}백만원)
            </p>
            <p style={{ margin: "0 0 4px", fontSize: "0.95rem", color: "#d4d4d8" }}>
              보수~낙관: {result.p10.toFixed(1)} ~ {result.p90.toFixed(1)} 백만원
            </p>
            <p style={{ margin: 0, fontSize: "0.95rem", color: "#a78bfa" }}>
              전분기 대비 변화율: {result.growth_pct >= 0 ? "+" : ""}{result.growth_pct.toFixed(1)}%
            </p>
          </div>
        )}
      </div>
    </>
  );
}
