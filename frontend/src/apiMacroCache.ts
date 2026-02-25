import { API_BASE } from "./apiBase";

export type MacroRow = {
  year_quarter: string;
  year?: number;
  quarter?: number;
  cpi?: number;
  policy_rate?: number;
  ccsi?: number;
  cpi_yoy?: number;
  shock_score?: number | null;
  macro_shock?: number | null;
};

let macroPromise: Promise<MacroRow[]> | null = null;

const MACRO_PATH = "/api/dashboard/macro";

function fetchMacroFromUrl(url: string): Promise<MacroRow[]> {
  return fetch(url)
    .then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const ct = r.headers.get("content-type") ?? "";
      if (!ct.includes("application/json")) {
        return r.text().then((t) => {
          throw new Error(`Non-JSON: ${t.slice(0, 80)}`);
        });
      }
      return r.json();
    })
    .then((arr: unknown) => (Array.isArray(arr) ? (arr as MacroRow[]) : []));
}

/** 거시 데이터. 직행(API_BASE) 실패 시 상대경로(프록시)로 1회 재시도 후 빈 배열. */
export function getMacroData(): Promise<MacroRow[]> {
  if (!macroPromise) {
    const directUrl = API_BASE + MACRO_PATH;
    macroPromise = fetchMacroFromUrl(directUrl).catch((e) => {
      console.warn("[macro] 직행 실패, 프록시로 재시도:", e?.message ?? e);
      return fetchMacroFromUrl(MACRO_PATH).catch((e2) => {
        console.error("[macro] 프록시도 실패:", e2?.message ?? e2);
        return [];
      });
    });
  }
  return macroPromise;
}
