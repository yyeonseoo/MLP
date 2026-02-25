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

/** 상대 경로만 사용해 5173→8000 프록시를 타도록 함 (직행 시 Failed to fetch / 재시도 시 HTML 방지). */
const MACRO_PATH = "/api/dashboard/macro";

// #region agent log
function debugLog(
  hypothesisId: string,
  message: string,
  data: Record<string, unknown>
): void {
  const payload = {
    sessionId: "62212c",
    hypothesisId,
    location: "apiMacroCache.ts",
    message,
    data: { ...data, origin: typeof window !== "undefined" ? window.location?.origin : undefined },
    timestamp: Date.now(),
  };
  console.log("[macro-debug]", hypothesisId, message, payload.data);
  fetch("http://127.0.0.1:7606/ingest/e72ae99e-15e9-4397-a84e-d737af9aa433", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Debug-Session-Id": "62212c" },
    body: JSON.stringify(payload),
  }).catch(() => {});
}
// #endregion

function fetchMacroFromUrl(url: string): Promise<MacroRow[]> {
  // #region agent log
  debugLog("H1", "fetchMacroFromUrl called", { url });
  // #endregion
  return fetch(url, { cache: "no-store" })
    .then((r) => {
      // #region agent log
      const ct = r.headers.get("content-type") ?? "";
      debugLog("H3", "macro response received", {
        url,
        status: r.status,
        contentType: ct,
        ok: r.ok,
      });
      debugLog("H4", "macro response status/type", {
        status: r.status,
        contentType: ct,
        isJson: ct.includes("application/json"),
      });
      // #endregion
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      if (!ct.includes("application/json")) {
        return r.text().then((t) => {
          // #region agent log
          debugLog("H3", "Non-JSON body", { url, bodySlice: t.slice(0, 120) });
          // #endregion
          throw new Error(`Non-JSON: ${t.slice(0, 80)}`);
        });
      }
      return r.json();
    })
    .then((arr: unknown) => (Array.isArray(arr) ? (arr as MacroRow[]) : []));
}

/** 거시 데이터. 상대 경로(프록시) 시도 후 실패 시 8000 직행 1회 재시도. */
const MACRO_DIRECT = "http://127.0.0.1:8000/api/dashboard/macro";

export function getMacroData(): Promise<MacroRow[]> {
  // #region agent log
  debugLog("H1", "getMacroData entry", {
    MACRO_PATH,
    cached: !!macroPromise,
  });
  debugLog("H5", "macroPromise cache", { hadCache: !!macroPromise });
  // #endregion
  if (!macroPromise) {
    macroPromise = fetchMacroFromUrl(MACRO_PATH)
      .catch((e) => {
        // #region agent log
        debugLog("H3", "macro fetch failed, retry direct", {
          phase: "fetch_fail",
          message: e?.message ?? String(e),
        });
        // #endregion
        console.warn("[macro] 프록시 실패, 8000 직행 재시도:", e?.message ?? e);
        return fetchMacroFromUrl(MACRO_DIRECT).catch((e2) => {
          console.error("[macro] 불러오기 실패:", e2?.message ?? e2);
          return [];
        });
      })
      .then((arr) => {
        if (arr.length === 0) macroPromise = null;
        return arr;
      });
  }
  return macroPromise;
}
