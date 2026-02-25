export type MacroRow = {
  year_quarter: string;
  year?: number;
  quarter?: number;
  cpi?: number;
  policy_rate?: number;
  unemployment?: number;
  ccsi?: number;
  cpi_yoy?: number;
  shock_score?: number | null;
  macro_shock?: number | null;
};

let macroPromise: Promise<MacroRow[]> | null = null;

const NUMERIC_KEYS = new Set([
  "year",
  "quarter",
  "cpi",
  "policy_rate",
  "unemployment",
  "ccsi",
  "cpi_yoy",
  "ccsi_diff",
  "policy_rate_diff",
  "unemployment_diff",
  "shock_score",
  "macro_shock",
]);

function parseMacroCsv(csvText: string): MacroRow[] {
  const lines = csvText.trim().split(/\r?\n/).filter((line) => line.length > 0);
  if (lines.length < 2) return [];
  const headers = lines[0].split(",").map((h) => h.trim());
  const rows: MacroRow[] = [];
  for (let i = 1; i < lines.length; i++) {
    const cells = lines[i].split(",");
    const row: Record<string, string | number | undefined | null> = {};
    for (let j = 0; j < headers.length; j++) {
      const key = headers[j];
      const raw = cells[j]?.trim() ?? "";
      if (NUMERIC_KEYS.has(key)) {
        const n = Number(raw);
        row[key] = raw === "" || Number.isNaN(n) ? undefined : n;
        if (key === "macro_shock" || key === "shock_score") {
          if (raw !== "" && !Number.isNaN(Number(raw))) row[key] = Number(raw);
          else if (key === "shock_score") row[key] = null;
        }
      } else {
        row[key] = raw || undefined;
      }
    }
    rows.push(row as MacroRow);
  }
  return rows;
}

function fetchMacroCsv(): Promise<MacroRow[]> {
  return fetch("/macro_quarterly.csv", { cache: "no-store" })
    .then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.text();
    })
    .then((text) => parseMacroCsv(text));
}

/** 거시 데이터. public/macro_quarterly.csv를 fetch 후 파싱해 반환. */
export function getMacroData(): Promise<MacroRow[]> {
  if (!macroPromise) {
    macroPromise = fetchMacroCsv().then((arr) => {
      if (arr.length === 0) macroPromise = null;
      return arr;
    });
  }
  return macroPromise;
}
