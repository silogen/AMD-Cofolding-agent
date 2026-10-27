// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { StructurePanel } from "../types/api";
import { sharedRelative } from "./assets";
import { toSdf, type Cell, type Sheet } from "./csv";

/**
 * Every molecule a REINVENT run generated, from its results CSV
 * (results_1.csv), for the Results tab.
 *
 * REINVENT 4 writes one row per sampled molecule:
 *
 *   Agent,Prior,Target,Score,SMILES,SMILES_state,Input_SMILES,Scaffold,
 *   Aqueous solubility,Aqueous solubility (raw),std (Aqueous solubility),...,step
 *
 * Agent/Prior/Target are likelihoods, Score is the total, each component has
 * its transformed value and "<name> (raw)", and component metadata comes as
 * "<key> (<name>)" (a Boltz component adds "structure_path (<name>)").
 * SMILES_state is 0 for an invalid SMILES, 1 valid, 2 a duplicate in the
 * batch; a component that failed leaves NaN or an empty cell.
 */

export type CellValue = number | string | boolean | null;

export type ColumnRole = "total" | "component" | "raw" | "meta" | "structure" | "step" | "state" | "likelihood" | "text";

export interface RunColumn {
  /** The CSV header. */
  key: string;
  label: string;
  role: ColumnRole;
  numeric: boolean;
  /** Every value a whole number (step, state): shown without decimals. */
  integer: boolean;
  /** The component a raw or metadata column belongs to. */
  component: string | null;
}

export interface RunRow {
  /** Position in the CSV, so rows keep an identity through sorting. */
  id: number;
  smiles: string;
  step: number | null;
  total: number | null;
  state: number | null;
  /** Why the row has no usable score, or null when it was scored. */
  failure: string | null;
  structure: string | null;
  /** One value per column, in column order. */
  cells: CellValue[];
}

export interface RunTable {
  columns: RunColumn[];
  rows: RunRow[];
  components: string[];
  failed: number;
  /** Index of the structure path column, or -1 when the run did not cofold. */
  structureColumn: number;
}

const SMILES_KEYS = ["smiles", "canonical_smiles", "smiles_string"];
const TOTAL_KEYS = ["score", "total_score", "total score", "totalscore"];
const STATE_KEYS = ["smiles_state", "smilesstate", "state"];
const STEP_KEYS = ["step", "batch", "iteration"];
const LIKELIHOOD_KEYS = ["agent", "prior", "target", "agent nll", "prior nll", "augmented nll"];
const TEXT_KEYS = ["input_smiles", "scaffold", "inchikey", "smiles_input"];
const MISSING = new Set(["", "nan", "none", "null", "na", "n/a", "--", "-", "inf", "-inf"]);
const STRUCTURE_FILE = /\.(cif|mmcif|pdb)$/i;

/** RFC 4180 records: quoted fields may hold commas, quotes ("") and line breaks. */
export function parseCsv(text: string): string[][] {
  const records: string[][] = [];
  let record: string[] = [];
  let field = "";
  let quoted = false;
  let i = 0;
  const n = text.length;
  while (i < n) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i += 2;
          continue;
        }
        quoted = false;
      } else field += ch;
      i += 1;
      continue;
    }
    if (ch === '"' && field === "") quoted = true;
    else if (ch === ",") {
      record.push(field);
      field = "";
    } else if (ch === "\n" || ch === "\r") {
      record.push(field);
      records.push(record);
      record = [];
      field = "";
      if (ch === "\r" && text[i + 1] === "\n") i += 1;
    } else field += ch;
    i += 1;
  }
  if (field !== "" || record.length) {
    record.push(field);
    records.push(record);
  }
  return records.filter((r) => r.length > 1 || r[0] !== "");
}

/** A CSV cell as a value: numbers (also "1e-3"), booleans, null for NaN and friends, else the text. */
export function parseCell(raw: string): CellValue {
  const text = raw.trim();
  if (MISSING.has(text.toLowerCase())) return null;
  if (text === "True" || text === "true") return true;
  if (text === "False" || text === "false") return false;
  const n = Number(text);
  return Number.isFinite(n) && /^[-+]?(\d|\.\d)/.test(text) ? n : text;
}

const find = (keys: string[], headers: string[], taken: Set<number>) => {
  const index = headers.findIndex((h, i) => !taken.has(i) && keys.includes(h.trim().toLowerCase()));
  if (index >= 0) taken.add(index);
  return index;
};

const SUFFIX = /^(.+?)\s*\((.+)\)$/;

export function parseRunResults(text: string): RunTable {
  const records = parseCsv(text);
  const headers = (records[0] ?? []).map((h) => h.trim());
  const width = headers.length;
  // A results file read while REINVENT writes it can end in half a line.
  const body = records.slice(1).filter((r) => r.length === width);
  const values = body.map((r) => r.map(parseCell));

  const taken = new Set<number>();
  const smilesAt = find(SMILES_KEYS, headers, taken);
  const totalAt = find(TOTAL_KEYS, headers, taken);
  const stateAt = find(STATE_KEYS, headers, taken);
  const stepAt = find(STEP_KEYS, headers, taken);

  const numeric = headers.map((_, c) => {
    let numbers = 0;
    let filled = 0;
    for (const row of values) {
      const v = row[c];
      if (v === null) continue;
      filled += 1;
      if (typeof v === "number") numbers += 1;
    }
    return filled === 0 || numbers / filled > 0.5;
  });
  const integer = headers.map((_, c) => numeric[c] && values.every((row) => typeof row[c] !== "number" || Number.isInteger(row[c])));
  const looksLikeFiles = (c: number) => values.some((row) => typeof row[c] === "string" && STRUCTURE_FILE.test(row[c] as string));

  const rawOf = new Map<number, string>();
  headers.forEach((h, c) => {
    const match = /^(.+?)\s*\(raw\)$/i.exec(h);
    if (!taken.has(c) && match) rawOf.set(c, match[1]);
  });
  const components = new Set(rawOf.values());
  headers.forEach((h, c) => {
    const lower = h.toLowerCase();
    if (taken.has(c) || rawOf.has(c) || SUFFIX.test(h) || LIKELIHOOD_KEYS.includes(lower) || TEXT_KEYS.includes(lower)) return;
    if (numeric[c]) components.add(h);
  });

  const columns: RunColumn[] = headers.map((key, c) => {
    const lower = key.toLowerCase();
    const column = (role: ColumnRole, label = key, component: string | null = null): RunColumn => ({ key, label, role, numeric: numeric[c], integer: integer[c], component });
    if (c === smilesAt) return column("text", "SMILES");
    if (c === totalAt) return column("total", "Score");
    if (c === stateAt) return column("state", "State");
    if (c === stepAt) return column("step", "Step");
    if (rawOf.has(c)) return column("raw", `${rawOf.get(c)} (raw)`, rawOf.get(c) ?? null);
    if (components.has(key)) return column("component");
    if (LIKELIHOOD_KEYS.includes(lower)) return column("likelihood", `${key} NLL`.replace(/ NLL NLL$/, " NLL"));
    const match = SUFFIX.exec(key);
    const owner = match && components.has(match[2]) ? match[2] : null;
    if ((/structure|pose|complex/i.test(key) && !numeric[c]) || looksLikeFiles(c)) return column("structure", "Structure", owner);
    if (owner) return column("meta", key, owner);
    return column(numeric[c] ? "meta" : "text");
  });
  const structureColumn = columns.findIndex((c) => c.role === "structure");
  const componentIndexes = columns.flatMap((c, i) => (c.role === "component" ? [i] : []));

  let failed = 0;
  const rows: RunRow[] = values.map((cells, id) => {
    const num = (c: number) => (c >= 0 && typeof cells[c] === "number" ? (cells[c] as number) : null);
    const state = num(stateAt);
    const total = num(totalAt);
    let failure: string | null = null;
    if (state === 0) failure = "invalid SMILES";
    else if (state === 2) failure = "duplicate";
    else if (totalAt >= 0 && total === null) failure = "no score";
    else {
      const broken = componentIndexes.filter((c) => cells[c] === null).map((c) => columns[c].key);
      if (broken.length) failure = `${broken.join(", ")} failed`;
    }
    if (failure) failed += 1;
    const smiles = smilesAt >= 0 && cells[smilesAt] !== null ? String(cells[smilesAt]) : "";
    const structure = structureColumn >= 0 && typeof cells[structureColumn] === "string" ? (cells[structureColumn] as string) : null;
    return { id, smiles, step: num(stepAt), total, state, failure, structure, cells };
  });

  return { columns, rows, components: [...components], failed, structureColumn };
}

// -- Viewing -----------------------------------------------------------------

/** What a column is sorted by: a column index, or the row's SMILES. */
export type SortKey = number | "smiles";

export interface Filter {
  query: string;
  minScore: number | null;
  showFailed: boolean;
  /** One row per SMILES, its best scoring one. */
  unique: boolean;
}

export function filterRows(table: RunTable, filter: Filter): RunRow[] {
  const needle = filter.query.trim().toLowerCase();
  const textColumns = table.columns.flatMap((c, i) => (c.role === "text" || c.role === "structure" ? [i] : []));
  let rows = table.rows.filter((row) => {
    if (!filter.showFailed && row.failure) return false;
    if (filter.minScore !== null && (row.total === null || row.total < filter.minScore)) return false;
    if (!needle) return true;
    if (row.smiles.toLowerCase().includes(needle) || row.failure?.toLowerCase().includes(needle)) return true;
    return textColumns.some((c) => typeof row.cells[c] === "string" && (row.cells[c] as string).toLowerCase().includes(needle));
  });
  if (filter.unique) {
    const best = new Map<string, RunRow>();
    for (const row of rows) {
      const seen = best.get(row.smiles);
      if (!seen || (row.total ?? -Infinity) > (seen.total ?? -Infinity)) best.set(row.smiles, row);
    }
    rows = rows.filter((row) => best.get(row.smiles) === row);
  }
  return rows;
}

const rank = (v: CellValue) => (v === null ? 3 : typeof v === "number" ? 0 : typeof v === "boolean" ? 1 : 2);

/** Sorted copy; empty values always last, whichever the direction. */
export function sortRows(rows: readonly RunRow[], key: SortKey, descending: boolean): RunRow[] {
  const value = (row: RunRow): CellValue => (key === "smiles" ? row.smiles : row.cells[key]);
  const sign = descending ? -1 : 1;
  return [...rows].sort((a, b) => {
    const x = value(a);
    const y = value(b);
    const rx = rank(x);
    const ry = rank(y);
    if (rx !== ry || rx === 3) return rx - ry || a.id - b.id;
    if (typeof x === "number" && typeof y === "number") return (x - y) * sign || a.id - b.id;
    return String(x).localeCompare(String(y)) * sign || a.id - b.id;
  });
}

/** Columns shown at first: score, step and components; likelihoods and metadata on demand. */
export function defaultColumns(table: RunTable): Set<string> {
  return new Set(table.columns.filter((c) => c.role === "total" || c.role === "step" || c.role === "component" || c.role === "raw").map((c) => c.key));
}

/** A value for the table; `integer` columns without decimals, others with a fixed few so they line up. */
export function formatCell(value: CellValue, integer = false): string {
  if (value === null) return "—";
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "number") {
    if (integer) return String(value);
    const abs = Math.abs(value);
    return abs !== 0 && (abs < 0.001 || abs >= 100_000) ? value.toExponential(2) : value.toFixed(abs >= 100 ? 1 : 3);
  }
  return value;
}

/** The Mol2Mol seeds a run conditioned on: its distinct Input_SMILES. */
export function runSeeds(table: RunTable, limit = 50): string[] {
  const at = table.columns.findIndex((c) => c.key.toLowerCase() === "input_smiles");
  if (at < 0) return [];
  const seeds = new Set<string>();
  for (const row of table.rows) {
    const v = row.cells[at];
    if (typeof v === "string" && v) seeds.add(v);
    if (seeds.size >= limit) break;
  }
  return [...seeds];
}

/** A row's numbers by column header, for looking up components and their raw values. */
export function rowValues(table: RunTable, row: RunRow): Record<string, number | null> {
  const out: Record<string, number | null> = {};
  table.columns.forEach((c, i) => {
    if (c.role === "component" || c.role === "raw" || c.role === "total") out[c.key] = typeof row.cells[i] === "number" ? (row.cells[i] as number) : null;
  });
  return out;
}

// -- Files -------------------------------------------------------------------

/** The rows as a sheet with every CSV column, plus why a row failed. */
export function resultsSheet(table: RunTable, rows: readonly RunRow[]): Sheet {
  const columns = [...table.columns.map((c) => c.key), "failure"];
  return {
    columns,
    rows: rows.map((row) => {
      const out: Record<string, Cell> = { failure: row.failure };
      table.columns.forEach((c, i) => {
        out[c.key] = row.cells[i];
      });
      return out;
    }),
  };
}

/** The rows as SD records, titled by SMILES, with every column as a data field. */
export function resultsSdf(table: RunTable, rows: readonly RunRow[], molblock: (smiles: string) => string | null, title: (row: RunRow, i: number) => string = (_, i) => `molecule ${i + 1}`): string {
  const sheet = resultsSheet(table, rows);
  return toSdf(rows.map((row, i) => ({ molblock: row.smiles ? molblock(row.smiles) : null, title: title(row, i), fields: sheet.rows[i] })));
}

/** A structure path from the CSV as a viewer panel, or null when the gateway cannot serve it. */
export function structurePanel(path: string, smiles: string | null, tool = "reinvent"): StructurePanel | null {
  const rel = sharedRelative(path);
  if (!rel || !STRUCTURE_FILE.test(rel)) return null;
  const name = rel.split("/").pop() ?? rel;
  return {
    kind: "structure",
    tool,
    name,
    path: rel,
    url: `/api/artifacts/${rel.split("/").map(encodeURIComponent).join("/")}`,
    format: /\.pdb$/i.test(name) ? "pdb" : "cif",
    ...(smiles ? { smiles } : {}),
  };
}
