// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { propertyKey } from "./moleculeFacts";
import type { RunRow, RunTable } from "./runResults";

/**
 * A REINVENT run's results CSV as the Analysis view reads it: one entry per
 * unique valid molecule (its best scoring row), each endpoint as a column of
 * numbers (NaN when missing), and per step the mean, min and max of every
 * endpoint's score for the trajectory.
 *
 * An endpoint is the total or one score component. What the plots and the
 * table show of a component is its raw value in its own unit ("Affinity
 * (pIC50)", "MW (Da)"), the 0–1 score only when the CSV has no raw value.
 */

export const TOTAL = "Score";

/** Which side of an endpoint is better: higher, lower, a window, or not known. */
export type Direction = "up" | "down" | "target" | "none";
/** How values spread along an axis or a colour scale; log1p is log₁₀(1 + v), for counts with zeros. */
export type ScaleKind = "linear" | "log" | "log1p";

export interface Endpoint {
  /** The CSV header of the score; the total is "Score". */
  key: string;
  /** Short name ("Affinity", "MW"). */
  label: string;
  /** "pIC50", "Da", "" for unitless, "score" when only the 0–1 score is known. */
  unit: string;
  kind: "total" | "component";
  /** Where the values shown come from. */
  source: "total" | "raw" | "score";
  /** The ENDPOINT_TYPES entry it matched, or "other". */
  type: string;
  /** Decimals shown. */
  digits: number;
  scale: ScaleKind;
  /** The better side when the TTP does not tell. */
  better: Direction;
  /** Raw CSV value → value shown, when they differ. */
  convert?: (v: number) => number;
  /** The conversion reverses the order (pIC50 = 6 − log₁₀ IC50 in µM), so a TTP's direction flips. */
  flips?: boolean;
}

export interface Molecules {
  n: number;
  smiles: string[];
  step: Int32Array;
  /** The CSV row each molecule came from, for its page. */
  rows: RunRow[];
  endpoints: Endpoint[];
  /** Endpoint key → the value shown per molecule (raw, converted), NaN when missing. */
  values: Map<string, Float64Array>;
  /** Endpoint key → the 0–1 score per molecule. */
  scores: Map<string, Float64Array>;
  index: Map<string, number>;
  lastStep: number;
}

export interface StepStats {
  step: number;
  mean: number;
  min: number;
  max: number;
}

const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

interface EndpointType {
  id: string;
  /** On the component's name, lower case with spaces for underscores. */
  match: (name: string) => boolean;
  /** Short label; the component's own name when absent. */
  label?: string;
  unit: string;
  digits: number;
  /** "auto": log when the values are positive and span two decades or more. */
  scale: ScaleKind | "auto";
  better: Direction;
  convert?: (v: number) => number;
  flips?: boolean;
}

const property = (key: string) => (name: string) => propertyKey(name) === key;

/**
 * Raw values the view knows, first match wins: label, unit, decimals, the
 * axis and colour scale, and the better side when the run's TTP does not say
 * (its transform does: sigmoid up, reverse sigmoid down, double sigmoid a window).
 * Boltz writes affinity as log₁₀(IC50 / µM); it is shown as pIC50 = 6 − value.
 */
export const ENDPOINT_TYPES: readonly EndpointType[] = [
  { id: "affinity", match: (n) => /affinity/.test(n) && !/prob/.test(n), label: "Affinity", unit: "pIC50", digits: 1, scale: "linear", better: "up", convert: (v) => 6 - v, flips: true },
  { id: "potency", match: (n) => /\b(ic50|ec50|ki|kd)\b/.test(n), unit: "", digits: 0, scale: "log", better: "down" },
  { id: "qed", match: property("qed"), label: "QED", unit: "", digits: 2, scale: "linear", better: "up" },
  { id: "mw", match: property("mw"), label: "MW", unit: "Da", digits: 0, scale: "linear", better: "none" },
  { id: "logd", match: (n) => /\blogd\b/.test(n), label: "LogD", unit: "", digits: 1, scale: "linear", better: "none" },
  { id: "logp", match: property("clogp"), label: "cLogP", unit: "", digits: 1, scale: "linear", better: "none" },
  { id: "tpsa", match: property("tpsa"), label: "TPSA", unit: "Å²", digits: 1, scale: "linear", better: "none" },
  { id: "sa", match: property("sa"), label: "SA", unit: "", digits: 1, scale: "linear", better: "down" },
  { id: "hbd", match: property("hbd"), label: "HBD", unit: "", digits: 0, scale: "linear", better: "none" },
  { id: "hba", match: property("hba"), label: "HBA", unit: "", digits: 0, scale: "linear", better: "none" },
  { id: "rotb", match: property("rotb"), label: "RotB", unit: "", digits: 0, scale: "linear", better: "none" },
  { id: "solubility", match: (n) => /solub/.test(n), label: "Solubility", unit: "log M", digits: 1, scale: "linear", better: "up" },
  { id: "efflux", match: (n) => /efflux/.test(n), label: "Efflux", unit: "ratio", digits: 1, scale: "auto", better: "down" },
  { id: "permeability", match: (n) => /permeab|caco|papp|mdck/.test(n), label: "Permeability", unit: "log cm/s", digits: 2, scale: "auto", better: "up" },
  { id: "pose", match: (n) => /pose|posebusters/.test(n), unit: "", digits: 2, scale: "linear", better: "up" },
  { id: "confidence", match: (n) => /iptm|confidence|plddt/.test(n), unit: "", digits: 2, scale: "linear", better: "up" },
  { id: "contacts", match: (n) => /contact|interaction|h ?bond/.test(n), unit: "", digits: 0, scale: "auto", better: "up" },
];

const normalName = (name: string) => name.trim().toLowerCase().replace(/[_\s]+/g, " ");

/** The q-quantile of sorted values. */
const quantile = (sorted: readonly number[], q: number) => sorted[Math.min(sorted.length - 1, Math.max(0, Math.round((sorted.length - 1) * q)))];

/** Log when strictly positive over two decades or more; log1p for skewed counts with zeros. */
export function autoScale(values: ArrayLike<number>): ScaleKind {
  const sorted = Array.from(values).filter(Number.isFinite).sort((a, b) => a - b);
  if (sorted.length < 3) return "linear";
  const lo = quantile(sorted, 0.02);
  const hi = quantile(sorted, 0.98);
  if (sorted[0] > 0 && hi / lo >= 100) return "log";
  if (sorted[0] >= 0 && sorted.every(Number.isInteger) && hi >= 8 * Math.max(1, quantile(sorted, 0.5))) return "log1p";
  return "linear";
}

function autoDigits(values: ArrayLike<number>): number {
  const sorted = Array.from(values).filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length || sorted.every(Number.isInteger)) return 0;
  const span = Math.max(Math.abs(quantile(sorted, 0.98)), Math.abs(quantile(sorted, 0.02)));
  return span >= 100 ? 0 : span >= 10 ? 1 : span >= 1 ? 2 : 3;
}

/** A component's endpoint, from its name and its raw values (already converted), or as a score when there are none. */
export function describeEndpoint(key: string, raw: ArrayLike<number> | null): Endpoint {
  if (!raw) return { key, label: key, unit: "score", kind: "component", source: "score", type: "score", digits: 2, scale: "linear", better: "up" };
  const type = ENDPOINT_TYPES.find((t) => t.match(normalName(key)));
  let scale = type?.scale ?? "auto";
  if (scale === "auto") scale = autoScale(raw);
  let unit = type?.unit ?? "";
  if (type?.id === "permeability" && scale === "log") unit = "cm/s";
  if (type?.id === "potency") unit = /\bnm\b/i.test(key) ? "nM" : /\b(um|µm)\b/i.test(key) ? "µM" : "";
  return {
    key,
    label: type?.label ?? key,
    unit,
    kind: "component",
    source: "raw",
    type: type?.id ?? "other",
    digits: type ? type.digits : autoDigits(raw),
    scale,
    better: type?.better ?? "none",
    ...(type?.convert ? { convert: type.convert, flips: !!type.flips } : {}),
  };
}

const TOTAL_ENDPOINT: Endpoint = { key: TOTAL, label: "Total", unit: "", kind: "total", source: "total", type: "total", digits: 2, scale: "linear", better: "up" };

/** "Affinity (pIC50)", "QED", "Pose quality (score)". */
export const heading = (e: Pick<Endpoint, "label" | "unit">) => (e.unit ? `${e.label} (${e.unit})` : e.label);

function columnIndex(table: RunTable, endpoint: Pick<Endpoint, "key" | "kind">): number {
  if (endpoint.kind === "total") return table.columns.findIndex((c) => c.role === "total");
  return table.columns.findIndex((c) => c.key === endpoint.key && c.role === "component");
}

const usable = (row: RunRow) => !!row.smiles && row.state !== 0 && row.state !== 2;

/** One molecule per SMILES, its best total; invalid SMILES and in-batch duplicates left out. */
export function moleculesOf(table: RunTable): Molecules {
  const best = new Map<string, RunRow>();
  let lastStep = 0;
  for (const row of table.rows) {
    if (row.step !== null && row.step > lastStep) lastStep = row.step;
    if (!usable(row)) continue;
    const seen = best.get(row.smiles);
    if (!seen || (row.total ?? -Infinity) > (seen.total ?? -Infinity)) best.set(row.smiles, row);
  }
  const rows = [...best.values()];
  const n = rows.length;
  const column = (at: number) => {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const v = at >= 0 ? rows[i].cells[at] : null;
      out[i] = finite(v) ? v : NaN;
    }
    return out;
  };
  const endpoints: Endpoint[] = [];
  const values = new Map<string, Float64Array>();
  const scores = new Map<string, Float64Array>();
  const totalAt = table.columns.findIndex((c) => c.role === "total");
  if (totalAt >= 0) {
    const total = column(totalAt);
    endpoints.push(TOTAL_ENDPOINT);
    values.set(TOTAL, total);
    scores.set(TOTAL, total);
  }
  table.columns.forEach((c, at) => {
    if (c.role !== "component") return;
    const score = column(at);
    const rawAt = table.columns.findIndex((r) => r.role === "raw" && r.component === c.key);
    let raw = rawAt >= 0 ? column(rawAt) : null;
    if (raw && !raw.some(Number.isFinite)) raw = null;
    const convert = raw ? ENDPOINT_TYPES.find((t) => t.match(normalName(c.key)))?.convert : undefined;
    if (raw && convert) raw = raw.map(convert);
    endpoints.push(describeEndpoint(c.key, raw));
    values.set(c.key, raw ?? score);
    scores.set(c.key, score);
  });
  // Two components of one type ("Affinity" twice) go by their own names.
  for (const e of endpoints) if (endpoints.some((o) => o !== e && o.label === e.label)) e.label = e.key;
  const step = new Int32Array(n);
  rows.forEach((row, i) => (step[i] = row.step ?? 0));
  const smiles = rows.map((r) => r.smiles);
  return { n, smiles, step, rows, endpoints, values, scores, index: new Map(smiles.map((s, i) => [s, i])), lastStep };
}

// -- Numbers ---------------------------------------------------------------------

const SUPERSCRIPT = "⁰¹²³⁴⁵⁶⁷⁸⁹";
const power = (e: number) => `10${String(e).replace("-", "⁻").replace(/\d/g, (d) => SUPERSCRIPT[Number(d)])}`;
const minus = (text: string) => text.replace(/^-(?=.*[1-9])/, "−").replace(/^-/, "");

/**
 * A value with a fixed number of decimals, thousands grouped, a true minus;
 * on a log scale two significant digits, with powers of ten for the very
 * small or large (never "7.2e-4").
 */
export function formatValue(v: number, digits = 2, scale: ScaleKind = "linear"): string {
  if (!Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (scale === "log" && a !== 0) {
    const e = Math.floor(Math.log10(a));
    if (e < -3 || e >= 6) {
      const m = v / 10 ** e;
      return minus(`${Math.abs(m - Math.round(m)) < 0.05 ? Math.round(m) : m.toFixed(1)}×${power(e)}`).replace(/^1×/, "").replace(/^−1×/, "−");
    }
    return minus(a >= 100 ? Math.round(v).toLocaleString("en-US") : String(Number(v.toPrecision(2))));
  }
  if (a >= 10_000) return minus(Math.round(v).toLocaleString("en-US"));
  return minus(v.toFixed(digits));
}

export const formatEndpoint = (e: Pick<Endpoint, "digits" | "scale">, v: number) => formatValue(v, e.digits, e.scale);

/** Round ticks, 1, 2, 2.5 or 5 × 10ⁿ apart, inside [lo, hi]. */
export function niceTicks(lo: number, hi: number, count = 4): number[] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [];
  if (!(hi > lo)) return [lo];
  const rough = (hi - lo) / Math.max(1, count);
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= rough * 0.999) ?? magnitude * 10;
  const out: number[] = [];
  for (let t = Math.ceil(lo / step - 1e-9) * step; t <= hi + step * 1e-9; t += step) out.push(Number(t.toPrecision(12)));
  return out;
}

/** An axis: value → position (log₁₀ for log scales) and back. */
export function axisOf(kind: ScaleKind): { to: (v: number) => number; from: (a: number) => number } {
  if (kind === "log") return { to: (v) => (v > 0 ? Math.log10(v) : NaN), from: (a) => 10 ** a };
  if (kind === "log1p") return { to: (v) => (v > -1 ? Math.log10(1 + v) : NaN), from: (a) => 10 ** a - 1 };
  return { to: (v) => v, from: (a) => a };
}

/** Ticks in value units for an axis spanning [lo, hi] in axis units. */
export function axisTicks(kind: ScaleKind, lo: number, hi: number, count = 4): number[] {
  if (kind === "linear") return niceTicks(lo, hi, count);
  const { from } = axisOf(kind);
  if (kind === "log") {
    const decades: number[] = [];
    for (let k = Math.ceil(lo - 1e-9); k <= hi + 1e-9; k++) decades.push(k);
    const every = Math.max(1, Math.ceil(decades.length / count));
    const kept = decades.filter((_, i) => i % every === 0).map((k) => 10 ** k);
    return kept.length >= 2 ? kept : [from(lo), from(hi)];
  }
  const counts = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10_000].filter((c) => {
    const a = Math.log10(1 + c);
    return a >= lo - 1e-9 && a <= hi + 1e-9;
  });
  const every = Math.max(1, Math.ceil(counts.length / count));
  return counts.filter((_, i) => i % every === 0);
}

/** Per step, the mean, min and max of each endpoint's score over the valid rows. */
export function stepStats(table: RunTable, endpoints: readonly Pick<Endpoint, "key" | "kind">[]): Map<string, StepStats[]> {
  const out = new Map<string, StepStats[]>();
  for (const endpoint of endpoints) {
    const at = columnIndex(table, endpoint);
    if (at < 0) continue;
    const acc = new Map<number, { sum: number; count: number; min: number; max: number }>();
    for (const row of table.rows) {
      if (row.step === null || !usable(row)) continue;
      const v = row.cells[at];
      if (!finite(v)) continue;
      const a = acc.get(row.step);
      if (a) {
        a.sum += v;
        a.count += 1;
        if (v < a.min) a.min = v;
        if (v > a.max) a.max = v;
      } else acc.set(row.step, { sum: v, count: 1, min: v, max: v });
    }
    const list = [...acc.entries()].map(([step, a]) => ({ step, mean: a.sum / a.count, min: a.min, max: a.max })).sort((a, b) => a.step - b.step);
    if (list.length) out.set(endpoint.key, list);
  }
  return out;
}

/** A centred moving average over *window* points; NaN values are skipped. */
export function smooth(values: readonly number[], window: number): number[] {
  const half = Math.max(0, Math.floor(window / 2));
  if (half === 0) return [...values];
  return values.map((_, i) => {
    let sum = 0;
    let count = 0;
    for (let j = Math.max(0, i - half); j <= Math.min(values.length - 1, i + half); j++) {
      if (Number.isFinite(values[j])) {
        sum += values[j];
        count += 1;
      }
    }
    return count ? sum / count : NaN;
  });
}

/** A smoothing window that suits a run of *steps* steps: none for short runs. */
export const smoothWindow = (steps: number) => (steps < 12 ? 1 : Math.min(15, Math.max(3, Math.round(steps / 20)) | 1));

/**
 * The *n* components with the highest weight, in that order. TTPs often
 * weigh most objectives alike, so ties go to the component that varies most
 * across molecules (*spread*), then to the CSV's order. Names match
 * case-insensitively. Without weights, the most varied first.
 */
export function topByWeight(components: readonly string[], weights: readonly { name: string; weight: number }[], n = 3, spread?: ReadonlyMap<string, number>): string[] {
  const weightOf = new Map<string, number>();
  for (const w of weights) if (Number.isFinite(w.weight)) weightOf.set(w.name.trim().toLowerCase(), Math.max(weightOf.get(w.name.trim().toLowerCase()) ?? -Infinity, w.weight));
  const spreadOf = (name: string) => spread?.get(name) ?? 0;
  let ranked = components
    .map((name, order) => ({ name, order, weight: weightOf.get(name.trim().toLowerCase()) }))
    .filter((c) => c.weight !== undefined && c.weight > 0);
  if (!ranked.length) ranked = components.map((name, order) => ({ name, order, weight: 1 }));
  ranked.sort((a, b) => (b.weight ?? 0) - (a.weight ?? 0) || spreadOf(b.name) - spreadOf(a.name) || a.order - b.order);
  return ranked.slice(0, n).map((c) => c.name);
}

/** The standard deviation of each column's finite values. */
export function spreads(values: ReadonlyMap<string, Float64Array>): Map<string, number> {
  const out = new Map<string, number>();
  for (const [key, column] of values) {
    let n = 0;
    let mean = 0;
    let m2 = 0;
    for (const v of column) {
      if (!Number.isFinite(v)) continue;
      n += 1;
      const d = v - mean;
      mean += d / n;
      m2 += d * (v - mean);
    }
    out.set(key, n > 1 ? Math.sqrt(m2 / (n - 1)) : 0);
  }
  return out;
}

// -- Violins -------------------------------------------------------------------

export interface Density {
  lo: number;
  hi: number;
  /** Density at evenly spaced points from lo to hi, normalised so the peak is 1 (for the full set). */
  grid: Float64Array;
  count: number;
}

export interface Quartiles {
  q1: number;
  median: number;
  q3: number;
}

/** The finite values of a column, optionally only where *mask* is set. */
export function finiteValues(column: Float64Array, mask?: Uint8Array | null): number[] {
  const out: number[] = [];
  for (let i = 0; i < column.length; i++) if (Number.isFinite(column[i]) && (!mask || mask[i])) out.push(column[i]);
  return out;
}

export function extent(values: readonly number[]): [number, number] | null {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return lo <= hi ? [lo, hi] : null;
}

/** Silverman's rule of thumb, never narrower than a hundredth of the range. */
export function bandwidth(values: readonly number[], lo: number, hi: number): number {
  const n = values.length;
  const floor = (hi - lo) / 100 || 1e-6;
  if (n < 2) return Math.max(floor, (hi - lo) / 10);
  let mean = 0;
  for (const v of values) mean += v;
  mean /= n;
  let variance = 0;
  for (const v of values) variance += (v - mean) ** 2;
  const sd = Math.sqrt(variance / (n - 1));
  return Math.max(floor, 1.06 * sd * n ** -0.2);
}

/**
 * A Gaussian KDE of *values* on *points* grid points over [lo, hi], through a
 * 256-bin histogram so a hundred thousand values cost the same as a hundred.
 * *scale* divides the result (pass the full set's peak to draw a subset to the same scale).
 */
export function kde(values: readonly number[], lo: number, hi: number, h: number, points = 48, scale?: number): Float64Array {
  const grid = new Float64Array(points);
  if (!values.length) return grid;
  const span = hi - lo;
  if (span <= 0) {
    grid.fill(1);
    return grid;
  }
  const BINS = 256;
  const counts = new Float64Array(BINS);
  for (const v of values) counts[Math.min(BINS - 1, Math.max(0, Math.floor(((v - lo) / span) * BINS)))] += 1;
  for (let g = 0; g < points; g++) {
    const x = lo + (g / (points - 1)) * span;
    let sum = 0;
    for (let b = 0; b < BINS; b++) {
      if (!counts[b]) continue;
      const z = (x - (lo + ((b + 0.5) / BINS) * span)) / h;
      if (z > 4 || z < -4) continue;
      sum += counts[b] * Math.exp(-0.5 * z * z);
    }
    grid[g] = sum;
  }
  const peak = scale ?? Math.max(...grid);
  if (peak > 0) for (let g = 0; g < points; g++) grid[g] /= peak;
  return grid;
}

export function quartiles(values: readonly number[]): Quartiles | null {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const at = (q: number) => {
    const pos = (sorted.length - 1) * q;
    const base = Math.floor(pos);
    const next = sorted[Math.min(sorted.length - 1, base + 1)];
    return sorted[base] + (next - sorted[base]) * (pos - base);
  };
  return { q1: at(0.25), median: at(0.5), q3: at(0.75) };
}

/** The raw (unnormalised) peak of a KDE, to draw subsets on the same scale. */
export function kdePeak(values: readonly number[], lo: number, hi: number, h: number, points = 48): number {
  const grid = kde(values, lo, hi, h, points, 1);
  return Math.max(0, ...grid);
}

// -- Selection -----------------------------------------------------------------

export type Ranges = Readonly<Record<string, readonly [number, number]>>;

/**
 * Which molecules the current selection keeps: inside every endpoint range
 * (AND) and, when there is a region drawn on the map, in it. Null when
 * nothing is selected.
 */
export function selectionMask(molecules: Pick<Molecules, "n" | "values" | "smiles">, ranges: Ranges, region: ReadonlySet<string> | null): Uint8Array | null {
  const active = Object.entries(ranges).filter(([key]) => molecules.values.has(key));
  if (!active.length && !region) return null;
  const mask = new Uint8Array(molecules.n);
  for (let i = 0; i < molecules.n; i++) {
    let keep = !region || region.has(molecules.smiles[i]);
    for (let r = 0; keep && r < active.length; r++) {
      const [key, [lo, hi]] = active[r];
      const v = (molecules.values.get(key) as Float64Array)[i];
      keep = Number.isFinite(v) && v >= lo && v <= hi;
    }
    mask[i] = keep ? 1 : 0;
  }
  return mask;
}

export const countMask = (mask: Uint8Array | null) => (mask ? mask.reduce((sum, v) => sum + v, 0) : 0);

/** Even-odd ray casting; *polygon* as [x0, y0, x1, y1, ...]. */
export function inPolygon(x: number, y: number, polygon: ArrayLike<number>): boolean {
  let inside = false;
  const n = polygon.length / 2;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = polygon[2 * i];
    const yi = polygon[2 * i + 1];
    const xj = polygon[2 * j];
    const yj = polygon[2 * j + 1];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

export const inBox = (x: number, y: number, x0: number, y0: number, x1: number, y1: number) =>
  x >= Math.min(x0, x1) && x <= Math.max(x0, x1) && y >= Math.min(y0, y1) && y <= Math.max(y0, y1);

/** Indices of the points (xs[i], ys[i]) inside a lasso, or a box given by its two corners. */
export function pointsInside(xs: ArrayLike<number>, ys: ArrayLike<number>, shape: { kind: "box"; x0: number; y0: number; x1: number; y1: number } | { kind: "lasso"; polygon: ArrayLike<number> }): number[] {
  const out: number[] = [];
  for (let i = 0; i < xs.length; i++) {
    const hit = shape.kind === "box" ? inBox(xs[i], ys[i], shape.x0, shape.y0, shape.x1, shape.y1) : inPolygon(xs[i], ys[i], shape.polygon);
    if (hit) out.push(i);
  }
  return out;
}

// -- Map -------------------------------------------------------------------------

/** At most this many molecules are embedded. */
export const UMAP_CAP = 2500;

/** FNV-1a: a stable pseudo-random order, so the sample changes little as a run grows. */
export function hash(text: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/** A stable offset in [−1, 1) per text, so jittered points keep their place. */
export function jitter(text: string): number {
  let h = hash(text);
  h = Math.imul(h ^ (h >>> 16), 0x45d9f3b);
  h ^= h >>> 16;
  return (h >>> 0) / 2 ** 31 - 1;
}

/** The molecules to embed: all when few, else the best fifth of the cap by total plus a stable spread of the rest. */
export function sampleForMap(molecules: Pick<Molecules, "n" | "smiles" | "values">, cap = UMAP_CAP): number[] {
  const all = Array.from({ length: molecules.n }, (_, i) => i);
  if (molecules.n <= cap) return all;
  const total = molecules.values.get(TOTAL);
  const byScore = total ? [...all].sort((a, b) => (Number.isFinite(total[b]) ? total[b] : -Infinity) - (Number.isFinite(total[a]) ? total[a] : -Infinity)) : all;
  const top = byScore.slice(0, Math.floor(cap / 5));
  const taken = new Set(top);
  const rest = all.filter((i) => !taken.has(i)).sort((a, b) => hash(molecules.smiles[a]) - hash(molecules.smiles[b]));
  return [...top, ...rest.slice(0, cap - top.length)].sort((a, b) => a - b);
}

/** Steps a live run must advance before the map is computed again; more when the last one was slow. */
export const SLOW_MS = 8000;

export function shouldRecompute(state: { busy: boolean; computedStep: number | null; latestStep: number; lastMs: number | null; points: number; computedPoints: number | null }): boolean {
  if (state.busy || state.points < MIN_MAP_POINTS) return false;
  if (state.computedStep === null || state.computedPoints === null) return true;
  if (state.latestStep < state.computedStep) return true;
  const gap = state.lastMs !== null && state.lastMs > SLOW_MS ? 10 : 2;
  return state.latestStep - state.computedStep >= gap && state.points !== state.computedPoints;
}

export const MIN_MAP_POINTS = 8;

/** Per point, the density of points around it on a smoothed grid, 0..1. */
export function pointDensity(xs: ArrayLike<number>, ys: ArrayLike<number>, size = 48): Float64Array {
  const n = xs.length;
  const out = new Float64Array(n);
  if (!n) return out;
  const ex = extent(Array.from(xs)) ?? [0, 1];
  const ey = extent(Array.from(ys)) ?? [0, 1];
  const sx = ex[1] - ex[0] || 1;
  const sy = ey[1] - ey[0] || 1;
  const cell = (v: number, lo: number, span: number) => Math.min(size - 1, Math.max(0, Math.floor(((v - lo) / span) * size)));
  const grid = new Float64Array(size * size);
  for (let i = 0; i < n; i++) grid[cell(ys[i], ey[0], sy) * size + cell(xs[i], ex[0], sx)] += 1;
  // Two box blurs of radius 2 approximate a Gaussian.
  let a = grid;
  for (let pass = 0; pass < 2; pass++) {
    const b = new Float64Array(size * size);
    for (let y = 0; y < size; y++)
      for (let x = 0; x < size; x++) {
        let sum = 0;
        for (let dy = -2; dy <= 2; dy++)
          for (let dx = -2; dx <= 2; dx++) {
            const yy = y + dy;
            const xx = x + dx;
            if (yy >= 0 && yy < size && xx >= 0 && xx < size) sum += a[yy * size + xx];
          }
        b[y * size + x] = sum;
      }
    a = b;
  }
  let peak = 0;
  for (let i = 0; i < n; i++) {
    out[i] = a[cell(ys[i], ey[0], sy) * size + cell(xs[i], ex[0], sx)];
    if (out[i] > peak) peak = out[i];
  }
  if (peak > 0) for (let i = 0; i < n; i++) out[i] /= peak;
  return out;
}

/** The 2nd and 98th percentiles, so a few outliers do not wash out the colours. */
export function colourDomain(values: ArrayLike<number>): [number, number] | null {
  const finiteSorted = Array.from(values).filter(Number.isFinite).sort((a, b) => a - b);
  if (!finiteSorted.length) return null;
  const lo = quantile(finiteSorted, 0.02);
  const hi = quantile(finiteSorted, 0.98);
  return hi > lo ? [lo, hi] : [finiteSorted[0], finiteSorted[finiteSorted.length - 1] + (finiteSorted[0] === finiteSorted[finiteSorted.length - 1] ? 1 : 0)];
}

/** Viridis, from five stops. */
const STOPS: [number, number, number][] = [
  [68, 1, 84],
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 98],
  [253, 231, 37],
];

export function viridis(t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0)) * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(x));
  const f = x - i;
  const [a, b] = [STOPS[i], STOPS[i + 1]];
  return [Math.round(a[0] + (b[0] - a[0]) * f), Math.round(a[1] + (b[1] - a[1]) * f), Math.round(a[2] + (b[2] - a[2]) * f)];
}

/** The part of the scale used: the darkest purples vanish on a dark background. */
export const viridisFrom = (dark: boolean) => (dark ? 0.18 : 0);

/** Viridis from *from* to 1 as a CSS gradient, for legends. */
export function viridisCss(from = 0): string {
  const stops = Array.from({ length: 6 }, (_, i) => viridis(from + ((1 - from) * i) / 5));
  return `linear-gradient(to right, ${stops.map((c) => `rgb(${c.join(",")})`).join(", ")})`;
}

/** Below a window blue, in it yellow (the bright end, as for viridis), above it red. */
const DIVERGING: [number, number, number][] = [
  [62, 96, 184],
  [116, 164, 212],
  [246, 214, 92],
  [236, 132, 72],
  [184, 50, 66],
];

function interpolate(stops: readonly [number, number, number][], t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const [a, b] = [stops[i], stops[i + 1]];
  return [Math.round(a[0] + (b[0] - a[0]) * f), Math.round(a[1] + (b[1] - a[1]) * f), Math.round(a[2] + (b[2] - a[2]) * f)];
}

/** The transform of a TTP objective, as far as the direction goes. */
export interface TransformLike {
  type: string;
  low?: unknown;
  high?: unknown;
}

export interface Goal {
  direction: Direction;
  /** The window of a target, in the units shown. */
  target?: [number, number];
}

/**
 * Which side of an endpoint's values is better. Scores and the total: higher.
 * A raw value: from the TTP's transform (sigmoid and right step up; reverse
 * sigmoid, left step and exponential decay down; double sigmoid and step a
 * window), its type's usual side when the TTP is not known (*transform*
 * undefined), higher when the TTP scores it as it is (null).
 */
export function goalOf(e: Endpoint, transform: TransformLike | null | undefined): Goal {
  if (e.source !== "raw") return { direction: "up" };
  if (transform === undefined) return { direction: e.better };
  if (transform === null) return { direction: "up" };
  const type = String(transform.type).toLowerCase().replace(/[^a-z]/g, "");
  const low = finite(transform.low) ? transform.low : null;
  const high = finite(transform.high) ? transform.high : null;
  let goal: Goal;
  if (type === "sigmoid" || type === "rightstep") goal = { direction: "up" };
  else if (type === "reversesigmoid" || type === "leftstep" || type === "exponentialdecay") goal = { direction: "down" };
  else if ((type === "doublesigmoid" || type === "step") && low !== null && high !== null) goal = { direction: "target", target: [Math.min(low, high), Math.max(low, high)] };
  else return { direction: e.better };
  if (e.flips) {
    if (goal.direction === "up") goal = { direction: "down" };
    else if (goal.direction === "down") goal = { direction: "up" };
  }
  if (goal.target && e.convert) {
    const [a, b] = goal.target.map(e.convert);
    goal = { direction: "target", target: [Math.min(a, b), Math.max(a, b)] };
  }
  return goal;
}

export interface ColourScale {
  kind: ScaleKind;
  direction: Direction;
  /** Ends of the scale in axis units (log₁₀ for log scales): the 2nd and 98th percentiles, or symmetric about a window. */
  lo: number;
  hi: number;
  target?: [number, number];
  /** Tick values, in value units. */
  ticks: number[];
  /** Where a value sits on the legend, 0 (low) to 1 (high); NaN when missing. */
  position: (v: number) => number;
  /** The colour at a position on the legend. */
  paint: (p: number, dark: boolean) => [number, number, number];
  /** 0 to 1, higher is better (or later, or denser): the drawing order, so the best stay on top. */
  goodness: (v: number) => number;
}

/**
 * The colour scale of an endpoint over *values*: viridis, bright at the
 * better end (reversed when lower is better); for a window, diverging around
 * its middle, yellow inside, blue below, red above. Linear or log as the
 * endpoint's axis, with robust ends so a few outliers do not wash it out.
 */
export function colourScale(values: ArrayLike<number>, kind: ScaleKind, goal: Goal): ColourScale | null {
  const axis = axisOf(kind);
  const domain = colourDomain(Array.from(values, axis.to));
  if (!domain) return null;
  let [lo, hi] = domain;
  const target = goal.direction === "target" && goal.target ? goal.target : undefined;
  if (target) {
    const a = axis.to(target[0]);
    const b = axis.to(target[1]);
    if (Number.isFinite(a) && Number.isFinite(b)) {
      const centre = (a + b) / 2;
      const half = Math.max(Math.abs(lo - centre), Math.abs(hi - centre), ((b - a) / 2) * 1.25) || 1;
      lo = centre - half;
      hi = centre + half;
    }
  }
  const span = hi - lo || 1;
  const position = (v: number) => {
    const a = axis.to(v);
    return Number.isFinite(a) ? Math.min(1, Math.max(0, (a - lo) / span)) : NaN;
  };
  const diverging = !!target;
  const paint = (p: number, dark: boolean) => {
    if (diverging) return interpolate(DIVERGING, p);
    const from = viridisFrom(dark);
    return viridis(from + (1 - from) * (goal.direction === "down" ? 1 - p : p));
  };
  const goodness = (v: number) => {
    const p = position(v);
    if (!Number.isFinite(p)) return -1;
    return goal.direction === "down" ? 1 - p : diverging ? 1 - Math.abs(p - 0.5) * 2 : p;
  };
  const ticks = axisTicks(kind, lo, hi, 4).filter((t) => {
    const p = position(t);
    return p >= 0 && p <= 1;
  });
  return { kind, direction: goal.direction, lo, hi, ...(target ? { target } : {}), ticks, position, paint, goodness };
}

/** A colour scale as a CSS gradient, low to high. */
export function scaleCss(scale: Pick<ColourScale, "paint">, dark: boolean): string {
  const stops = Array.from({ length: 7 }, (_, i) => scale.paint(i / 6, dark));
  return `linear-gradient(to right, ${stops.map((c) => `rgb(${c.join(",")})`).join(", ")})`;
}

// -- Table -----------------------------------------------------------------------

/** Molecule indices kept by *mask* (all when null), sorted by *key* descending, NaN last. */
export function sortedIndices(molecules: Pick<Molecules, "n" | "values" | "step">, mask: Uint8Array | null, key: string, descending = true): number[] {
  const column = key === "step" ? null : molecules.values.get(key);
  const value = (i: number) => (column ? column[i] : key === "step" ? molecules.step[i] : NaN);
  const out: number[] = [];
  for (let i = 0; i < molecules.n; i++) if (!mask || mask[i]) out.push(i);
  const sign = descending ? -1 : 1;
  return out.sort((a, b) => {
    const x = value(a);
    const y = value(b);
    const fx = Number.isFinite(x);
    const fy = Number.isFinite(y);
    if (fx !== fy) return fx ? -1 : 1;
    if (!fx) return a - b;
    return (x - y) * sign || a - b;
  });
}
