// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { RunRow, RunTable } from "./runResults";

/**
 * A REINVENT run's results CSV as the Analysis view reads it: one entry per
 * unique valid molecule (its best scoring row), each endpoint as a column of
 * numbers (NaN when missing), and per step the mean, min and max of every
 * endpoint for the trajectory.
 */

export const TOTAL = "Score";

export interface Endpoint {
  /** The CSV header; the total is "Score". */
  key: string;
  label: string;
  kind: "total" | "component" | "raw";
}

export interface Molecules {
  n: number;
  smiles: string[];
  step: Int32Array;
  /** The CSV row each molecule came from, for its page. */
  rows: RunRow[];
  endpoints: Endpoint[];
  /** Endpoint key → one value per molecule, NaN when missing. */
  values: Map<string, Float64Array>;
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

/** Total, each component, then each component's raw value, in the CSV's order. */
export function endpointsOf(table: RunTable): Endpoint[] {
  const out: Endpoint[] = [];
  if (table.columns.some((c) => c.role === "total")) out.push({ key: TOTAL, label: "Total", kind: "total" });
  for (const c of table.columns) if (c.role === "component") out.push({ key: c.key, label: c.key, kind: "component" });
  for (const c of table.columns) if (c.role === "raw" && c.numeric) out.push({ key: c.key, label: c.label, kind: "raw" });
  return out;
}

function columnIndex(table: RunTable, endpoint: Endpoint): number {
  if (endpoint.kind === "total") return table.columns.findIndex((c) => c.role === "total");
  return table.columns.findIndex((c) => c.key === endpoint.key && c.role === endpoint.kind);
}

const usable = (row: RunRow) => !!row.smiles && row.state !== 0 && row.state !== 2;

/** One molecule per SMILES, its best total; invalid SMILES and in-batch duplicates left out. */
export function moleculesOf(table: RunTable): Molecules {
  const endpoints = endpointsOf(table);
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
  const values = new Map<string, Float64Array>();
  for (const endpoint of endpoints) {
    const at = columnIndex(table, endpoint);
    const column = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const v = at >= 0 ? rows[i].cells[at] : null;
      column[i] = finite(v) ? v : NaN;
    }
    values.set(endpoint.key, column);
  }
  const step = new Int32Array(n);
  rows.forEach((row, i) => (step[i] = row.step ?? 0));
  const smiles = rows.map((r) => r.smiles);
  return { n, smiles, step, rows, endpoints, values, index: new Map(smiles.map((s, i) => [s, i])), lastStep };
}

/** Per step, the mean, min and max of each endpoint over the valid rows. */
export function stepStats(table: RunTable, endpoints: readonly Endpoint[]): Map<string, StepStats[]> {
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
  const at = (q: number) => finiteSorted[Math.min(finiteSorted.length - 1, Math.max(0, Math.round((finiteSorted.length - 1) * q)))];
  const lo = at(0.02);
  const hi = at(0.98);
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
