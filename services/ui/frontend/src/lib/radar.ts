// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { PropertiesPanel } from "../types/api";
import { endpointLabel } from "./properties";

/**
 * Predicted properties as a radar: each endpoint becomes a 0-1 desirability
 * so that further out is always better and a bigger polygon is a better
 * profile.
 *
 * Ranges follow common medicinal-chemistry rules of thumb for the units the
 * ADMET service reports (admet-predictor/src/admet_predictor/core.py).
 */

export type Direction = "up" | "down" | "window";

export interface AxisRule {
  direction: Direction;
  /** Value scored 0 ("up", and the low edge of a window) ... */
  lo: number;
  /** ... value scored 1 ("up"; 0 for "down" and the high edge of a window). */
  hi: number;
  /** For "window": the values scored 1. */
  ideal?: [number, number];
  /** Why the range, for the legend. */
  note: string;
  /** Scaled between the shown molecules because no absolute range is known. */
  relative?: boolean;
}

const RULES: Record<string, AxisRule> = {
  logp: { direction: "window", lo: -1, hi: 6, ideal: [1, 3], note: "best 1–3; 0 at ≤ −1 or ≥ 6" },
  logd: { direction: "window", lo: -1, hi: 5, ideal: [1, 3], note: "best 1–3 at pH 7.4; 0 at ≤ −1 or ≥ 5" },
  aqueous_solubility: { direction: "up", lo: -6, hi: -3, note: "log mol/L: 0 at ≤ 1 µM, 1 at ≥ 1 mM" },
  kinetic_solubility: { direction: "up", lo: -6, hi: -3, note: "log mol/L: 0 at ≤ 1 µM, 1 at ≥ 1 mM" },
  caco2_permeability: { direction: "up", lo: -6.5, hi: -4.5, note: "log cm/s: 0 at ≤ 3×10⁻⁷, 1 at ≥ 3×10⁻⁵" },
  efflux_mdck: { direction: "down", lo: 0.3, hi: 1.3, note: "log ratio: 1 at ER ≤ 2, 0 at ER ≥ 20" },
  efflux_caco2: { direction: "down", lo: 0.3, hi: 1.3, note: "log ratio: 1 at ER ≤ 2, 0 at ER ≥ 20" },
};

const POTENCY: AxisRule = { direction: "up", lo: 4, hi: 9, note: "0 at ≤ 4 (100 µM), 1 at ≥ 9 (1 nM)" };

/** The rule for an endpoint; RetroDMTA endpoints are matched by suffix ("FXA_pKi"). */
export function axisRule(endpoint: string, values: number[] = []): AxisRule {
  const key = endpoint.toLowerCase();
  if (RULES[key]) return RULES[key];
  if (/(^|_)p(ki|kd|ic50|ec50)$/.test(key)) return POTENCY;
  if (/(^|_)logd$/.test(key)) return RULES.logd;
  if (/(^|_)logs$/.test(key)) return RULES.aqueous_solubility;
  const finite = values.filter(Number.isFinite);
  const lo = finite.length ? Math.min(...finite) : 0;
  const hi = finite.length ? Math.max(...finite) : 1;
  return { direction: "up", lo, hi: hi > lo ? hi : lo + 1, note: "no reference range: scaled between the molecules shown", relative: true };
}

const clamp = (v: number) => Math.min(1, Math.max(0, v));

/** A value as 0-1, further out being better. */
export function desirability(rule: AxisRule, value: number): number {
  if (!Number.isFinite(value)) return 0;
  const span = rule.hi - rule.lo || 1;
  if (rule.direction === "up") return clamp((value - rule.lo) / span);
  if (rule.direction === "down") return clamp((rule.hi - value) / span);
  const [a, b] = rule.ideal ?? [rule.lo, rule.hi];
  if (value >= a && value <= b) return 1;
  if (value < a) return a > rule.lo ? clamp((value - rule.lo) / (a - rule.lo)) : 0;
  return rule.hi > b ? clamp((rule.hi - value) / (rule.hi - b)) : 0;
}

export function describeRule(rule: AxisRule): string {
  const arrow = rule.direction === "up" ? "higher is better" : rule.direction === "down" ? "lower is better" : "best inside a window";
  return `${arrow}: ${rule.note}`;
}

export interface RadarAxis {
  endpoint: string;
  label: string;
  rule: AxisRule;
}

export interface RadarSeries {
  label: string;
  smiles: string | null;
  /** Desirability per axis; null where the molecule has no value. */
  scores: (number | null)[];
  /** The predicted means, per axis. */
  raw: (number | null)[];
}

/** Axes and polygons for the first `max` molecules with predictions; null with fewer than three axes. */
export function radarData(panel: PropertiesPanel, max = 5): { axes: RadarAxis[]; series: RadarSeries[] } | null {
  const molecules = panel.molecules.filter((m) => m.ok);
  const axes = panel.endpoints
    .filter((endpoint) => molecules.some((m) => m.values[endpoint] !== undefined))
    .map((endpoint) => ({
      endpoint,
      label: endpointLabel(endpoint),
      rule: axisRule(
        endpoint,
        molecules.map((m) => m.values[endpoint]?.mean).filter((v): v is number => v !== undefined),
      ),
    }));
  if (axes.length < 3 || molecules.length === 0) return null;
  const series = molecules.slice(0, max).map((m, i) => {
    const raw = axes.map((axis) => m.values[axis.endpoint]?.mean ?? null);
    return {
      label: m.label ?? m.smiles ?? `#${i + 1}`,
      smiles: m.smiles,
      raw,
      scores: raw.map((value, j) => (value === null ? null : desirability(axes[j].rule, value))),
    };
  });
  return { axes, series };
}

/** The point at radius `r` (0-1) on axis `i` of `n`, the first axis pointing up. */
export function radarPoint(i: number, n: number, r: number, cx: number, cy: number, radius: number): [number, number] {
  const angle = -Math.PI / 2 + (2 * Math.PI * i) / n;
  return [cx + Math.cos(angle) * r * radius, cy + Math.sin(angle) * r * radius];
}
