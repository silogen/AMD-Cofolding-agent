// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { StructureMetrics } from "../types/api";
import { cofoldFacts, endpointKey, type Fact, type FactGroup } from "./moleculeFacts";

/** How full a value's bar is (0–1) and whether the value looks fine. */
export interface Bar {
  fraction: number;
  tone: "ok" | "warn" | "plain";
}

export type BarFact = Fact & { bar?: Bar };

export interface BarGroup {
  title: string;
  facts: BarFact[];
}

const clamp = (value: number) => Math.min(1, Math.max(0, value));
/** The value as a number; a "mean ± std" by its mean. */
const numberOf = (fact: Fact) => (fact.value === null ? NaN : Number(fact.value.replace(/^−/, "-").replace(/\s*±.*$/, "")));

/** The cofold values of the molecule page, each with a bar: scores as they are, affinity on pIC50 4–9, PoseBusters as checks passed. */
export function cofoldBars(m: StructureMetrics): BarFact[] {
  return cofoldFacts(m).map((fact) => {
    const tone = fact.tone === "warn" ? "warn" : "ok";
    if (fact.label === "Affinity" && m.affinity !== undefined) {
      const p = 6 - m.affinity;
      return { ...fact, bar: { fraction: clamp((p - 4) / 5), tone: p >= 6 ? "ok" : "warn" } };
    }
    if (fact.label === "PoseBusters" && m.posebustersTotal) return { ...fact, bar: { fraction: clamp((m.posebustersPassed ?? 0) / m.posebustersTotal), tone } };
    const value = numberOf(fact);
    return Number.isFinite(value) ? { ...fact, bar: { fraction: clamp(value), tone } } : fact;
  });
}

/** Where a property's bar ends, and the drug-like side of it (Lipinski, Veber, QED, SA). */
const PROPERTY_SCALES: Record<string, { min?: number; max: number; ok: (v: number) => boolean }> = {
  QED: { max: 1, ok: (v) => v >= 0.5 },
  MW: { max: 700, ok: (v) => v <= 500 },
  cLogP: { min: -2, max: 7, ok: (v) => v <= 5 },
  TPSA: { max: 200, ok: (v) => v <= 140 },
  SA: { max: 10, ok: (v) => v <= 6 },
  HBD: { max: 10, ok: (v) => v <= 5 },
  HBA: { max: 15, ok: (v) => v <= 10 },
  RotB: { max: 15, ok: (v) => v <= 10 },
};

/** Properties (propertyFacts) with bars, amber outside the drug-like range. */
export function propertyBars(facts: Fact[]): BarFact[] {
  return facts.map((fact) => {
    const scale = PROPERTY_SCALES[fact.label];
    const value = numberOf(fact);
    if (!scale || !Number.isFinite(value)) return fact;
    const min = scale.min ?? 0;
    const ok = scale.ok(value);
    return { ...fact, ...(ok ? {} : { tone: "warn" as const }), bar: { fraction: clamp((value - min) / (scale.max - min)), tone: ok ? "ok" : "warn" } };
  });
}

/** A value with a bar when it is a unitless 0–1 (a probability, a desirability). */
function unitBar(fact: Fact): BarFact {
  const value = numberOf(fact);
  return !fact.unit && value >= 0 && value <= 1 ? { ...fact, bar: { fraction: value, tone: "plain" } } : fact;
}

/**
 * The ADMET endpoints' ranges (by endpointKey) and their usual favourable
 * side: logS ≥ −6, Caco-2 log Papp ≥ −5.15, efflux ratio < 2.
 */
const ENDPOINT_SCALES: Record<string, { min: number; max: number; ok: (v: number) => boolean }> = {
  logp: { min: -2, max: 7, ok: (v) => v <= 5 },
  logd: { min: -2, max: 7, ok: (v) => v <= 5 },
  aqueoussolubility: { min: -8, max: 0, ok: (v) => v >= -6 },
  kineticsolubility: { min: -8, max: 0, ok: (v) => v >= -6 },
  caco2permeability: { min: -7, max: -4, ok: (v) => v >= -5.15 },
  effluxmdck: { min: -1, max: 2, ok: (v) => v < Math.log10(2) },
  effluxcaco2: { min: -1, max: 2, ok: (v) => v < Math.log10(2) },
};

/** A prediction with a bar: on its endpoint's range, amber on the unfavourable side; else when it is a unitless 0–1. */
function predictionBar(fact: Fact): BarFact {
  const scale = ENDPOINT_SCALES[endpointKey(fact.endpoint ?? fact.label)];
  const value = numberOf(fact);
  if (!scale || !Number.isFinite(value)) return unitBar(fact);
  const ok = scale.ok(value);
  return { ...fact, ...(ok ? {} : { tone: "warn" as const }), bar: { fraction: clamp((value - scale.min) / (scale.max - scale.min)), tone: ok ? "ok" : "warn" } };
}

export function predictionBars(groups: FactGroup[]): BarGroup[] {
  return groups.map((group) => ({ title: group.title, facts: group.facts.map(predictionBar) }));
}

/** The total score, then its components (componentFacts), all desirabilities on 0–1. */
export function scoreBars(score: number | null | undefined, components: Fact[]): BarFact[] {
  return [...(typeof score === "number" ? [{ label: "Total", value: score.toFixed(2) }] : []), ...components].map(unitBar);
}
