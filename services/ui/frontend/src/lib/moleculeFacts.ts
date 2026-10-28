// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { PropertiesPanel, StructureMetrics } from "../types/api";
import type { Sourced } from "./artifacts";
import { GATE } from "./gate";
import { endpointLabel, formatValue, serviceName, shortUnit } from "./properties";
import type { Descriptors } from "./rdkit";

/**
 * What is known about one molecule, as short facts ("MW 412.6 Da") grouped
 * the way the molecule page and the structure details show them.
 */

export interface Fact {
  label: string;
  /** Already formatted, or null when unknown. */
  value: string | null;
  unit?: string;
  /** The full story on hover: raw value, rule, spread. */
  title?: string;
  /** "warn" when the value fails the quality gate. */
  tone?: "warn";
  /** The service's name for a predicted endpoint ("aqueous_solubility"). */
  endpoint?: string;
}

export interface FactGroup {
  title: string;
  facts: Fact[];
}

/** A score component: its desirability (0–1) and, when known, the raw value it came from. */
export interface ScoreComponent {
  name: string;
  value: number | null;
  raw?: number | null;
  /** A cofolding or pose component: its raw value is a Boltz or PoseBusters number, not a property. */
  cofold?: boolean;
}

const fixed = (value: number, digits: number) => value.toFixed(digits).replace(/^-/, "−");

/** A number with as many decimals as its size needs. */
export function shortNumber(value: number): string {
  return fixed(value, Math.abs(value) >= 10 ? 1 : 2);
}

/** Boltz affinity, log10(IC50 / 1 µM), as a concentration: { value: "22", unit: "nM" }. */
export function ic50(log10Micromolar: number): { value: string; unit: string } {
  const micromolar = 10 ** log10Micromolar;
  const [amount, unit] = micromolar >= 1000 ? [micromolar / 1000, "mM"] : micromolar >= 1 ? [micromolar, "µM"] : [micromolar * 1000, "nM"];
  return { value: amount >= 100 ? amount.toFixed(0) : amount.toPrecision(2), unit };
}

/** Affinity, ligand ipTM, confidence and PoseBusters, each marked when it fails the gate. */
export function cofoldFacts(m: StructureMetrics): Fact[] {
  const facts: Fact[] = [];
  if (m.affinity !== undefined) {
    const { value, unit } = ic50(m.affinity);
    const split = m.affinity1 !== undefined && m.affinity2 !== undefined && Math.abs(m.affinity1 - m.affinity2) > 1;
    facts.push({
      label: "Affinity",
      value,
      unit,
      title: `Predicted IC50 · log₁₀ µM ${fixed(m.affinity, 2)} · pIC50 ${(6 - m.affinity).toFixed(2)}${split ? ` · the two models disagree (${fixed(m.affinity1 as number, 2)}, ${fixed(m.affinity2 as number, 2)})` : ""}`,
    });
  }
  if (m.ligandIptm !== undefined) {
    facts.push({ label: "Ligand ipTM", value: m.ligandIptm.toFixed(2), title: `Gate > ${GATE.ligandIptm}`, ...(m.ligandIptm > GATE.ligandIptm ? {} : { tone: "warn" as const }) });
  }
  if (m.confidence !== undefined) {
    facts.push({ label: "Confidence", value: m.confidence.toFixed(2), title: `Gate > ${GATE.confidence}`, ...(m.confidence > GATE.confidence ? {} : { tone: "warn" as const }) });
  }
  if (m.posebustersPassed !== undefined && m.posebustersTotal !== undefined) {
    facts.push({
      label: "PoseBusters",
      value: `${m.posebustersPassed}/${m.posebustersTotal}`,
      title: m.failedChecks?.length ? `Failed: ${m.failedChecks.join(", ")}` : "Checks passed",
      ...(m.posebustersPassed === m.posebustersTotal ? {} : { tone: "warn" as const }),
    });
  }
  return facts;
}

/** Molecular properties the page knows by name, in the order it shows them. */
const PROPERTIES: { key: string; label: string; unit?: string; digits: number; names: RegExp }[] = [
  { key: "qed", label: "QED", digits: 2, names: /^qed$/ },
  { key: "mw", label: "MW", unit: "Da", digits: 1, names: /^(mw|molecular weight|mol ?wt|molwt|exact mol wt)$/ },
  { key: "clogp", label: "cLogP", digits: 2, names: /^(c?logp|slogp|alogp|crippen ?c?logp)$/ },
  { key: "tpsa", label: "TPSA", unit: "Å²", digits: 1, names: /^tpsa$/ },
  { key: "sa", label: "SA", digits: 1, names: /^(sa|sa ?score|sascore|synthetic accessibility)$/ },
  { key: "hbd", label: "HBD", digits: 0, names: /^(hbd|num ?h ?donors|h-?bond donors)$/ },
  { key: "hba", label: "HBA", digits: 0, names: /^(hba|num ?h ?acceptors|h-?bond acceptors)$/ },
  { key: "rotb", label: "RotB", digits: 0, names: /^(rotb|num ?rotatable ?bonds|rotatable bonds)$/ },
];

const normal = (name: string) => name.trim().toLowerCase().replace(/[_\s]+/g, " ").replace(/\s*\(raw\)$/, "");

/** The property a column or component name stands for ("Molecular weight (raw)" → "mw"), or null. */
export function propertyKey(name: string): string | null {
  const n = normal(name);
  return PROPERTIES.find((p) => p.names.test(n))?.key ?? null;
}

/**
 * QED, MW, cLogP, TPSA, SA... from what the conversation computed (`values`,
 * by column or component name) and, for what it did not, from RDKit.
 */
export function propertyFacts(values: Record<string, number | null | undefined>, rdkit?: Descriptors | null): Fact[] {
  const known = new Map<string, number>();
  for (const [name, value] of Object.entries(values)) {
    const key = propertyKey(name);
    if (key && typeof value === "number" && Number.isFinite(value) && !known.has(key)) known.set(key, value);
  }
  if (rdkit) {
    const computed: Record<string, number> = { mw: rdkit.mw, clogp: rdkit.clogp, tpsa: rdkit.tpsa, hbd: rdkit.hbd, hba: rdkit.hba, rotb: rdkit.rotb };
    for (const [key, value] of Object.entries(computed)) if (!known.has(key) && Number.isFinite(value)) known.set(key, value);
  }
  return PROPERTIES.flatMap((p) => {
    const value = known.get(p.key);
    return value === undefined ? [] : [{ label: p.label, value: fixed(value, p.digits), ...(p.unit ? { unit: p.unit } : {}) }];
  });
}

/** The raw values of the components that are neither properties above nor cofolding, by name ("Aqueous solubility": −4.1). */
export function otherRawValues(components: readonly ScoreComponent[]): Fact[] {
  return components.flatMap((c) =>
    c.cofold || c.raw === null || c.raw === undefined || propertyKey(c.name) ? [] : [{ label: c.name, value: shortNumber(c.raw) }],
  );
}

/** Score components as desirabilities, with the raw value on hover. */
export function componentFacts(components: readonly ScoreComponent[]): Fact[] {
  return components.map((c) => ({
    label: c.name,
    value: c.value === null ? null : shortNumber(c.value),
    ...(c.raw !== null && c.raw !== undefined ? { title: `Raw ${shortNumber(c.raw)} → desirability` } : {}),
  }));
}

/**
 * ADMET and RetroDMTA predictions of `smiles` in the conversation, one group
 * per service, newest first. `key` (canonical SMILES) matches a molecule the
 * service wrote differently.
 */
export function predictionGroups(smiles: string, panels: readonly Sourced<PropertiesPanel>[], key?: (smiles: string) => string): FactGroup[] {
  const groups: FactGroup[] = [];
  const wanted = key?.(smiles);
  const same = (s: string | null | undefined) => !!s && (s === smiles || (wanted !== undefined && key?.(s) === wanted));
  for (const { item } of panels) {
    const molecule = item.molecules.find((m) => m.ok && (m.smiles === smiles || m.standardized === smiles)) ?? item.molecules.find((m) => m.ok && (same(m.smiles) || same(m.standardized)));
    if (!molecule) continue;
    const title = serviceName(item);
    if (groups.some((g) => g.title === title)) continue;
    groups.push({
      title,
      facts: item.endpoints.flatMap((endpoint) => {
        const value = molecule.values[endpoint];
        if (!value) return [];
        const unit = shortUnit(value.unit);
        return [{ label: endpointLabel(endpoint), endpoint, value: formatValue(value).replace(/^-/, "−"), ...(unit ? { unit } : {}), ...(value.unit ? { title: value.unit } : {}) }];
      }),
    });
  }
  return groups.filter((g) => g.facts.length > 0);
}

/** An endpoint's name for comparison: "aqueous_solubility", "Aqueous solubility" and "Aq. solubility" are one. */
export const endpointKey = (name: string) => name.toLowerCase().replace(/\(raw\)$/, "").replace(/[^a-z0-9]/g, "");

/**
 * A run's raw values of the components that are neither properties nor
 * cofolding (otherRawValues), less the endpoints a prediction service in
 * `groups` already gave for the molecule.
 */
export function runPredictions(components: readonly ScoreComponent[], groups: readonly FactGroup[]): Fact[] {
  const known = new Set(groups.flatMap((g) => g.facts.flatMap((f) => [endpointKey(f.label), ...(f.endpoint ? [endpointKey(f.endpoint)] : [])])));
  return otherRawValues(components).filter((fact) => !known.has(endpointKey(fact.label)));
}

/** Metric names a results table or a scoring tool uses for the Boltz and PoseBusters values. */
const METRIC_NAMES: Record<string, keyof StructureMetrics> = {
  confidence: "confidence",
  confidence_score: "confidence",
  ligand_iptm: "ligandIptm",
  iptm: "iptm",
  complex_plddt: "plddt",
  affinity: "affinity",
  affinity_pred_value: "affinity",
  affinity_pred_value1: "affinity1",
  affinity_pred_value2: "affinity2",
  n_checks_passed: "posebustersPassed",
  posebusters_passed: "posebustersPassed",
  n_checks_total: "posebustersTotal",
  posebusters_total: "posebustersTotal",
};

/** Boltz and PoseBusters values found among a table row's numbers, for a molecule cofolded by a tool that only returned a table. */
export function metricsFromValues(values: Record<string, number | null | undefined>): StructureMetrics | undefined {
  const out: Record<string, number> = {};
  for (const [name, value] of Object.entries(values)) {
    const key = METRIC_NAMES[name.trim().toLowerCase()];
    if (key && typeof value === "number" && !(key in out)) out[key] = value;
  }
  return Object.keys(out).length ? (out as StructureMetrics) : undefined;
}

/** Whether a column name is one of those metrics, so it is not listed again as a score component. */
export const isMetricName = (name: string) => name.trim().toLowerCase() in METRIC_NAMES;
