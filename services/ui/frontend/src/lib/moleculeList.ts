// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { PropertiesPanel, StructureMetrics } from "../types/api";
import type { Sourced } from "./artifacts";
import type { Candidate } from "./candidates";
import { type Cell, type Sheet } from "./csv";
import { isMetricName, metricsFromValues, propertyKey } from "./moleculeFacts";
import { endpointLabel, endpointUnit, shortUnit } from "./properties";
import type { Descriptors } from "./rdkit";
import type { ShortlistEntry } from "./shortlist";

/** One molecule of the conversation, whichever tools reported it. */
export interface ListedMolecule {
  /** Canonical SMILES when known, what rows are merged by. */
  key: string;
  /** As first reported, for the drawing and the page. */
  smiles: string;
  name: string | null;
  /** Where it came from, in short words: "Cofolded", "ADMET", "Starred". */
  origins: string[];
  /** Every tool that reported it, for the tooltip. */
  tools: string[];
  starred: boolean;
  /** The run it was starred from, so its page can show the run's scores. */
  run: string | null;
  /** By ListColumn key. */
  values: Record<string, number>;
}

export interface ListColumn {
  key: string;
  label: string;
  unit: string | null;
  title: string;
  /** "fraction": a 0–1 value drawn on its own scale; "range": placed among the column's values. */
  bar: "fraction" | "range";
  tone: "score" | "cofold" | "admet" | "property";
}

export interface MoleculeSources {
  candidates: readonly Candidate[];
  properties: readonly Sourced<PropertiesPanel>[];
  starred: readonly ShortlistEntry[];
  /** Canonical SMILES; the input itself when it cannot be parsed or RDKit is not loaded. */
  key?: (smiles: string) => string;
  descriptors?: (smiles: string) => ListedDescriptors | undefined;
}

/** RDKit's, as rdkit.ts computes them. */
type ListedDescriptors = Pick<Descriptors, "mw" | "clogp" | "tpsa">;

/** `name`: what a star remembers the value as, read back by metricsFromValues. */
const COFOLD: { key: keyof StructureMetrics; name?: string; label: string; unit: string | null; bar: ListColumn["bar"] }[] = [
  { key: "confidence", name: "confidence", label: "Confidence", unit: null, bar: "fraction" },
  { key: "ligandIptm", name: "ligand_iptm", label: "Ligand ipTM", unit: null, bar: "fraction" },
  { key: "affinity", name: "affinity", label: "Affinity", unit: "log₁₀ µM", bar: "range" },
  { key: "binderProbability", label: "Binder prob.", unit: null, bar: "fraction" },
];
/** Labels as the shortlist and the candidates panel name them. */
const DESCRIPTORS: { key: keyof ListedDescriptors; label: string; unit: string | null }[] = [
  { key: "mw", label: "MW", unit: "Da" },
  { key: "clogp", label: "cLogP", unit: null },
  { key: "tpsa", label: "TPSA", unit: "Å²" },
];
/** Score components shown, the most common first; the page and the exports have the rest. */
const MAX_COMPONENTS = 3;
const RUN_ID = /^[0-9a-f]{8}$/;

const isNumber = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);

/** A short word for what a tool did with a molecule. */
export function originWord(tool: string): string {
  const t = tool.toLowerCase();
  if (/boltz|cofold|fold/.test(t)) return "Cofolded";
  if (/retrodmta/.test(t)) return "RetroDMTA";
  if (/admet/.test(t)) return "ADMET";
  if (/reinvent|mol2mol|generat|optimi|sampl|design/.test(t)) return "Generated";
  if (/score|rank|dock/.test(t)) return "Scored";
  if (/pubchem|chembl|lookup|search|fetch|resolve|name|drugbank|similar/.test(t)) return "Looked up";
  if (/draw|sketch|upload|user/.test(t)) return "Drawn";
  return tool.replace(/_/g, " ");
}

const serviceWord = (panel: PropertiesPanel) => (panel.service === "retrodmta" || panel.tool === "predict_retrodmta" ? "RetroDMTA" : "ADMET");

/** Score components worth a column: not a cofolding metric, a descriptor or a raw value shown elsewhere. */
const isComponent = (name: string) => !isMetricName(name) && !propertyKey(name) && !/\(raw\)$/.test(name);

/**
 * Every molecule of the conversation once, merged by canonical SMILES:
 * the candidates (tables, drawn molecules, cofolded ligands), the molecules
 * of predictions and the starred ones, in that order. Newer knowledge wins:
 * candidates and predictions come newest first, and what a star remembered
 * only fills what nothing else knows.
 */
export function listMolecules(sources: MoleculeSources): ListedMolecule[] {
  const keyOf = sources.key ?? ((smiles: string) => smiles);
  const byKey = new Map<string, ListedMolecule>();
  const ensure = (raw: string): ListedMolecule => {
    const smiles = raw.trim();
    const key = keyOf(smiles);
    let row = byKey.get(key);
    if (!row) {
      row = { key, smiles, name: null, origins: [], tools: [], starred: false, run: null, values: {} };
      byKey.set(key, row);
    }
    return row;
  };
  const add = (list: string[], value: string) => {
    if (!list.includes(value)) list.push(value);
  };
  const fill = (row: ListedMolecule, key: string, value: unknown) => {
    if (isNumber(value) && !(key in row.values)) row.values[key] = value;
  };
  const fillMetrics = (row: ListedMolecule, metrics: StructureMetrics | undefined) => {
    for (const m of COFOLD) fill(row, `metric:${m.key}`, metrics?.[m.key]);
  };

  for (const c of sources.candidates) {
    if (!c.smiles.trim()) continue;
    const row = ensure(c.smiles);
    row.name ??= c.name ?? null;
    if (c.metrics) add(row.origins, "Cofolded");
    for (const tool of c.origins) {
      add(row.tools, tool);
      add(row.origins, originWord(tool));
    }
    fill(row, "score", c.score);
    fillMetrics(row, c.metrics ?? metricsFromValues(c.components));
    for (const [name, value] of Object.entries(c.components)) if (isComponent(name)) fill(row, `c:${name}`, value);
  }

  for (const { item, tool } of sources.properties) {
    const service = serviceWord(item);
    for (const m of item.molecules) {
      const smiles = m.smiles ?? m.standardized;
      if (!smiles?.trim() || !m.ok) continue;
      const row = ensure(smiles);
      row.name ??= m.label;
      add(row.tools, tool);
      add(row.origins, service);
      for (const [endpoint, value] of Object.entries(m.values)) fill(row, `p:${service}:${endpoint}`, value.mean);
    }
  }

  for (const entry of sources.starred) {
    if (!entry.smiles.trim()) continue;
    const row = ensure(entry.smiles);
    row.starred = true;
    row.name ??= entry.label ?? null;
    if (entry.source && RUN_ID.test(entry.source)) row.run ??= entry.source;
    if (!row.origins.length) add(row.origins, row.run ? `Run ${row.run}` : "Starred");
    const values = entry.values ?? {};
    fill(row, "score", values.Score);
    fillMetrics(row, metricsFromValues(values));
    for (const [name, value] of Object.entries(values)) if (name !== "Score" && isComponent(name)) fill(row, `c:${name}`, value);
  }

  const rows = [...byKey.values()];
  if (sources.descriptors) {
    for (const row of rows) {
      const d = sources.descriptors(row.smiles);
      for (const desc of DESCRIPTORS) fill(row, `desc:${desc.key}`, d?.[desc.key]);
    }
  }
  return rows;
}

/** The columns at least one row has a value for: score, cofolding, the common score components, predictions, descriptors. */
export function listColumns(rows: readonly ListedMolecule[], properties: readonly Sourced<PropertiesPanel>[] = []): ListColumn[] {
  const has = (key: string) => rows.some((r) => key in r.values);
  const out: ListColumn[] = [];
  if (has("score")) out.push({ key: "score", label: "Score", unit: null, title: "The total score the agent gave it, 0–1", bar: "fraction", tone: "score" });
  for (const m of COFOLD) {
    const key = `metric:${m.key}`;
    if (has(key)) out.push({ key, label: m.label, unit: m.unit, title: `Boltz-2 ${m.label.toLowerCase()} of its predicted complex`, bar: m.bar, tone: "cofold" });
  }

  const counts = new Map<string, number>();
  for (const r of rows) for (const key of Object.keys(r.values)) if (key.startsWith("c:")) counts.set(key, (counts.get(key) ?? 0) + 1);
  const components = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, MAX_COMPONENTS);
  for (const [key] of components) out.push({ key, label: key.slice(2), unit: null, title: `Score component ${key.slice(2)}`, bar: "range", tone: "score" });

  const seen = new Set<string>();
  for (const { item } of properties) {
    const service = serviceWord(item);
    for (const endpoint of item.endpoints) {
      const key = `p:${service}:${endpoint}`;
      if (seen.has(key) || !has(key)) continue;
      seen.add(key);
      const unit = endpointUnit(item, endpoint);
      out.push({ key, label: endpointLabel(endpoint), unit: shortUnit(unit), title: `${service} ${endpoint}${unit ? ` — ${unit}` : ""}`, bar: "range", tone: "admet" });
    }
  }

  for (const d of DESCRIPTORS) {
    const key = `desc:${d.key}`;
    if (has(key)) out.push({ key, label: d.label, unit: d.unit, title: `${d.label}, computed with RDKit`, bar: "range", tone: "property" });
  }
  return out;
}

/** Rows matching the search (name, SMILES, origin or tool) and, when asked, starred. */
export function filterMolecules(rows: readonly ListedMolecule[], query: string, starredOnly: boolean): ListedMolecule[] {
  const q = query.trim().toLowerCase();
  return rows.filter(
    (r) => (!starredOnly || r.starred) && (!q || [r.name ?? "", r.smiles, ...r.origins, ...r.tools].join(" ").toLowerCase().includes(q)),
  );
}

/** Rows by a column, missing values last either way; the list order when unsorted. */
export function sortMolecules(rows: readonly ListedMolecule[], key: string | null, desc: boolean): ListedMolecule[] {
  if (!key) return [...rows];
  const value = (r: ListedMolecule): string | number | undefined => (key === "name" ? (r.name ?? undefined) : r.values[key]);
  return [...rows].sort((a, b) => {
    const va = value(a);
    const vb = value(b);
    if (va === undefined) return vb === undefined ? 0 : 1;
    if (vb === undefined) return -1;
    const order = typeof va === "number" && typeof vb === "number" ? va - vb : String(va).localeCompare(String(vb));
    return desc ? -order : order;
  });
}

/** [min, max] of a column over the rows, or null when there is nothing to place a value among. */
export function columnRange(rows: readonly ListedMolecule[], key: string): [number, number] | null {
  let min = Infinity;
  let max = -Infinity;
  for (const r of rows) {
    const v = r.values[key];
    if (v === undefined) continue;
    min = Math.min(min, v);
    max = Math.max(max, v);
  }
  return max > min ? [min, max] : null;
}

/** "0.82", "12.4", "412", "3": as many digits as the size of the number needs. */
export function tidyNumber(value: number): string {
  if (Number.isInteger(value) && Math.abs(value) < 1e5) return String(value);
  const size = Math.abs(value);
  return value.toFixed(size >= 100 ? 0 : size >= 10 ? 1 : 2);
}

/** What a star keeps of a row, by the names the shortlist and listMolecules read back. */
export function starValues(row: ListedMolecule): Record<string, number> {
  const out: Record<string, number> = {};
  const put = (name: string | undefined, value: number | undefined) => {
    if (name && value !== undefined) out[name] = value;
  };
  put("Score", row.values.score);
  for (const m of COFOLD) put(m.name, row.values[`metric:${m.key}`]);
  for (const [key, value] of Object.entries(row.values)) if (key.startsWith("c:")) put(key.slice(2), value);
  for (const d of DESCRIPTORS) put(d.label, row.values[`desc:${d.key}`]);
  return out;
}

/** The rows as a sheet for CSV and SD fields, one column per ListColumn. */
export function moleculeSheet(rows: readonly ListedMolecule[], columns: readonly ListColumn[]): Sheet {
  const header = (c: ListColumn) => (c.unit ? `${c.label} (${c.unit})` : c.label);
  return {
    columns: ["smiles", "name", "origin", "starred", ...columns.map(header)],
    rows: rows.map((r) => {
      const row: Record<string, Cell> = { smiles: r.smiles, name: r.name, origin: r.origins.join("; "), starred: r.starred };
      for (const c of columns) row[header(c)] = r.values[c.key];
      return row;
    }),
  };
}
