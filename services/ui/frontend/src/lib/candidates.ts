// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { StructureMetrics } from "../types/api";
import type { Artifacts } from "./artifacts";
import { gate, type Gate } from "./gate";

/** One molecule the conversation produced, with everything known about it. */
export interface Candidate {
  smiles: string;
  name?: string;
  /** The tool that most recently reported this molecule. */
  origin: string;
  /** Every tool that reported it, newest first. */
  origins: string[];
  /** The total score, from a "score"/"total_score" column. */
  score?: number;
  /** Every other numeric column of the tables it appears in. */
  components: Record<string, number>;
  /** Boltz and PoseBusters values of its newest predicted complex. */
  metrics?: StructureMetrics;
  gate: Gate;
  /** Path of its newest predicted complex. */
  structurePath?: string;
}

/** Column names that hold a SMILES, in the order the gateway prefers them. */
const SMILES_COLUMNS = ["smiles", "SMILES", "canonical_smiles", "smi"];
const NAME_COLUMNS = ["name", "label", "id", "molecule_id"];
const SCORE_COLUMNS = ["total_score", "score", "Score", "total"];
/** Numeric columns that index a row rather than describe a molecule. */
const INDEX_COLUMNS = new Set(["#", "rank", "index", "step", "Step", "batch", "SMILES_state"]);

const isNumber = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);

/**
 * Candidates by SMILES, from every table with a SMILES column, every drawn
 * molecule and every structure whose ligand is known. Artifacts are newest
 * first, so the newest value of a column or metric wins.
 */
export function buildCandidates(artifacts: Pick<Artifacts, "tables" | "molecules" | "structures">): Candidate[] {
  const byKey = new Map<string, Candidate>();

  const ensure = (smiles: string, tool: string): Candidate => {
    let candidate = byKey.get(smiles);
    if (!candidate) {
      candidate = { smiles, origin: tool, origins: [], components: {}, gate: gate(undefined) };
      byKey.set(smiles, candidate);
    }
    if (!candidate.origins.includes(tool)) candidate.origins.push(tool);
    return candidate;
  };

  for (const { item, tool } of artifacts.tables) {
    const smilesColumn = SMILES_COLUMNS.find((column) => item.columns.includes(column));
    if (!smilesColumn) continue;
    const nameColumn = NAME_COLUMNS.find((column) => item.columns.includes(column));
    const scoreColumn = SCORE_COLUMNS.find((column) => item.columns.includes(column));
    for (const row of item.rows) {
      const raw = row[smilesColumn];
      if (typeof raw !== "string" || !raw.trim()) continue;
      const candidate = ensure(raw.trim(), tool);
      const name = nameColumn ? row[nameColumn] : undefined;
      if (candidate.name === undefined && (typeof name === "string" || isNumber(name)) && String(name).trim()) {
        candidate.name = String(name);
      }
      const score = scoreColumn ? row[scoreColumn] : undefined;
      if (candidate.score === undefined && isNumber(score)) candidate.score = score;
      for (const column of item.columns) {
        if (column === scoreColumn || column === nameColumn || INDEX_COLUMNS.has(column)) continue;
        const value = row[column];
        if (isNumber(value) && !(column in candidate.components)) candidate.components[column] = value;
      }
    }
  }

  for (const { item, tool } of artifacts.molecules) {
    if (!item.smiles.trim()) continue;
    const candidate = ensure(item.smiles.trim(), tool);
    if (candidate.name === undefined && item.label) candidate.name = item.label;
  }

  for (const { item, tool } of artifacts.structures) {
    if (!item.smiles?.trim()) continue;
    const candidate = ensure(item.smiles.trim(), tool);
    if (candidate.name === undefined && item.label) candidate.name = item.label;
    if (candidate.structurePath === undefined) {
      candidate.structurePath = item.path;
      if (item.metrics) {
        candidate.metrics = item.metrics;
        candidate.gate = gate(item.metrics);
      }
    }
  }

  return [...byKey.values()];
}

/** The metrics that are plotted and shown next to the score components. */
export const METRIC_LABELS: Partial<Record<keyof StructureMetrics, string>> = {
  confidence: "Confidence",
  ligandIptm: "Ligand ipTM",
  iptm: "ipTM",
  plddt: "pLDDT",
  affinity: "Affinity (log10 µM)",
  binderProbability: "Binder probability",
};

/** A numeric value of a candidate by column: "score", a component, or "metric:<key>". */
export function candidateValue(candidate: Candidate, key: string): number | undefined {
  if (key === "score") return candidate.score;
  if (key.startsWith("metric:")) {
    const value = candidate.metrics?.[key.slice(7) as keyof StructureMetrics];
    return isNumber(value) ? value : undefined;
  }
  return candidate.components[key];
}

/** A candidate's score and components by name, as the shortlist keeps them. */
export function candidateNumbers(candidate: Candidate): Record<string, number> {
  return { ...candidate.components, ...(candidate.score !== undefined ? { Score: candidate.score } : {}) };
}

/** Every numeric column at least one candidate has, as {key, label}. */
export function numericColumns(candidates: Candidate[]): { key: string; label: string }[] {
  const columns: { key: string; label: string }[] = [];
  if (candidates.some((c) => c.score !== undefined)) columns.push({ key: "score", label: "Score" });
  const components = new Set<string>();
  for (const c of candidates) for (const key of Object.keys(c.components)) components.add(key);
  for (const key of components) columns.push({ key, label: key });
  for (const [metric, label] of Object.entries(METRIC_LABELS)) {
    const key = `metric:${metric}`;
    if (candidates.some((c) => candidateValue(c, key) !== undefined)) columns.push({ key, label: label as string });
  }
  return columns;
}

/** The bar colour of a 0–1 value. */
export function scoreColor(value: number): string {
  if (value >= 0.7) return "#3ccf91";
  if (value >= 0.4) return "#f5a524";
  return "#ff4d4f";
}

const csvCell = (value: unknown): string => {
  const text = value === null || value === undefined ? "" : String(value);
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
};

/** Candidates as CSV, one column per score component and metric any of them has. */
export function candidatesToCsv(candidates: Candidate[], props?: (smiles: string) => MolProps | undefined): string {
  const components = [...new Set(candidates.flatMap((c) => Object.keys(c.components)))];
  const metrics = Object.keys(METRIC_LABELS) as (keyof StructureMetrics)[];
  const descriptors = props ? DESCRIPTOR_COLUMNS : [];
  const header = ["smiles", "name", "origin", "score", ...components, ...metrics, ...descriptors.map((d) => d.label), "gate", "structure"];
  const lines = candidates.map((c) =>
    [
      c.smiles,
      c.name,
      c.origin,
      c.score,
      ...components.map((key) => c.components[key]),
      ...metrics.map((key) => c.metrics?.[key]),
      ...descriptors.map((d) => {
        const value = descriptorValue(props?.(c.smiles), d.key);
        return value === undefined ? undefined : Number(value.toFixed(d.digits));
      }),
      c.gate.status,
      c.structurePath,
    ]
      .map(csvCell)
      .join(","),
  );
  return [header.join(","), ...lines].join("\n");
}

/** Descriptors computed in the browser with RDKit, keyed as in DESCRIPTOR_COLUMNS. */
export interface MolProps {
  mw: number;
  clogp: number;
  tpsa: number;
  hbd: number;
  hba: number;
  rotb: number;
  ro5: number;
}

/** Descriptor columns as {key, label, digits}; keys are "desc:<field of MolProps>". */
export const DESCRIPTOR_COLUMNS: { key: `desc:${keyof MolProps}`; label: string; digits: number }[] = [
  { key: "desc:mw", label: "MW", digits: 1 },
  { key: "desc:clogp", label: "cLogP", digits: 2 },
  { key: "desc:tpsa", label: "TPSA", digits: 1 },
  { key: "desc:hbd", label: "HBD", digits: 0 },
  { key: "desc:hba", label: "HBA", digits: 0 },
  { key: "desc:rotb", label: "RotB", digits: 0 },
  { key: "desc:ro5", label: "Ro5", digits: 0 },
];

/** Lipinski rule-of-five violations: MW > 500, cLogP > 5, HBD > 5, HBA > 10. */
export function ro5Violations(p: Pick<MolProps, "mw" | "clogp" | "hbd" | "hba">): number {
  return [p.mw > 500, p.clogp > 5, p.hbd > 5, p.hba > 10].filter(Boolean).length;
}

/** A descriptor value by "desc:<field>" key, or undefined. */
export function descriptorValue(props: MolProps | undefined, key: string): number | undefined {
  if (!props || !key.startsWith("desc:")) return undefined;
  const value = props[key.slice(5) as keyof MolProps];
  return isNumber(value) ? value : undefined;
}

const sdfField = (name: string, value: unknown) =>
  value === undefined || value === null || value === "" ? "" : `>  <${name}>\n${String(value)}\n\n`;

/**
 * Candidates as an SD file. *molblock* turns a SMILES into a V2000 block
 * (RDKit's get_molblock); molecules it cannot parse are skipped.
 */
export function candidatesToSdf(
  candidates: Candidate[],
  molblock: (smiles: string) => string | null,
  props: (smiles: string) => MolProps | undefined = () => undefined,
): string {
  const records: string[] = [];
  for (const c of candidates) {
    const block = molblock(c.smiles);
    if (!block) continue;
    const lines = block.replace(/\r\n/g, "\n").replace(/\n*$/, "").split("\n");
    if (c.name) lines[0] = c.name;
    const p = props(c.smiles);
    const fields = [
      sdfField("SMILES", c.smiles),
      sdfField("name", c.name),
      sdfField("origin", c.origin),
      sdfField("score", c.score),
      ...Object.entries(c.components).map(([key, value]) => sdfField(key, value)),
      ...DESCRIPTOR_COLUMNS.map((d) => sdfField(d.label, descriptorValue(p, d.key))),
      sdfField("gate", c.gate.status),
      sdfField("structure", c.structurePath),
    ].join("");
    records.push(`${lines.join("\n")}\n${fields}$$$$`);
  }
  return records.length ? `${records.join("\n")}\n` : "";
}

/** How many molecules a cofolding request names at most; each one is a Boltz run. */
export const MAX_COFOLD = 2;

/** A request to cofold the first MAX_COFOLD of the given molecules. */
export function cofoldPrompt(candidates: Candidate[]): string {
  const chosen = candidates.slice(0, MAX_COFOLD);
  const list = chosen.map((c) => `- ${c.name ? `${c.name}: ` : ""}${c.smiles}`).join("\n");
  const rest =
    candidates.length > chosen.length
      ? `\n\n(${candidates.length} were selected; cofold only these ${chosen.length} for now.)`
      : "";
  const noun = chosen.length === 1 ? "this molecule" : "these molecules";
  return `Cofold ${noun} with the current target protein and report confidence, ligand ipTM, PoseBusters and affinity:\n${list}${rest}`;
}

/** A request to predict ADMET properties of the given molecules. */
export function admetPrompt(candidates: Candidate[]): string {
  return `Predict ADMET properties for these molecules:\n${candidates.map((c) => `- ${c.smiles}`).join("\n")}`;
}
