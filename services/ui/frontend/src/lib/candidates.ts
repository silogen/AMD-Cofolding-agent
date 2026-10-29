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

type CandidateSources = Pick<Artifacts, "tables" | "molecules" | "structures">;
let last: { sources: CandidateSources; candidates: Candidate[] } | null = null;

/**
 * buildCandidates, computed once for the lists it reads: the tab badge and
 * the panels showing candidates all ask for the same artifacts.
 */
export function candidatesOf(artifacts: CandidateSources): Candidate[] {
  if (last && last.sources.tables === artifacts.tables && last.sources.molecules === artifacts.molecules && last.sources.structures === artifacts.structures) {
    return last.candidates;
  }
  const candidates = buildCandidates(artifacts);
  last = { sources: { tables: artifacts.tables, molecules: artifacts.molecules, structures: artifacts.structures }, candidates };
  return candidates;
}

/**
 * Candidates by SMILES, from every table with a SMILES column, every drawn
 * molecule and every structure whose ligand is known. Artifacts are newest
 * first, so the newest value of a column or metric wins.
 */
export function buildCandidates(artifacts: CandidateSources): Candidate[] {
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

/** How many molecules a cofolding request names at most; each one is a Boltz run. */
export const MAX_COFOLD = 2;

/** A request to cofold the first MAX_COFOLD of the given molecules. */
export function cofoldPrompt(candidates: Pick<Candidate, "smiles" | "name">[]): string {
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
export function admetPrompt(candidates: Pick<Candidate, "smiles">[]): string {
  return `Predict ADMET properties for these molecules:\n${candidates.map((c) => `- ${c.smiles}`).join("\n")}`;
}
