// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { PropertiesPanel, StructureMetrics } from "../types/api";
import { jobFileUrl } from "./api";
import type { Sourced } from "./artifacts";
import type { Candidate } from "./candidates";
import { rowMolecule, type JobStructure } from "./jobStructures";
import {
  componentFacts,
  isMetricName,
  metricsFromValues,
  predictionGroups,
  propertyFacts,
  propertyKey,
  runPredictions,
  type Fact,
  type FactGroup,
  type ScoreComponent,
} from "./moleculeFacts";
import type { Descriptors } from "./rdkit";
import { parseRunResults, type RunRow, type RunTable } from "./runResults";

/**
 * Opens a molecule's page in the workbench from anywhere: a Results row, a
 * top candidate in Jobs, a shortlist entry. What the caller knows about the
 * molecule comes along; the page adds what the conversation knows.
 */
export interface MoleculeContext {
  name?: string | null;
  /** The REINVENT run that generated it. */
  run?: string | null;
  step?: number | null;
  score?: number | null;
  components?: ScoreComponent[];
  /** Its predicted complex, relative to the shared volume. */
  structure?: { path: string; metrics?: StructureMetrics } | null;
  /** The TTP the run scored it with, to explain the score. */
  ttpId?: string | null;
  ttpVersion?: number | null;
}

export interface MoleculeRequest {
  smiles: string;
  context?: MoleculeContext;
}

type Listener = (request: MoleculeRequest) => void;
const listeners = new Set<Listener>();

export function openMolecule(smiles: string, context?: MoleculeContext): void {
  listeners.forEach((listener) => listener({ smiles, ...(context ? { context } : {}) }));
}

/** For the workbench, which shows the page. */
export function onMoleculeRequest(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/** "Run e0dceaaa · step 5", "Run e0dceaaa", or null for a molecule from the chat. */
export function originLabel(context: Pick<MoleculeContext, "run" | "step"> | undefined): string | null {
  if (!context?.run) return null;
  return context.step === null || context.step === undefined ? `Run ${context.run}` : `Run ${context.run} · step ${context.step}`;
}

export interface MoleculeView {
  name: string | null;
  origin: string;
  /** Every tool that reported it, for the origin's tooltip. */
  tools: string[];
  score: number | null;
  components: Fact[];
  cofold: StructureMetrics | null;
  structurePath: string | null;
  properties: Fact[];
  predictions: FactGroup[];
}

/**
 * The page's content: what the caller passed (a run row), else what the
 * conversation knows (`candidate`, merged from its tables, molecules and
 * structures), else the run's structures read from the gateway (`fetched`).
 */
export function moleculeView(input: {
  smiles: string;
  context?: MoleculeContext;
  candidate?: Candidate;
  fetched?: JobStructure | null;
  descriptors?: Descriptors | null;
  predictions?: readonly Sourced<PropertiesPanel>[];
  /** Canonical SMILES, to find the molecule in predictions that wrote it differently. */
  key?: (smiles: string) => string;
}): MoleculeView {
  const { smiles, context, candidate, fetched } = input;
  const values = candidate?.components ?? {};
  const structure = context?.structure ?? (fetched ? { path: fetched.path, metrics: fetched.metrics } : null);
  const step = context?.step ?? fetched?.step ?? null;
  const fromRun = context?.components;
  const other: ScoreComponent[] = Object.entries(values)
    .filter(([name]) => !propertyKey(name) && !isMetricName(name))
    .map(([name, value]) => ({ name, value }));
  const raws = Object.fromEntries((fromRun ?? []).map((c) => [c.name, c.raw]));
  const services = predictionGroups(smiles, input.predictions ?? [], input.key);
  const scored = runPredictions(fromRun ?? [], services);
  return {
    name: context?.name ?? candidate?.name ?? null,
    origin: originLabel(context?.run ? { run: context.run, step } : undefined) ?? "Chat",
    tools: candidate?.origins ?? [],
    score: context?.score ?? fetched?.score ?? candidate?.score ?? null,
    components: componentFacts(fromRun ?? other),
    cofold: structure?.metrics ?? candidate?.metrics ?? metricsFromValues(values) ?? null,
    structurePath: structure?.path ?? candidate?.structurePath ?? null,
    properties: propertyFacts({ ...raws, ...values }, input.descriptors),
    predictions: services.concat(scored.length ? [{ title: "Run", facts: scored }] : []),
  };
}

/** The run row of `smiles` (its best scoring one) as page context; `key` (canonical SMILES) when the CSV wrote it differently. */
export function runRowContext(table: RunTable, smiles: string, run: string, key?: (smiles: string) => string): MoleculeContext | null {
  let rows = table.rows.filter((row) => row.smiles === smiles);
  if (!rows.length && key) {
    const wanted = key(smiles);
    rows = table.rows.filter((row) => row.smiles && key(row.smiles) === wanted);
  }
  const rank = (row: RunRow) => (row.failure ? -Infinity : (row.total ?? -Infinity));
  const best = rows.reduce<RunRow | null>((top, row) => (!top || rank(row) > rank(top) ? row : top), null);
  return best ? rowMolecule(table, best, run) : null;
}

/** A run's results CSV (GET /api/jobs/{id}/files/results_1.csv), or null when it has none yet. */
export async function loadRunTable(run: string): Promise<RunTable | null> {
  const response = await fetch(jobFileUrl(run, "results_1.csv"));
  return response.ok ? parseRunResults(await response.text()) : null;
}
