// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { StructureMetrics, StructurePanel } from "../types/api";
import { fetchJobStructures } from "./api";
import type { ScoreComponent } from "./moleculeFacts";
import type { MoleculeContext } from "./moleculePage";
import { structurePanel, type RunRow, type RunTable } from "./runResults";
import { addUserStructure } from "./userStructures";
import { sendToViewer } from "./viewerBus";

/**
 * The complex Boltz predicted for a molecule while it scored a REINVENT run.
 * The Results tab reads it from the run's CSV; the Jobs panel, which only
 * shows the top few, asks the gateway (GET /api/jobs/{id}/structures).
 */
export interface JobStructure {
  smiles: string;
  step: number | null;
  score: number | null;
  /** Relative to the shared volume. */
  path: string;
  metrics: StructureMetrics;
}

/** Results CSV metadata keys ("<key> (<component>)") by the name the viewer uses. */
const METRIC_KEYS: Record<string, keyof StructureMetrics> = {
  confidence_score: "confidence",
  ligand_iptm: "ligandIptm",
  iptm: "iptm",
  complex_plddt: "plddt",
  affinity_pred_value: "affinity",
  affinity_pred_value1: "affinity1",
  affinity_pred_value2: "affinity2",
  affinity_probability_binary: "binderProbability",
  n_checks_passed: "posebustersPassed",
  n_checks_total: "posebustersTotal",
};
const SUFFIXED = /^(.+?)\s*\((.+)\)$/;

/** The Boltz and PoseBusters values of a results row, or undefined when it has none. */
export function rowMetrics(table: RunTable, row: RunRow): StructureMetrics | undefined {
  const metrics: Record<string, number> = {};
  table.columns.forEach((column, i) => {
    const name = METRIC_KEYS[SUFFIXED.exec(column.key)?.[1] ?? column.key];
    const value = row.cells[i];
    if (name && typeof value === "number" && !(name in metrics)) metrics[name] = value;
  });
  return Object.keys(metrics).length ? (metrics as StructureMetrics) : undefined;
}

/** A results row as what the molecule page shows: run, step, score, components and complex. */
export function rowMolecule(table: RunTable, row: RunRow, run: string, ttp?: { ttpId?: string | null; ttpVersion?: number | null }): MoleculeContext {
  const number = (i: number) => (i >= 0 && typeof row.cells[i] === "number" ? (row.cells[i] as number) : null);
  const cofold = new Set(
    table.columns.flatMap((c) => (c.component && (c.role === "structure" || METRIC_KEYS[SUFFIXED.exec(c.key)?.[1] ?? ""]) ? [c.component] : [])),
  );
  const components: ScoreComponent[] = table.components.map((name) => ({
    name,
    value: number(table.columns.findIndex((c) => c.role === "component" && c.key === name)),
    raw: number(table.columns.findIndex((c) => c.role === "raw" && c.component === name)),
    ...(cofold.has(name) ? { cofold: true } : {}),
  }));
  const metrics = rowMetrics(table, row);
  return {
    run,
    step: row.step,
    score: row.total,
    components,
    structure: row.structure ? { path: row.structure, ...(metrics ? { metrics } : {}) } : null,
    ...(ttp?.ttpId ? { ttpId: ttp.ttpId, ttpVersion: ttp.ttpVersion ?? null } : {}),
  };
}

/** A run molecule's complex as a viewer panel, titled by run and step; null when the gateway cannot serve it. */
export function runStructurePanel(path: string, smiles: string | null, runId: string, step: number | null, metrics?: StructureMetrics): StructurePanel | null {
  const panel = structurePanel(path, smiles, `REINVENT ${runId}`);
  if (!panel) return null;
  return { ...panel, label: step === null ? `Run ${runId}` : `Run ${runId} · step ${step}`, ...(metrics && Object.keys(metrics).length ? { metrics } : {}) };
}

export function bySmiles(list: readonly JobStructure[]): Map<string, JobStructure> {
  return new Map(list.map((entry) => [entry.smiles, entry]));
}

const cache = new Map<string, Promise<Map<string, JobStructure>>>();

/** A run's structures by SMILES; read once per run, again only when `refresh` (the run is still going). */
export function loadJobStructures(jobId: string, refresh = false): Promise<Map<string, JobStructure>> {
  const cached = cache.get(jobId);
  if (cached && !refresh) return cached;
  const loading = fetchJobStructures(jobId).then(bySmiles);
  cache.set(jobId, loading);
  loading.catch(() => {
    if (cache.get(jobId) === loading) cache.delete(jobId);
  });
  return loading;
}

/** Add the complex to the conversation's structures and show it; `known` when the conversation already has it. */
export function openRunStructure(session: string, panel: StructurePanel, known = false): void {
  if (!known) addUserStructure(session, panel);
  // Let the app pick up the added structure before the viewer asks for it.
  window.setTimeout(() => sendToViewer({ path: panel.path, action: { kind: "ligand" }, source: "results" }), known ? 0 : 50);
}
