// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { parseResult } from "./ttp";

/**
 * The REINVENT run a new TTP can start right away (run_optimization in
 * reinvent4_mcp/server.py): the prior, how many RL steps and how many
 * molecules per step. The last settings used are kept in localStorage.
 */

export type Prior = "reinvent" | "mol2mol";

export interface RunSettings {
  maxSteps: number;
  batchSize: number;
  prior: Prior;
  /** The Mol2Mol prior to use when `prior` is "mol2mol". */
  model: string;
  /** Seed SMILES for Mol2Mol, one per line. */
  seeds: string;
}

/** run_optimization's defaults and the server config's batch_size (reinvent4_mcp_config.yaml). */
export const DEFAULT_RUN: RunSettings = { maxSteps: 300, batchSize: 16, prior: "reinvent", model: ".m2m_medium", seeds: "" };

export const MAX_STEPS = 5000;
export const MAX_BATCH = 512;

type KeyValueStore = Pick<Storage, "getItem" | "setItem">;

const browserStorage = (): KeyValueStore | null => {
  try {
    return typeof window !== "undefined" ? window.localStorage : null;
  } catch {
    return null;
  }
};

export const RUN_KEY = "cofold.ttp-run";

const whole = (value: unknown, fallback: number, max: number) =>
  typeof value === "number" && Number.isInteger(value) && value >= 1 && value <= max ? value : fallback;

export function loadRunSettings(storage: KeyValueStore | null = browserStorage()): RunSettings {
  try {
    const value = JSON.parse(storage?.getItem(RUN_KEY) ?? "null") as Partial<RunSettings> | null;
    if (!value || typeof value !== "object") return DEFAULT_RUN;
    return {
      maxSteps: whole(value.maxSteps, DEFAULT_RUN.maxSteps, MAX_STEPS),
      batchSize: whole(value.batchSize, DEFAULT_RUN.batchSize, MAX_BATCH),
      prior: value.prior === "mol2mol" ? "mol2mol" : "reinvent",
      model: typeof value.model === "string" && value.model.startsWith(".m2m") ? value.model : DEFAULT_RUN.model,
      seeds: typeof value.seeds === "string" ? value.seeds : "",
    };
  } catch {
    return DEFAULT_RUN;
  }
}

export function saveRunSettings(settings: RunSettings, storage: KeyValueStore | null = browserStorage()): void {
  try {
    storage?.setItem(RUN_KEY, JSON.stringify(settings));
  } catch {
    // Storage full or disabled: the defaults come back next time.
  }
}

/** Seed SMILES from free text: one per line, or separated by spaces or commas. */
export const seedList = (text: string) => text.split(/[\s,]+/).filter(Boolean);

/** Why run_optimization would refuse these settings, or would not do what was meant. */
export function runProblems(settings: RunSettings): string[] {
  const problems: string[] = [];
  if (!Number.isInteger(settings.maxSteps) || settings.maxSteps < 1 || settings.maxSteps > MAX_STEPS) problems.push(`Steps must be a whole number from 1 to ${MAX_STEPS}.`);
  if (!Number.isInteger(settings.batchSize) || settings.batchSize < 1 || settings.batchSize > MAX_BATCH) problems.push(`Batch size must be a whole number from 1 to ${MAX_BATCH}.`);
  if (settings.prior === "mol2mol" && seedList(settings.seeds).length === 0) problems.push("Mol2Mol needs at least one seed SMILES.");
  return problems;
}

export const runModel = (settings: RunSettings) => (settings.prior === "mol2mol" ? settings.model : ".reinvent");

/** The run_optimization arguments that start a run on a stored TTP. */
export function runArguments(ttpId: string, settings: RunSettings): Record<string, unknown> {
  const args: Record<string, unknown> = { ttp_id: ttpId, model: runModel(settings), max_steps: settings.maxSteps, batch_size: settings.batchSize };
  if (settings.prior === "mol2mol") args.seed_smiles = seedList(settings.seeds);
  return args;
}

/** Molecules a run generates and scores, and how many of them Boltz-2 cofolds. */
export function runCost(settings: RunSettings, cofold: boolean): { molecules: number; cofolds: number } {
  const molecules = settings.maxSteps * settings.batchSize;
  return { molecules, cofolds: cofold ? molecules : 0 };
}

export interface PriorModel {
  key: string;
  label: string;
  available: boolean;
}

/** "Reinvent (de novo)", "Mol2Mol medium similarity", ... */
export function modelLabel(key: string | null | undefined): string {
  if (!key) return "—";
  if (key === ".reinvent") return "Reinvent (de novo)";
  const m2m = /^\.m2m_(.+)$/.exec(key);
  if (m2m) return `Mol2Mol ${m2m[1].replace(/_/g, " ")}${["high", "medium"].includes(m2m[1]) ? " similarity" : ""}`;
  return key.replace(/^\./, "").split(/[\\/]/).pop() ?? key;
}

/** The Mol2Mol priors list_available_models reports, the ones on disk first. */
export function parseMol2MolModels(value: unknown): PriorModel[] {
  const parsed = parseResult(value);
  if (!Array.isArray(parsed)) return [];
  return parsed
    .filter((m): m is Record<string, unknown> => typeof m === "object" && m !== null && typeof (m as Record<string, unknown>).key === "string")
    .filter((m) => String(m.key).startsWith(".m2m") || /mol2mol/i.test(String(m.model_type ?? "")))
    .map((m) => ({ key: String(m.key), label: modelLabel(String(m.key)), available: m.exists !== false }))
    .sort((a, b) => Number(b.available) - Number(a.available));
}

/** The job a run_optimization call started, or its error. */
export function startedRun(value: unknown): { jobId: string } | { error: string } {
  const parsed = parseResult(value);
  if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
    const record = parsed as Record<string, unknown>;
    if (typeof record.error === "string") return { error: record.error };
    if (typeof record.job_id === "string" && record.job_id) return { jobId: record.job_id };
  }
  return { error: typeof value === "string" && value.trim() ? value.trim() : "run_optimization gave no job id." };
}
