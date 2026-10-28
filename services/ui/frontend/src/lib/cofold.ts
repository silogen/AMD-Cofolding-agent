// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { TurnBlock } from "../types/api";

type ToolStep = Extract<TurnBlock, { kind: "tool" }>;

/** One /score request Boltz is working on (GET /api/services/cofold). */
export interface CofoldRequest {
  id: string;
  molecules: number;
  done: number;
  queued: number;
  running: number;
  state: "preparing" | "queued" | "running";
  elapsedSeconds: number | null;
}

export type CofoldStatus =
  | { available: false }
  | {
      available: true;
      status: string;
      mode: string;
      workers: { total: number; busy: number; loading: number; failed: number };
      gpus: { total: number; busy: number };
      queueLength: number;
      requests: CofoldRequest[];
      active: boolean;
    };

/** A chat tool call that is (or may be) cofolding right now. */
export interface CofoldCall {
  id: string;
  molecules: number;
  /** Epoch ms when the call started. */
  at: number;
  /** False when its TTP is only named and this conversation never showed it. */
  certain: boolean;
}

/** Fast while a cofold is believed to run; a slow probe for runs that may score with Boltz. */
export const COFOLD_POLL_MS = 5_000;
export const COFOLD_PROBE_MS = 60_000;

const TTP_TOOLS = new Set(["create_ttp", "update_ttp", "get_ttp"]);

const mentionsBoltz = (value: unknown) => {
  if (value === undefined || value === null) return false;
  return (typeof value === "string" ? value : JSON.stringify(value)).toLowerCase().includes("boltz");
};

/**
 * Whether the TTP *ttpId* scores with Boltz, from the newest step of this
 * conversation that showed it; null when none did.
 */
export function ttpUsesBoltz(steps: readonly ToolStep[], ttpId: string | null): boolean | null {
  if (!ttpId) return null;
  const newestFirst = [...steps].sort((a, b) => b.at - a.at);
  for (const step of newestFirst) {
    if (!TTP_TOOLS.has(step.name) || !step.result || step.result.isError) continue;
    const text = step.result.result;
    if (!text.includes(ttpId) && step.arguments.ttp_id !== ttpId) continue;
    const withoutId = (value: string) => value.split(ttpId).join("");
    return mentionsBoltz(withoutId(text)) || mentionsBoltz(withoutId(JSON.stringify(step.arguments)));
  }
  return null;
}

/** The cofold a score_molecules call starts, or null if it does not cofold. */
export function cofoldCall(step: ToolStep, steps: readonly ToolStep[]): CofoldCall | null {
  if (step.name !== "score_molecules") return null;
  const args = step.arguments;
  const molecules = Array.isArray(args.smiles) ? args.smiles.length : 0;
  let certain: boolean;
  if (args.ttp && typeof args.ttp === "object") certain = mentionsBoltz(args.ttp);
  else if (typeof args.ttp_id === "string" && args.ttp_id) {
    const known = ttpUsesBoltz(steps, args.ttp_id);
    if (known === false) return null;
    certain = known === true;
  } else certain = mentionsBoltz(args.components);
  if (!certain && !(typeof args.ttp_id === "string" && args.ttp_id)) return null;
  return { id: step.id, molecules, at: step.at, certain };
}

/** The cofolding calls of the turn that started at *turnAt* and has not ended. */
export function inFlightCofold(steps: readonly ToolStep[], turnAt: number | null): CofoldCall[] {
  if (turnAt === null) return [];
  const calls: CofoldCall[] = [];
  for (const step of steps) {
    if (step.result || step.at < turnAt) continue;
    const call = cofoldCall(step, steps);
    if (call) calls.push(call);
  }
  return calls;
}

/** Whether a REINVENT run scores with Boltz: yes, no, or unknown (null). */
export function runUsesBoltz(steps: readonly ToolStep[], ttpId: string | null, components: readonly { name: string }[] | undefined): boolean | null {
  const known = ttpUsesBoltz(steps, ttpId);
  if (known !== null) return known;
  if (components?.length) return components.some((c) => /boltz|cofold|affinity/i.test(c.name));
  return null;
}

/**
 * How long until the next poll of /api/services/cofold, or null for none:
 * the service sits behind a slow tunnel, so it is only asked while something
 * says a cofold runs and the page is visible.
 */
export function cofoldPollDelay(input: { visible: boolean; calls: readonly CofoldCall[]; boltzRun: boolean | null; running: boolean; lastActive: boolean }): number | null {
  if (!input.visible) return null;
  if (input.calls.length > 0 || input.lastActive || (input.running && input.boltzRun === true)) return COFOLD_POLL_MS;
  if (input.running && input.boltzRun === null) return COFOLD_PROBE_MS;
  return null;
}

/** "0:41", "12:05", "1:02:05". */
export function clock(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = String(total % 60).padStart(2, "0");
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${s}` : `${m}:${s}`;
}

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

export interface CofoldView {
  label: string;
  /** "2 molecules", "4 molecules on 3 GPUs". */
  detail: string;
  done: number;
  total: number;
  /** "1 queued", "Boltz is loading"; empty when there is nothing to add. */
  note: string;
  elapsedMs: number | null;
  /** One line per request, for a tooltip. */
  title: string;
}

/** Whether the cofold strip shows at all; cofoldView without the clock. */
export const cofoldShown = (status: CofoldStatus | null, calls: readonly CofoldCall[]) =>
  (status?.available === true && status.active) || calls.some((call) => call.certain);

/**
 * What the cofold strip says, from the last answer of the service (read at
 * *readAt*) and the cofolding calls in flight; null when nothing cofolds.
 * Calls show at once, before any answer, and the service then fills in the
 * progress.
 */
export function cofoldView(status: CofoldStatus | null, readAt: number, calls: readonly CofoldCall[], now: number): CofoldView | null {
  const since = Math.max(0, now - readAt);
  if (status?.available && status.active) {
    const requests = status.requests;
    const total = requests.reduce((sum, r) => sum + r.molecules, 0);
    const done = requests.reduce((sum, r) => sum + Math.min(r.done, r.molecules), 0);
    const queued = requests.length ? requests.reduce((sum, r) => sum + r.queued, 0) : status.queueLength;
    const oldest = requests.reduce<number | null>((max, r) => (r.elapsedSeconds === null ? max : Math.max(max ?? 0, r.elapsedSeconds)), null);
    const preparing = requests.length > 0 && requests.every((r) => r.state === "preparing");
    const loading = status.workers.busy === 0 && status.workers.loading > 0;
    const gpus = status.gpus.busy;
    let detail: string;
    if (requests.length === 0) detail = plural(gpus, "GPU") + " busy";
    else if (requests.length === 1 || gpus < 2) detail = plural(total, "molecule");
    else detail = `${plural(total, "molecule")} on ${gpus} GPUs`;
    return {
      label: preparing ? "Preparing cofold" : "Cofolding",
      detail,
      done,
      total,
      note: loading ? "Boltz is loading" : queued > 0 ? `${queued} queued` : "",
      elapsedMs: oldest === null ? null : oldest * 1000 + since,
      title: requests.length
        ? requests.map((r) => `${r.state}: ${r.done}/${r.molecules} done${r.queued ? `, ${r.queued} queued` : ""}`).join("\n")
        : `${status.workers.busy} of ${status.workers.total} Boltz workers busy`,
    };
  }
  const sure = calls.filter((call) => call.certain);
  if (sure.length === 0) return null;
  const total = sure.reduce((sum, call) => sum + call.molecules, 0);
  const started = Math.min(...sure.map((call) => call.at));
  return {
    label: "Cofolding",
    detail: plural(total, "molecule"),
    done: 0,
    total,
    note: "",
    elapsedMs: now - started,
    title: "Scoring with Boltz-2",
  };
}
