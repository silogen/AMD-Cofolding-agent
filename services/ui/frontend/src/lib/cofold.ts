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
  /** Epoch seconds on the compute node; absent from older gateways. */
  startedAt?: number | null;
  /** How long each running molecule has been on its worker; absent from older Boltz services. */
  runningSeconds?: number[] | null;
}

export type CofoldStatus =
  | { available: false }
  | {
      available: true;
      status: string;
      mode: string;
      /** Typical wall time of one molecule lately; absent before the first, and from older services. */
      moleculeSeconds?: number | null;
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

/** How far the browser's clock and the compute node's may disagree when a call is paired with a request. */
const SKEW_MS = 30_000;

/**
 * The requests of this conversation. Boltz does not say who sent a request,
 * so a call in flight claims the first request with no more molecules that
 * started after it, and a run of the conversation that scores with Boltz
 * (*runScoring*) claims the rest.
 */
export function ownRequests(requests: readonly CofoldRequest[], calls: readonly CofoldCall[], runScoring: boolean): CofoldRequest[] {
  if (runScoring) return [...requests];
  const claimed = new Set<string>();
  for (const call of [...calls].sort((a, b) => a.at - b.at)) {
    const match = requests
      .filter((r) => !claimed.has(r.id) && (call.molecules === 0 || r.molecules <= call.molecules) && (r.startedAt == null || r.startedAt * 1000 >= call.at - SKEW_MS))
      .sort((a, b) => (a.startedAt ?? 0) - (b.startedAt ?? 0))[0];
    if (match) claimed.add(match.id);
  }
  return requests.filter((r) => claimed.has(r.id));
}

/** *status* as far as this conversation is concerned: its own requests, active only for those. */
export function ownStatus(status: CofoldStatus, calls: readonly CofoldCall[], runScoring: boolean): CofoldStatus {
  if (!status.available) return status;
  const requests = ownRequests(status.requests, calls, runScoring);
  // A service without request tracking only says that it is busy.
  const untracked = status.requests.length === 0 && status.active && (runScoring || calls.length > 0);
  return { ...status, requests, active: requests.length > 0 || untracked };
}

/**
 * What earlier answers taught: how long a molecule takes, and when each
 * request last finished one, for services that do not report the time of
 * each running molecule.
 */
export interface CofoldMemory {
  typicalSeconds: number | null;
  /** Request id → molecules done, and the request's elapsed seconds when that count was reached. */
  requests: Record<string, { done: number; at: number }>;
}

export const EMPTY_MEMORY: CofoldMemory = { typicalSeconds: null, requests: {} };

export function learn(memory: CofoldMemory, status: CofoldStatus): CofoldMemory {
  if (!status.available) return memory;
  let typical = status.moleculeSeconds ?? memory.typicalSeconds;
  const requests: CofoldMemory["requests"] = {};
  for (const r of status.requests) {
    if (r.elapsedSeconds === null) continue;
    const seen = memory.requests[r.id];
    if (!seen) {
      requests[r.id] = { done: r.done, at: r.done === 0 ? 0 : r.elapsedSeconds };
    } else if (r.done > seen.done) {
      const sample = r.elapsedSeconds - seen.at;
      if (status.moleculeSeconds == null && sample > 0) typical = typical === null ? sample : (typical + sample) / 2;
      requests[r.id] = { done: r.done, at: r.elapsedSeconds };
    } else requests[r.id] = seen;
  }
  return { typicalSeconds: typical, requests };
}

/** A running molecule counts for at most this much of itself until it is done. */
const MAX_PARTIAL = 0.9;

/** Seconds each running molecule of *r* has run, from the service or else estimated from *memory*. */
function runningSeconds(r: CofoldRequest, memory: CofoldMemory): number[] {
  if (r.runningSeconds) return r.runningSeconds;
  const since = memory.requests[r.id]?.at;
  const elapsed = r.elapsedSeconds;
  if (elapsed === null || since === undefined) return [];
  return Array.from({ length: r.running }, () => Math.max(0, elapsed - since));
}

export interface CofoldView {
  label: string;
  /** "2 molecules", "4 molecules on 3 GPUs". */
  detail: string;
  done: number;
  total: number;
  /** 0..1: molecules done, and those running for how far they likely got. */
  fraction: number;
  /** "1 queued", "Boltz is loading"; empty when there is nothing to add. */
  note: string;
  elapsedMs: number | null;
  /** Null without a molecule time to go by. */
  remainingMs: number | null;
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
 * progress. *finished* shows the answer as a batch that completed, while a
 * run prepares its next one.
 */
export function cofoldView(
  status: CofoldStatus | null,
  readAt: number,
  calls: readonly CofoldCall[],
  now: number,
  { memory = EMPTY_MEMORY, finished = false }: { memory?: CofoldMemory; finished?: boolean } = {},
): CofoldView | null {
  const since = finished ? 0 : Math.max(0, now - readAt) / 1000;
  if (status?.available && status.active) {
    const requests = status.requests;
    const total = requests.reduce((sum, r) => sum + r.molecules, 0);
    const done = finished ? total : requests.reduce((sum, r) => sum + Math.min(r.done, r.molecules), 0);
    const queued = finished ? 0 : requests.length ? requests.reduce((sum, r) => sum + r.queued, 0) : status.queueLength;
    const oldest = requests.reduce<number | null>((max, r) => (r.elapsedSeconds === null ? max : Math.max(max ?? 0, r.elapsedSeconds)), null);
    const preparing = requests.length > 0 && requests.every((r) => r.state === "preparing");
    const loading = status.workers.busy === 0 && status.workers.loading > 0;
    const gpus = status.gpus.busy;
    const typical = status.moleculeSeconds ?? memory.typicalSeconds;

    let partial = 0;
    let remaining: number | null = typical ? 0 : null;
    if (!finished && typical) {
      for (const r of requests) {
        const running = runningSeconds(r, memory).map((s) => s + since);
        partial += running.reduce((sum, s) => sum + Math.min(MAX_PARTIAL, s / typical), 0);
        const wave = running.reduce((max, s) => Math.max(max, typical - s), 0);
        const waves = r.queued > 0 ? Math.ceil(r.queued / Math.max(1, r.running)) : 0;
        remaining = Math.max(remaining ?? 0, Math.max(0, wave) + waves * typical);
      }
    }

    let detail: string;
    if (requests.length === 0) detail = plural(gpus, "GPU") + " busy";
    else if (requests.length === 1 || gpus < 2) detail = plural(total, "molecule");
    else detail = `${plural(total, "molecule")} on ${gpus} GPUs`;
    return {
      label: preparing ? "Preparing" : "Cofolding",
      detail,
      done,
      total,
      fraction: total > 0 ? Math.min(1, (done + partial) / total) : 0,
      note: finished ? "" : loading ? "Boltz is loading" : queued > 0 ? `${queued} queued` : "",
      elapsedMs: oldest === null ? null : (oldest + since) * 1000,
      remainingMs: finished || remaining === null || requests.length === 0 ? null : remaining * 1000,
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
    fraction: 0,
    note: "",
    elapsedMs: now - started,
    remainingMs: null,
    title: "Scoring with Boltz-2",
  };
}
