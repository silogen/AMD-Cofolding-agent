// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { Contact, Panel, TurnBlock } from "../types/api";

/**
 * Target Product Profiles and the protein target, as the MCP server defines
 * them (reinvent4_mcp/ttp/schema.py) and as the conversation reveals them.
 *
 * The server stores TTPs (list_ttps, get_ttp, update_ttp); a session also
 * carries them inline in create_ttp, run_optimization and score_molecules
 * calls. The target card is read from the same events: UniProt lookups, TTP
 * target blocks and interaction contacts.
 */

// ---------------------------------------------------------------------------
// events
// ---------------------------------------------------------------------------

/** The part of a tool call or result these parsers read. */
export type ToolEvent =
  | { type: "tool_call"; name: string; arguments: Record<string, unknown> }
  | { type: "tool_result"; name: string; result: unknown; panels?: Panel[]; isError?: boolean };

type ToolStep = Extract<TurnBlock, { kind: "tool" }>;

/** Tool steps (in any order, e.g. `artifacts.steps`) as call/result events, oldest first. */
export function toolEvents(steps: ToolStep[]): ToolEvent[] {
  const events: ToolEvent[] = [];
  for (const step of [...steps].sort((a, b) => a.at - b.at)) {
    events.push({ type: "tool_call", name: step.name, arguments: step.arguments ?? {} });
    if (step.result) {
      events.push({
        type: "tool_result",
        name: step.name,
        result: step.result.result,
        panels: step.result.panels,
        isError: step.result.isError,
      });
    }
  }
  return events;
}

/** A tool result as data: results arrive as JSON text. */
export function parseResult(result: unknown): unknown {
  if (typeof result !== "string") return result;
  try {
    return JSON.parse(result);
  } catch {
    return null;
  }
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const str = (value: unknown): string | null => (typeof value === "string" && value.trim() ? value.trim() : null);
const num = (value: unknown): number | null => (typeof value === "number" && Number.isFinite(value) ? value : null);

// ---------------------------------------------------------------------------
// TTP schema
// ---------------------------------------------------------------------------

export const TRANSFORM_TYPES = [
  "sigmoid",
  "reverse_sigmoid",
  "double_sigmoid",
  "step",
  "right_step",
  "left_step",
  "exponential_decay",
  "value_mapping",
] as const;

export interface Transform {
  type: string;
  low?: number;
  high?: number;
  k?: number;
  coef_div?: number;
  coef_si?: number;
  coef_se?: number;
  [key: string]: unknown;
}

/** A scoring objective or a pre-scoring component; both are weighted, transformed scores. */
export interface Objective {
  name: string;
  type: "native" | "external";
  component?: string | null;
  service?: string | null;
  service_url?: string | null;
  service_params?: Record<string, unknown> | null;
  depends_on?: string | null;
  weight: number;
  transform?: Transform | null;
  params?: Record<string, unknown> | null;
  comment?: string | null;
}

export interface TtpTarget {
  protein_sequence: string;
  pdb_id?: string | null;
  msa_path?: string | null;
  protein_pdb_path?: string | null;
}

export interface Constraint {
  type: string;
  smarts?: string[] | null;
  [key: string]: unknown;
}

export interface Ttp {
  id: string;
  name: string;
  version: number;
  target?: TtpTarget | null;
  pre_scoring?: { comment?: string | null; components: Objective[] } | null;
  scoring: { comment?: string | null; objectives: Objective[]; aggregation?: "geometric_mean" | "arithmetic_mean" };
  constraints?: Constraint[] | null;
}

export interface TtpSummary {
  id: string;
  name: string;
  version: number;
}

/** A TTP from a tool payload, or null when the value is not one. */
export function asTtp(value: unknown): Ttp | null {
  if (!isRecord(value) || typeof value.id !== "string" || !isRecord(value.scoring)) return null;
  if (!Array.isArray(value.scoring.objectives)) return null;
  return {
    ...(value as unknown as Ttp),
    name: typeof value.name === "string" ? value.name : value.id,
    version: num(value.version) ?? 1,
  };
}

/** list_ttps result. */
export function asTtpList(value: unknown): TtpSummary[] {
  const parsed = parseResult(value);
  if (!Array.isArray(parsed)) return [];
  return parsed
    .filter((item): item is Record<string, unknown> => isRecord(item) && typeof item.id === "string")
    .map((item) => ({ id: item.id as string, name: str(item.name) ?? (item.id as string), version: num(item.version) ?? 1 }));
}

/** The error message of a tool payload such as `{"error": "TTP 'x' not found"}`. */
export function payloadError(value: unknown): string | null {
  const parsed = parseResult(value);
  return isRecord(parsed) && typeof parsed.error === "string" ? parsed.error : null;
}

/** The TTP a tool call or result carries, if any. */
function ttpIn(event: ToolEvent): Ttp | null {
  if (event.type === "tool_call") return asTtp(event.arguments.ttp);
  const payload = parseResult(event.result);
  if (event.name === "get_ttp") return asTtp(payload);
  if (isRecord(payload)) return asTtp(payload.ttp);
  return null;
}

/** The id of the TTP an event is about: the last one wins. */
function ttpIdIn(event: ToolEvent): string | null {
  if (event.type === "tool_call") return str(event.arguments.ttp_id) ?? asTtp(event.arguments.ttp)?.id ?? null;
  return ttpIn(event)?.id ?? null;
}

/** The TTP this session last worked with: its id, and the full definition when one was seen. */
export function latestTtp(events: ToolEvent[]): { id: string | null; ttp: Ttp | null } {
  let id: string | null = null;
  let ttp: Ttp | null = null;
  for (const event of events) {
    const seenId = ttpIdIn(event);
    if (seenId) id = seenId;
    const seen = ttpIn(event);
    if (seen) ttp = seen;
  }
  return { id, ttp: ttp && ttp.id === id ? ttp : id ? null : ttp };
}

// ---------------------------------------------------------------------------
// transforms
// ---------------------------------------------------------------------------

/** The editable numeric parameters of a transform type. */
export function transformParams(type: string): (keyof Transform & string)[] {
  switch (normaliseType(type)) {
    case "sigmoid":
    case "reverse_sigmoid":
      return ["low", "high", "k"];
    case "double_sigmoid":
      return ["low", "high", "coef_div", "coef_si", "coef_se"];
    case "step":
    case "right_step":
    case "left_step":
      return ["low", "high"];
    case "exponential_decay":
      return ["k"];
    default:
      return [];
  }
}

function normaliseType(type: string): string {
  const flat = type.toLowerCase().replace(/[-_]/g, "");
  return TRANSFORM_TYPES.find((t) => t.replace(/_/g, "") === flat) ?? type;
}

const DEFAULTS: Record<string, Partial<Transform>> = {
  sigmoid: { low: 0, high: 1, k: 0.5 },
  reverse_sigmoid: { low: 0, high: 1, k: 0.5 },
  double_sigmoid: { low: 0, high: 1, coef_div: 1, coef_si: 10, coef_se: 10 },
  step: { low: 0, high: 1 },
  right_step: { low: 0, high: 1 },
  left_step: { low: 0, high: 1 },
  exponential_decay: { k: 1 },
};

/** A transform of another type, keeping the parameters the two share. */
export function retype(transform: Transform | null | undefined, type: string): Transform {
  const next: Transform = { ...(DEFAULTS[type] ?? {}), type };
  for (const key of transformParams(type)) {
    const value = transform?.[key];
    if (typeof value === "number") next[key] = value;
  }
  if (type === "value_mapping" && transform?.mapping) next.mapping = transform.mapping;
  return next;
}

/** REINVENT4's score transforms; null when the type has no curve (value_mapping). */
export function evaluateTransform(t: Transform, x: number): number | null {
  const low = num(t.low) ?? 0;
  const high = num(t.high) ?? 1;
  const k = num(t.k) ?? 0.5;
  switch (normaliseType(t.type)) {
    case "sigmoid":
      return high === low ? (x >= low ? 1 : 0) : 1 / (1 + 10 ** ((10 * k * (x - (low + high) / 2)) / (low - high)));
    case "reverse_sigmoid":
      return high === low ? (x <= low ? 1 : 0) : 1 / (1 + 10 ** ((10 * k * (x - (low + high) / 2)) / (high - low)));
    case "double_sigmoid": {
      const div = num(t.coef_div) || 1;
      const si = num(t.coef_si) ?? 10;
      const se = num(t.coef_se) ?? 10;
      const a = 10 ** ((se * x) / div);
      const left = a / (a + 10 ** ((se * low) / div));
      const b = 10 ** ((si * x) / div);
      const right = b / (b + 10 ** ((si * high) / div));
      const y = left - right;
      return Number.isFinite(y) ? Math.min(1, Math.max(0, y)) : 0;
    }
    case "step":
      return x >= low && x <= high ? 1 : 0;
    case "right_step":
      return x >= low ? 1 : 0;
    case "left_step":
      return x <= low ? 1 : 0;
    case "exponential_decay":
      return x < 0 ? 1 : Math.exp(-k * x);
    default:
      return null;
  }
}

/** Sampled curve of a transform over a range wide enough to show its shape. */
export function transformCurve(t: Transform, samples = 48): { x: number; y: number }[] {
  if (evaluateTransform(t, 0) === null) return [];
  let from: number;
  let to: number;
  if (normaliseType(t.type) === "exponential_decay") {
    from = 0;
    to = 5 / Math.max(num(t.k) ?? 1, 1e-3);
  } else {
    const low = num(t.low) ?? 0;
    const high = num(t.high) ?? 1;
    const span = Math.abs(high - low) || Math.max(Math.abs(low), 1);
    from = Math.min(low, high) - span * 0.6;
    to = Math.max(low, high) + span * 0.6;
  }
  const points: { x: number; y: number }[] = [];
  for (let i = 0; i <= samples; i++) {
    const x = from + ((to - from) * i) / samples;
    points.push({ x, y: evaluateTransform(t, x) ?? 0 });
  }
  return points;
}

/** "200 – 500", "≥ 0.5", "≤ 3": the window an objective rewards. */
export function targetRange(t: Transform | null | undefined): string | null {
  if (!t) return null;
  const f = (v: unknown) => (typeof v === "number" ? String(+v.toPrecision(4)) : "?");
  switch (normaliseType(t.type)) {
    case "double_sigmoid":
    case "step":
      return `${f(t.low)} – ${f(t.high)}`;
    case "sigmoid":
      return `↑ ${f(t.low)} → ${f(t.high)}`;
    case "reverse_sigmoid":
      return `↓ ${f(t.low)} → ${f(t.high)}`;
    case "right_step":
      return `≥ ${f(t.low)}`;
    case "left_step":
      return `≤ ${f(t.low)}`;
    case "exponential_decay":
      return `decay k=${f(t.k)}`;
    default:
      return null;
  }
}

// ---------------------------------------------------------------------------
// editing
// ---------------------------------------------------------------------------

export type TtpSection = "scoring" | "pre_scoring";

export function objectivesOf(ttp: Ttp, section: TtpSection): Objective[] {
  return section === "scoring" ? ttp.scoring.objectives : (ttp.pre_scoring?.components ?? []);
}

/** A copy of the TTP with one objective changed. */
export function patchObjective(ttp: Ttp, section: TtpSection, index: number, patch: Partial<Objective>): Ttp {
  const list = objectivesOf(ttp, section).map((o, i) => (i === index ? { ...o, ...patch } : o));
  return withObjectives(ttp, section, list);
}

/** A copy of the TTP without one objective. */
export function removeObjective(ttp: Ttp, section: TtpSection, index: number): Ttp {
  return withObjectives(
    ttp,
    section,
    objectivesOf(ttp, section).filter((_, i) => i !== index),
  );
}

function withObjectives(ttp: Ttp, section: TtpSection, list: Objective[]): Ttp {
  if (section === "scoring") return { ...ttp, scoring: { ...ttp.scoring, objectives: list } };
  return { ...ttp, pre_scoring: { ...(ttp.pre_scoring ?? {}), components: list } };
}

/** Weights as shares of the total, for the bars. */
export function weightShares(ttp: Ttp): Map<string, number> {
  const all = [...objectivesOf(ttp, "pre_scoring"), ...objectivesOf(ttp, "scoring")];
  const total = all.reduce((sum, o) => sum + (o.weight > 0 ? o.weight : 0), 0) || 1;
  return new Map(all.map((o) => [o.name, Math.max(0, o.weight) / total]));
}

/**
 * The weight slider range: 0-1 in steps of 0.01 when every weight is at most
 * 1, otherwise wide enough for the largest weight.
 */
export function weightRange(ttp: Ttp): { min: number; max: number; step: number } {
  const all = [...objectivesOf(ttp, "pre_scoring"), ...objectivesOf(ttp, "scoring")];
  const largest = all.reduce((m, o) => Math.max(m, o.weight), 0);
  if (largest <= 1) return { min: 0, max: 1, step: 0.01 };
  return { min: 0, max: Math.max(2, Math.ceil(largest * 2)), step: 0.05 };
}

/** First-class services from KNOWN_EXTERNAL_SERVICES in reinvent4_mcp/ttp/schema.py. */
export const KNOWN_SERVICES = new Set(["boltz", "posebusters", "admet", "retrodmta", "interaction_profiler"]);

/** "native" REINVENT4 component, a configured "service", or a custom "external" REST endpoint. */
export function objectiveSource(o: Pick<Objective, "type" | "service">): "native" | "service" | "external" {
  if (o.type === "native") return "native";
  return o.service && KNOWN_SERVICES.has(o.service) ? "service" : "external";
}

/** When the TTP was created, if the payload says so. */
export function ttpCreated(ttp: Ttp): string | null {
  const record = ttp as unknown as Record<string, unknown>;
  for (const key of ["created", "created_at"]) {
    const value = record[key];
    if (typeof value === "string" && value.trim()) return value;
  }
  return null;
}

/** Generation settings (mode, model, scaffold, seeds), if the payload has any. */
export function ttpGeneration(ttp: Ttp): { label: string; value: string }[] {
  const generation = (ttp as unknown as Record<string, unknown>).generation;
  if (!isRecord(generation)) return [];
  const libinvent = isRecord(generation.libinvent_input) ? generation.libinvent_input : {};
  const seeds = generation.seeds ?? generation.seed_smiles;
  const rows: [string, unknown][] = [
    ["Mode", generation.mode],
    ["Model", generation.model],
    ["Scaffold", generation.scaffold ?? libinvent.scaffold],
    ["Seeds", Array.isArray(seeds) ? seeds.join(", ") : seeds],
  ];
  return rows
    .filter(([, value]) => (typeof value === "string" && value.trim()) || typeof value === "number")
    .map(([label, value]) => ({ label, value: String(value) }));
}

const same = (a: unknown, b: unknown) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);

/** Names of the objectives that differ between two versions, and any that were removed. */
export function changes(base: Ttp, draft: Ttp): { changed: string[]; removed: string[]; aggregation: boolean } {
  const changed: string[] = [];
  const removed: string[] = [];
  for (const section of ["pre_scoring", "scoring"] as const) {
    const before = new Map(objectivesOf(base, section).map((o) => [o.name, o]));
    const after = new Map(objectivesOf(draft, section).map((o) => [o.name, o]));
    for (const [name, o] of after) if (!same(before.get(name), o)) changed.push(name);
    for (const name of before.keys()) if (!after.has(name)) removed.push(name);
  }
  return { changed, removed, aggregation: (base.scoring.aggregation ?? "geometric_mean") !== (draft.scoring.aggregation ?? "geometric_mean") };
}

export function isDirty(base: Ttp, draft: Ttp): boolean {
  const c = changes(base, draft);
  return c.changed.length > 0 || c.removed.length > 0 || c.aggregation;
}

/**
 * The `updates` argument of update_ttp for an edited draft. Lists are
 * replaced, not merged, by the server, so whole sections are sent.
 */
export function ttpUpdates(base: Ttp, draft: Ttp): Record<string, unknown> {
  const updates: Record<string, unknown> = {};
  if (!same(base.scoring, draft.scoring)) updates.scoring = draft.scoring;
  if (!same(base.pre_scoring, draft.pre_scoring)) updates.pre_scoring = draft.pre_scoring;
  return updates;
}

/** What to ask the agent so it adopts the edited profile. */
export function applyPrompt(base: Ttp, draft: Ttp): string {
  const { changed, removed, aggregation } = changes(base, draft);
  const lines = [
    `Update TTP "${base.id}" (currently version ${base.version}) with my edited objectives, then use the new version for the next scoring or optimization.`,
  ];
  const summary: string[] = [];
  if (changed.length) summary.push(`changed: ${changed.join(", ")}`);
  if (removed.length) summary.push(`removed: ${removed.join(", ")}`);
  if (aggregation) summary.push(`aggregation → ${draft.scoring.aggregation}`);
  if (summary.length) lines.push(`Edits — ${summary.join("; ")}.`);
  lines.push("Call update_ttp with these updates:", "```json", JSON.stringify(ttpUpdates(base, draft), null, 2), "```");
  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// local drafts
// ---------------------------------------------------------------------------

export interface TtpDraft {
  /** The stored TTP this edits, at the version it was based on. */
  base: Ttp;
  draft: Ttp;
  savedAt: number;
}

type KeyValueStore = Pick<Storage, "getItem" | "setItem" | "removeItem">;

const browserStorage = (): KeyValueStore | null => {
  try {
    return typeof window !== "undefined" ? window.localStorage : null;
  } catch {
    return null;
  }
};

export const draftKey = (sessionId: string) => `cofold.ttp-draft.${sessionId}`;

/** A draft of `draft` against `base`, or null when there is nothing to keep. */
export function makeDraft(base: Ttp, draft: Ttp, current: Ttp = base): TtpDraft | null {
  return isDirty(current, draft) ? { base, draft, savedAt: Date.now() } : null;
}

export function loadDraft(sessionId: string, storage: KeyValueStore | null = browserStorage()): TtpDraft | null {
  if (!storage) return null;
  try {
    const raw = storage.getItem(draftKey(sessionId));
    if (!raw) return null;
    const value = JSON.parse(raw) as Partial<TtpDraft>;
    const base = asTtp(value.base);
    const draft = asTtp(value.draft);
    return base && draft ? { base, draft, savedAt: num(value.savedAt) ?? 0 } : null;
  } catch {
    return null;
  }
}

export function saveDraft(sessionId: string, value: TtpDraft | null, storage: KeyValueStore | null = browserStorage()): void {
  if (!storage) return;
  try {
    if (value) storage.setItem(draftKey(sessionId), JSON.stringify(value));
    else storage.removeItem(draftKey(sessionId));
  } catch {
    // Storage full or disabled: the draft lives until the page reloads.
  }
}

// ---------------------------------------------------------------------------
// building a new TTP
// ---------------------------------------------------------------------------

/** A configured scoring service, from list_external_services. */
export interface ServiceInfo {
  name: string;
  url: string;
  status: string;
  /** ADMET endpoints. */
  endpoints: string[];
  /** RetroDMTA endpoints by target. */
  targets: Record<string, string[]>;
}

/** list_external_services, keyed by service name. */
export function parseServices(value: unknown): Record<string, ServiceInfo> {
  const parsed = parseResult(value);
  if (!isRecord(parsed)) return {};
  const services: Record<string, ServiceInfo> = {};
  for (const [name, entry] of Object.entries(parsed)) {
    if (!isRecord(entry) || typeof entry.url !== "string") continue;
    const strings = (v: unknown) => (Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : []);
    const targets: Record<string, string[]> = {};
    if (isRecord(entry.targets_detail)) {
      for (const [target, detail] of Object.entries(entry.targets_detail)) targets[target] = isRecord(detail) ? strings(detail.endpoints) : [];
    } else {
      for (const target of strings(entry.targets)) targets[target] = [];
    }
    services[name] = { name, url: entry.url, status: str(entry.status) ?? "unknown", endpoints: strings(entry.endpoints), targets };
  }
  return services;
}

/** Native REINVENT4 scorers a profile can use without extra settings, from list_scoring_components. */
export function parseComponents(value: unknown): string[] {
  const parsed = parseResult(value);
  if (!Array.isArray(parsed)) return [];
  return parsed
    .filter(isRecord)
    .filter((c) => c.component_type === "scorer" && typeof c.class === "string" && !String(c.class).startsWith("Fragment"))
    .filter((c) => !Array.isArray(c.parameters) || c.parameters.length === 0)
    .map((c) => c.class as string)
    .sort((a, b) => a.localeCompare(b));
}

/** "BRD4 Cofolding (v2)" -> "brd4-cofolding-v2", a valid TTP id. */
export function slugify(name: string): string {
  return name
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48)
    .replace(/-+$/, "");
}

export const TTP_ID = /^[a-z0-9][a-z0-9_-]*$/;
const AMINO_ACIDS = /^[ACDEFGHIKLMNPQRSTVWY]+$/;

/** A pasted sequence or FASTA record as bare residues. */
export function normaliseSequence(text: string): string {
  return text
    .split(/\r?\n/)
    .filter((line) => !line.trim().startsWith(">"))
    .join("")
    .replace(/[\s\d*]/g, "")
    .toUpperCase();
}

/** Why the server would reject a sequence (reinvent4_mcp/ttp/schema.py TargetConfig), or null. */
export function sequenceProblem(sequence: string): string | null {
  if (!sequence) return null;
  if (sequence.length < 10) return "The protein sequence needs at least 10 residues.";
  if (!AMINO_ACIDS.test(sequence)) {
    const bad = [...new Set(sequence.replace(/[ACDEFGHIKLMNPQRSTVWY]/g, ""))].sort().join(", ");
    return `The sequence has non-standard residues: ${bad}.`;
  }
  return null;
}

/** Transforms that suit each objective out of the box; none for scores already in 0-1. */
const NATIVE_TRANSFORMS: Record<string, Transform | null> = {
  QED: null,
  Csp3: null,
  MolecularWeight: { type: "double_sigmoid", low: 200, high: 500, coef_div: 500, coef_si: 20, coef_se: 20 },
  SlogP: { type: "double_sigmoid", low: 1, high: 5, coef_div: 5, coef_si: 10, coef_se: 10 },
  TPSA: { type: "double_sigmoid", low: 20, high: 130, coef_div: 130, coef_si: 20, coef_se: 20 },
  HBondDonors: { type: "reverse_sigmoid", low: 0, high: 6, k: 0.5 },
  HBondAcceptors: { type: "reverse_sigmoid", low: 0, high: 12, k: 0.5 },
  NumRotBond: { type: "reverse_sigmoid", low: 0, high: 10, k: 0.5 },
  NumAromaticRings: { type: "reverse_sigmoid", low: 0, high: 5, k: 0.5 },
  NumHeavyAtoms: { type: "double_sigmoid", low: 15, high: 40, coef_div: 40, coef_si: 20, coef_se: 20 },
  SAScore: { type: "reverse_sigmoid", low: 1, high: 4, k: 0.4 },
  TanimotoSimilarity: { type: "sigmoid", low: 0.2, high: 0.6, k: 0.5 },
};

const ADMET_TRANSFORMS: Record<string, Transform> = {
  logp: { type: "double_sigmoid", low: 1, high: 3, coef_div: 3, coef_si: 10, coef_se: 10 },
  logd: { type: "double_sigmoid", low: 1, high: 3, coef_div: 3, coef_si: 10, coef_se: 10 },
  aqueous_solubility: { type: "sigmoid", low: -6, high: -3, k: 0.5 },
  kinetic_solubility: { type: "sigmoid", low: -6, high: -3, k: 0.5 },
  caco2_permeability: { type: "sigmoid", low: -6.5, high: -4.5, k: 0.5 },
  efflux_mdck: { type: "reverse_sigmoid", low: 0.3, high: 1.3, k: 0.5 },
  efflux_caco2: { type: "reverse_sigmoid", low: 0.3, high: 1.3, k: 0.5 },
};

const ENDPOINT_NAMES: Record<string, string> = {
  logp: "LogP",
  logd: "LogD",
  aqueous_solubility: "Aqueous solubility",
  kinetic_solubility: "Kinetic solubility",
  caco2_permeability: "Caco-2 permeability",
  efflux_mdck: "MDCK efflux",
  efflux_caco2: "Caco-2 efflux",
};

const COMPONENT_NAMES: Record<string, string> = {
  QED: "QED",
  MolecularWeight: "Molecular weight",
  SlogP: "cLogP",
  TPSA: "TPSA",
  HBondDonors: "H-bond donors",
  HBondAcceptors: "H-bond acceptors",
  NumRotBond: "Rotatable bonds",
  SAScore: "Synthetic accessibility",
  TanimotoSimilarity: "Similarity to reference",
};

export const DEFAULT_SERVICE_URLS: Record<string, string> = {
  boltz: "http://boltz:8080/score",
  posebusters: "http://posebusters:8080/score",
  admet: "http://admet-predictor:8084/score",
  retrodmta: "http://retrodmta-predictor:8085/score",
  interaction_profiler: "http://interaction-profiler:8080/score",
};

const serviceUrl = (services: Record<string, ServiceInfo>, service: string) => services[service]?.url ?? DEFAULT_SERVICE_URLS[service] ?? "";

export const BOLTZ_NAME = "Boltz affinity";

export function nativeObjective(component: string, weight = 1): Objective {
  const transform = NATIVE_TRANSFORMS[component];
  const objective: Objective = { name: COMPONENT_NAMES[component] ?? component, type: "native", component, weight, transform: transform ? { ...transform } : null };
  if (component === "TanimotoSimilarity") objective.params = { smiles: [], radius: 3, use_counts: true, use_features: false };
  return objective;
}

export function serviceObjective(services: Record<string, ServiceInfo>, service: string, selector?: string, target?: string): Objective {
  const base = { type: "external" as const, service, service_url: serviceUrl(services, service), weight: 1 };
  switch (service) {
    case "boltz":
      return { ...base, name: BOLTZ_NAME, transform: { type: "reverse_sigmoid", low: -3, high: 1, k: 1 }, comment: "log10 IC50 in µM; lower binds tighter" };
    case "posebusters":
      return { ...base, name: "Pose quality", weight: 0.5, depends_on: BOLTZ_NAME, transform: null, comment: "PoseBusters checks passed, 0-1" };
    case "interaction_profiler":
      return {
        ...base,
        name: `${selector ?? "hbond"} contacts`,
        weight: 0.5,
        depends_on: BOLTZ_NAME,
        service_params: { interaction_type: selector ?? "hbond" },
        transform: { type: "sigmoid", low: 0, high: 4, k: 0.5 },
      };
    case "admet": {
      const endpoint = selector ?? services.admet?.endpoints[0] ?? "logp";
      const transform = ADMET_TRANSFORMS[endpoint];
      return { ...base, name: ENDPOINT_NAMES[endpoint] ?? endpoint, service_params: { endpoint }, transform: transform ? { ...transform } : null };
    }
    case "retrodmta": {
      const endpoint = selector ?? "";
      const potency = /_p(ki|kd|ic50|ec50)$/i.test(endpoint);
      return {
        ...base,
        name: endpoint || target || "RetroDMTA",
        service_params: { target: target ?? "", ...(endpoint ? { endpoint } : {}) },
        transform: potency ? { type: "sigmoid", low: 5, high: 9, k: 0.5 } : null,
      };
    }
    default:
      return { ...base, name: service, transform: null };
  }
}

export type PresetId = "cofolding" | "admet" | "druglike";

export const PRESETS: { id: PresetId; label: string; hint: string }[] = [
  { id: "cofolding", label: "Cofolding", hint: "Boltz-2 affinity and PoseBusters pose checks on the target, plus drug-likeness. Needs a protein sequence." },
  { id: "admet", label: "ADMET", hint: "Predicted logD, solubility, permeability and efflux, plus QED. No target needed." },
  { id: "druglike", label: "Drug-likeness", hint: "Fast RDKit descriptors only: QED, weight, cLogP, TPSA, synthetic accessibility." },
];

export function presetObjectives(preset: PresetId, services: Record<string, ServiceInfo>): Objective[] {
  const native = (component: string, weight: number) => nativeObjective(component, weight);
  switch (preset) {
    case "cofolding":
      return [
        { ...serviceObjective(services, "boltz"), weight: 1 },
        { ...serviceObjective(services, "posebusters"), weight: 0.6 },
        native("QED", 0.6),
        native("MolecularWeight", 0.4),
        native("SAScore", 0.4),
      ];
    case "admet":
      return [
        { ...serviceObjective(services, "admet", "logd"), weight: 0.8 },
        { ...serviceObjective(services, "admet", "aqueous_solubility"), weight: 0.6 },
        { ...serviceObjective(services, "admet", "caco2_permeability"), weight: 0.6 },
        { ...serviceObjective(services, "admet", "efflux_mdck"), weight: 0.4 },
        native("QED", 0.6),
      ];
    case "druglike":
      return [native("QED", 1), native("MolecularWeight", 0.6), native("SlogP", 0.6), native("TPSA", 0.4), native("SAScore", 0.6)];
  }
}

/** An objective the "Add" menu offers. */
export interface CatalogueEntry {
  key: string;
  group: string;
  label: string;
  make: () => Objective;
}

const PREFERRED_NATIVE = ["QED", "MolecularWeight", "SlogP", "TPSA", "HBondDonors", "HBondAcceptors", "NumRotBond", "SAScore", "TanimotoSimilarity"];

/** Every objective the builder can add, grouped for a <select>. */
export function objectiveCatalogue(services: Record<string, ServiceInfo>, components: string[]): CatalogueEntry[] {
  const entries: CatalogueEntry[] = [];
  const has = (name: string) => name in services || Object.keys(services).length === 0;
  if (has("boltz")) entries.push({ key: "boltz", group: "Structure", label: "Boltz-2 binding affinity", make: () => serviceObjective(services, "boltz") });
  if (has("posebusters")) entries.push({ key: "posebusters", group: "Structure", label: "PoseBusters pose quality (after Boltz)", make: () => serviceObjective(services, "posebusters") });
  if (has("interaction_profiler")) {
    entries.push({ key: "ip:hbond", group: "Structure", label: "H-bond contacts (after Boltz)", make: () => serviceObjective(services, "interaction_profiler", "hbond") });
  }
  const endpoints = services.admet?.endpoints.length ? services.admet.endpoints : has("admet") ? Object.keys(ADMET_TRANSFORMS) : [];
  for (const endpoint of endpoints) {
    entries.push({ key: `admet:${endpoint}`, group: "ADMET", label: ENDPOINT_NAMES[endpoint] ?? endpoint, make: () => serviceObjective(services, "admet", endpoint) });
  }
  for (const [target, list] of Object.entries(services.retrodmta?.targets ?? {})) {
    for (const endpoint of list.length ? list : [""]) {
      entries.push({
        key: `retrodmta:${target}:${endpoint}`,
        group: "RetroDMTA",
        label: endpoint || target,
        make: () => serviceObjective(services, "retrodmta", endpoint || undefined, target),
      });
    }
  }
  const available = components.length ? new Set(components) : new Set(PREFERRED_NATIVE);
  const native = [...PREFERRED_NATIVE.filter((c) => available.has(c) || c === "TanimotoSimilarity"), ...[...available].filter((c) => !PREFERRED_NATIVE.includes(c))];
  for (const component of native) {
    entries.push({ key: `native:${component}`, group: "RDKit descriptors", label: COMPONENT_NAMES[component] ? `${COMPONENT_NAMES[component]} (${component})` : component, make: () => nativeObjective(component) });
  }
  return entries;
}

/** A name not yet taken: "QED", "QED 2", ... */
export function uniqueName(name: string, taken: string[]): string {
  if (!taken.includes(name)) return name;
  let n = 2;
  while (taken.includes(`${name} ${n}`)) n += 1;
  return `${name} ${n}`;
}

/** Rename one objective and keep the depends_on links pointing at it. */
export function renameObjective(list: Objective[], index: number, name: string): Objective[] {
  const previous = list[index]?.name;
  return list.map((o, i) => (i === index ? { ...o, name } : previous && o.depends_on === previous ? { ...o, depends_on: name } : o));
}

export interface TtpForm {
  name: string;
  id: string;
  description: string;
  sequence: string;
  pdbId: string;
  aggregation: "geometric_mean" | "arithmetic_mean";
  objectives: Objective[];
}

/** What stops the form from making a TTP the server accepts. */
export function formProblems(form: TtpForm, existingIds: string[] = []): string[] {
  const problems: string[] = [];
  if (!form.name.trim()) problems.push("Give the profile a name.");
  if (!form.id) problems.push("The id is empty.");
  else if (!TTP_ID.test(form.id)) problems.push("The id may only hold lowercase letters, digits, - and _, starting with a letter or digit.");
  else if (existingIds.includes(form.id)) problems.push(`A TTP with id "${form.id}" already exists.`);
  const sequence = normaliseSequence(form.sequence);
  const seqProblem = sequenceProblem(sequence);
  if (seqProblem) problems.push(seqProblem);
  if (form.objectives.length === 0) problems.push("Add at least one objective.");
  const names = form.objectives.map((o) => o.name.trim());
  if (names.some((n) => !n)) problems.push("Every objective needs a name.");
  const dupes = [...new Set(names.filter((n, i) => n && names.indexOf(n) !== i))];
  if (dupes.length) problems.push(`Objective names must be unique: ${dupes.join(", ")}.`);
  for (const o of form.objectives) {
    if (!(o.weight > 0)) problems.push(`${o.name || "An objective"} needs a weight above 0.`);
    if (o.depends_on && !names.includes(o.depends_on)) problems.push(`${o.name} runs after "${o.depends_on}", which is not in the profile.`);
    if (o.type === "external" && !o.service_url) problems.push(`${o.name} has no service URL.`);
    if (o.service === "retrodmta" && !o.service_params?.target) problems.push(`${o.name} needs a RetroDMTA target.`);
    if (o.component === "TanimotoSimilarity") {
      const smiles = o.params?.smiles;
      if (!Array.isArray(smiles) || !smiles.some((s) => typeof s === "string" && s.trim())) problems.push(`${o.name} needs a reference SMILES.`);
    }
    for (const key of transformParams(o.transform?.type ?? "")) {
      if (typeof o.transform?.[key] !== "number" || !Number.isFinite(o.transform[key])) problems.push(`${o.name}: the transform's ${key} is not a number.`);
    }
  }
  if (form.objectives.some((o) => o.service === "boltz") && !sequence) problems.push("Boltz scoring needs the target's protein sequence.");
  return problems;
}

/** The create_ttp argument for a form; empty optional fields are left out. */
export function buildTtp(form: TtpForm): Record<string, unknown> {
  const sequence = normaliseSequence(form.sequence);
  const objectives = form.objectives.map((o) => {
    const clean: Record<string, unknown> = { name: o.name.trim(), type: o.type, weight: o.weight };
    if (o.type === "native") clean.component = o.component;
    else {
      clean.service = o.service;
      clean.service_url = o.service_url;
      if (o.service_params && Object.keys(o.service_params).length) clean.service_params = o.service_params;
    }
    if (o.depends_on) clean.depends_on = o.depends_on;
    if (o.transform) clean.transform = o.transform;
    if (o.params && Object.keys(o.params).length) clean.params = o.params;
    if (o.comment) clean.comment = o.comment;
    return clean;
  });
  const ttp: Record<string, unknown> = {
    id: form.id,
    name: form.name.trim(),
    scoring: { ...(form.description.trim() ? { comment: form.description.trim() } : {}), aggregation: form.aggregation, objectives },
  };
  if (sequence) ttp.target = { protein_sequence: sequence, ...(form.pdbId.trim() ? { pdb_id: form.pdbId.trim().toUpperCase() } : {}) };
  return ttp;
}

/** The id and name create_ttp answered with, or its error. */
export function createdTtp(value: unknown): { id: string; name: string } | { error: string } {
  const parsed = parseResult(value);
  const error = payloadError(parsed);
  if (error) return { error };
  if (isRecord(parsed) && typeof parsed.id === "string") return { id: parsed.id, name: str(parsed.name) ?? parsed.id };
  return { error: typeof value === "string" && value.trim() ? value.trim() : "create_ttp gave no answer." };
}

// ---------------------------------------------------------------------------
// target facts
// ---------------------------------------------------------------------------

export interface PocketResidue {
  chain: string | null;
  resname: string;
  resnum: number | null;
  /** "ASN140". */
  label: string;
  /** Contacts with this residue across every profiled complex. */
  contacts: number;
  /** Interaction types seen: "hbond", "hydrophobic", ... */
  types: string[];
}

/** What the session has established about the protein target. */
export interface TargetFacts {
  name: string | null;
  gene: string | null;
  organism: string | null;
  uniprot: string | null;
  entryName: string | null;
  reviewed: boolean | null;
  function: string | null;
  sequence: string | null;
  sequenceLength: number | null;
  chains: string[];
  pdbIds: string[];
  pdbCount: number | null;
  msaPath: string | null;
  proteinPdbPath: string | null;
  ttpId: string | null;
  pocket: PocketResidue[];
  /** Number of complexes the pocket was profiled on. */
  profiledComplexes: number;
  /** Tools the facts came from, in the order first seen. */
  sources: string[];
}

export const emptyTargetFacts = (): TargetFacts => ({
  name: null,
  gene: null,
  organism: null,
  uniprot: null,
  entryName: null,
  reviewed: null,
  function: null,
  sequence: null,
  sequenceLength: null,
  chains: [],
  pdbIds: [],
  pdbCount: null,
  msaPath: null,
  proteinPdbPath: null,
  ttpId: null,
  pocket: [],
  profiledComplexes: 0,
  sources: [],
});

export const hasTargetFacts = (f: TargetFacts) =>
  Boolean(f.name || f.uniprot || f.sequence || f.pocket.length || f.pdbIds.length || f.gene);

const ACCESSION = /^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$/;
const SEQUENCE = /^[ACDEFGHIKLMNPQRSTVWYXBZUO]{10,}$/;
const SEQUENCE_KEYS = new Set(["protein_sequence", "target_sequence", "sequence", "protein", "receptor_sequence"]);

export const isAccession = (text: string) => ACCESSION.test(text.trim().toUpperCase());

function cleanSequence(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const seq = value.replace(/\s+/g, "").toUpperCase();
  return SEQUENCE.test(seq) ? seq : null;
}

/** A protein sequence anywhere in a call's arguments, a few levels deep. */
function findSequence(node: unknown, depth = 0): string | null {
  if (depth > 4) return null;
  if (Array.isArray(node)) {
    for (const item of node) {
      const found = findSequence(item, depth + 1);
      if (found) return found;
    }
    return null;
  }
  if (!isRecord(node)) return null;
  for (const [key, value] of Object.entries(node)) {
    if (SEQUENCE_KEYS.has(key)) {
      const seq = cleanSequence(value);
      if (seq) return seq;
    }
  }
  for (const value of Object.values(node)) {
    if (typeof value === "object") {
      const found = findSequence(value, depth + 1);
      if (found) return found;
    }
  }
  return null;
}

/** Facts from one fetch_target_info result (a UniProt entry). */
export function mergeUniprot(facts: TargetFacts, payload: unknown, tool = "fetch_target_info"): TargetFacts {
  const entry = parseResult(payload);
  if (!isRecord(entry) || typeof entry.error === "string") return facts;
  const sequence = cleanSequence(entry.sequence);
  const pdbIds = Array.isArray(entry.pdb_ids) ? entry.pdb_ids.filter((id): id is string => typeof id === "string") : [];
  return withSource(
    {
      ...facts,
      name: str(entry.protein_name) ?? facts.name,
      gene: str(entry.gene) ?? facts.gene,
      organism: str(entry.organism) ?? facts.organism,
      uniprot: str(entry.accession) ?? facts.uniprot,
      entryName: str(entry.entry_name) ?? facts.entryName,
      reviewed: typeof entry.reviewed === "boolean" ? entry.reviewed : facts.reviewed,
      function: str(entry.function) ?? facts.function,
      sequence: sequence ?? facts.sequence,
      sequenceLength: num(entry.sequence_length) ?? (sequence ? sequence.length : facts.sequenceLength),
      pdbIds: pdbIds.length ? pdbIds : facts.pdbIds,
      pdbCount: num(entry.pdb_count) ?? facts.pdbCount,
    },
    tool,
  );
}

function withSource(facts: TargetFacts, tool: string): TargetFacts {
  return facts.sources.includes(tool) ? facts : { ...facts, sources: [...facts.sources, tool] };
}

function mergeTtpTarget(facts: TargetFacts, ttp: Ttp, tool: string): TargetFacts {
  const target = ttp.target;
  if (!target) return { ...facts, ttpId: ttp.id };
  const sequence = cleanSequence(target.protein_sequence);
  const pdb = str(target.pdb_id)?.toUpperCase() ?? null;
  return withSource(
    {
      ...facts,
      ttpId: ttp.id,
      sequence: sequence ?? facts.sequence,
      sequenceLength: sequence ? sequence.length : facts.sequenceLength,
      pdbIds: pdb && !facts.pdbIds.includes(pdb) ? [pdb, ...facts.pdbIds] : facts.pdbIds,
      msaPath: str(target.msa_path) ?? facts.msaPath,
      proteinPdbPath: str(target.protein_pdb_path) ?? facts.proteinPdbPath,
    },
    tool,
  );
}

const residueKey = (c: Pick<Contact, "chain" | "resname" | "resnum">) => `${c.chain ?? ""}:${c.resname}:${c.resnum ?? ""}`;

/**
 * Everything the session's tool calls say about the target, oldest event
 * first so later lookups win. Pocket residues are the residues ligands
 * contacted in profiled complexes, most contacted first.
 */
export function extractTargetFacts(events: ToolEvent[]): TargetFacts {
  let facts = emptyTargetFacts();
  const pocket = new Map<string, PocketResidue>();
  const chains = new Set<string>();

  for (const event of events) {
    if (event.type === "tool_call") {
      const ttp = asTtp(event.arguments.ttp);
      if (ttp) facts = mergeTtpTarget(facts, ttp, event.name);
      else {
        const seq = findSequence(event.arguments);
        if (seq) facts = withSource({ ...facts, sequence: seq, sequenceLength: seq.length }, event.name);
      }
      const query = str(event.arguments.query);
      if (event.name === "fetch_target_info" && query && isAccession(query) && !facts.uniprot) {
        facts = { ...facts, uniprot: query.toUpperCase() };
      }
      continue;
    }

    if (event.isError) continue;
    if (event.name === "fetch_target_info") {
      facts = mergeUniprot(facts, event.result, event.name);
    } else {
      const ttp = ttpIn(event);
      if (ttp) facts = mergeTtpTarget(facts, ttp, event.name);
    }

    for (const panel of event.panels ?? []) {
      if (panel.kind !== "interactions" || panel.contacts.length === 0) continue;
      facts = withSource({ ...facts, profiledComplexes: facts.profiledComplexes + 1 }, event.name);
      for (const contact of panel.contacts) {
        if (contact.chain) chains.add(contact.chain);
        const key = residueKey(contact);
        const known = pocket.get(key);
        if (known) {
          known.contacts += 1;
          if (!known.types.includes(contact.type)) known.types.push(contact.type);
        } else {
          pocket.set(key, {
            chain: contact.chain,
            resname: contact.resname,
            resnum: contact.resnum,
            label: contact.label || `${contact.resname}${contact.resnum ?? ""}`,
            contacts: 1,
            types: [contact.type],
          });
        }
      }
    }
  }

  facts.chains = [...chains].sort();
  facts.pocket = [...pocket.values()].sort(
    (a, b) => b.contacts - a.contacts || (a.chain ?? "").localeCompare(b.chain ?? "") || (a.resnum ?? 0) - (b.resnum ?? 0),
  );
  return facts;
}

/** Pocket residues in sequence order, for a compact "ASN140 TYR97 …" line. */
export function pocketBySequence(pocket: PocketResidue[]): PocketResidue[] {
  return [...pocket].sort((a, b) => (a.chain ?? "").localeCompare(b.chain ?? "") || (a.resnum ?? 0) - (b.resnum ?? 0));
}
