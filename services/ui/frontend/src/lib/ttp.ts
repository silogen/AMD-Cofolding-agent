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

/** A structure Boltz-2 cofolds against (its YAML `templates:`); a forced template needs a threshold in Å. */
export interface TemplateConfig {
  path: string;
  chain_id?: string | null;
  force?: boolean;
  threshold?: number | null;
}

export interface TtpTarget {
  /** Absent when structure_path gives the protein. */
  protein_sequence?: string | null;
  /** A structure on the shared volume whose chain structure_chain the server folds. */
  structure_path?: string | null;
  structure_chain?: string | null;
  pdb_id?: string | null;
  msa_path?: string | null;
  protein_pdb_path?: string | null;
  templates?: TemplateConfig[] | null;
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
      return ["low", "high"];
    case "right_step":
    case "left_step":
      return ["low"];
    case "exponential_decay":
      return ["k"];
    default:
      return [];
  }
}

/** Plain-language names for the transform types. */
export const CURVE_LABELS: Record<string, string> = {
  sigmoid: "Higher is better",
  reverse_sigmoid: "Lower is better",
  double_sigmoid: "Within a range",
  step: "Inside a window (hard)",
  right_step: "At least (hard)",
  left_step: "At most (hard)",
  exponential_decay: "Decay from zero",
  value_mapping: "Value mapping",
};

export const curveLabel = (type: string | null | undefined) => (type ? (CURVE_LABELS[normaliseType(type)] ?? type) : "Score as is");

/** The parameters shown in the simple view, with the words that read them out. */
export function rangeFields(type: string): { key: keyof Transform & string; label: string }[] {
  switch (normaliseType(type)) {
    case "sigmoid":
    case "reverse_sigmoid":
      return [
        { key: "low", label: "from" },
        { key: "high", label: "to" },
      ];
    case "double_sigmoid":
    case "step":
      return [
        { key: "low", label: "between" },
        { key: "high", label: "and" },
      ];
    case "right_step":
      return [{ key: "low", label: "at least" }];
    case "left_step":
      return [{ key: "low", label: "at most" }];
    case "exponential_decay":
      return [{ key: "k", label: "rate" }];
    default:
      return [];
  }
}

/** The parameters behind "more": steepness and the double sigmoid's coefficients. */
export function advancedParams(type: string): (keyof Transform & string)[] {
  const shown = new Set(rangeFields(type).map((f) => f.key));
  return transformParams(type).filter((p) => !shown.has(p));
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
  right_step: { low: 0 },
  left_step: { low: 0 },
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
      // a / (a + c) written as 1 / (1 + c / a), so large x/div do not overflow to NaN.
      const left = 1 / (1 + 10 ** ((se * (low - x)) / div));
      const right = 1 / (1 + 10 ** ((si * (high - x)) / div));
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

/** The raw-value range that shows a transform's shape. */
export function curveDomain(t: Transform): { from: number; to: number } {
  const type = normaliseType(t.type);
  if (type === "exponential_decay") return { from: 0, to: 5 / Math.max(num(t.k) ?? 1, 1e-3) };
  const low = num(t.low) ?? 0;
  const high = type === "right_step" || type === "left_step" ? low : (num(t.high) ?? 1);
  const span = Math.abs(high - low) || Math.max(Math.abs(low), 1);
  return { from: Math.min(low, high) - span * 0.6, to: Math.max(low, high) + span * 0.6 };
}

/** Sampled curve of a transform over `domain`, by default one wide enough to show its shape. */
export function transformCurve(t: Transform, samples = 48, domain = curveDomain(t)): { x: number; y: number }[] {
  if (evaluateTransform(t, 0) === null) return [];
  const { from, to } = domain;
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
  return weightRangeOf([...objectivesOf(ttp, "pre_scoring"), ...objectivesOf(ttp, "scoring")]);
}

export function weightRangeOf(all: Objective[]): { min: number; max: number; step: number } {
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

const SERVICE_LABELS: Record<string, string> = {
  boltz: "Boltz-2",
  posebusters: "PoseBusters",
  admet: "ADMET",
  retrodmta: "RetroDMTA",
  interaction_profiler: "Interactions",
};

/** Where a score comes from, in a word: "RDKit", "Boltz-2", "ADMET", ... */
export function sourceLabel(o: Pick<Objective, "type" | "service">): string {
  if (o.type === "native") return "RDKit";
  return (o.service && SERVICE_LABELS[o.service]) ?? o.service ?? "external";
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

/** Native REINVENT4 scorers a profile can use without extra settings (none or only optional ones), from list_scoring_components. */
export function parseComponents(value: unknown): string[] {
  const parsed = parseResult(value);
  if (!Array.isArray(parsed)) return [];
  return parsed
    .filter(isRecord)
    .filter((c) => c.component_type === "scorer" && typeof c.class === "string" && !String(c.class).startsWith("Fragment"))
    .filter((c) => !Array.isArray(c.parameters) || c.parameters.every((p) => isRecord(p) && p.required === false))
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

/** The slug of *name*, numbered past any id already taken. */
export function freeSlug(name: string, taken: readonly string[]): string {
  const base = slugify(name);
  if (!base || !taken.includes(base)) return base;
  let n = 2;
  while (taken.includes(`${base}-${n}`)) n++;
  return `${base}-${n}`;
}

export const TTP_ID = /^[a-z0-9][a-z0-9_-]*$/;
const TEMPLATE_FILE = /\.(cif|mmcif|pdb|ent)$/i;
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
      return { ...base, name: BOLTZ_NAME, transform: { type: "reverse_sigmoid", low: -1.5, high: 2.5, k: 0.5 }, comment: "log10 IC50 in µM; lower binds tighter" };
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

/** What an objective scores, whatever it is named: "native:QED", "admet:logd", "boltz:", ... */
export function objectiveKey(o: Pick<Objective, "type" | "component" | "service" | "service_params">): string {
  if (o.type === "native") return `native:${o.component ?? ""}`;
  const p = o.service_params ?? {};
  const selector = [p.target, p.endpoint, p.interaction_type].filter((v) => typeof v === "string" && v).join(":");
  return `${o.service ?? ""}:${selector}`;
}

/** Windows for scores that are already 0-1, used once a curve is chosen for them. */
const UNIT_WINDOWS: Record<string, [number, number]> = {
  QED: [0.4, 0.8],
  Csp3: [0.2, 0.6],
  posebusters: [0.5, 1],
};

/** The curve an objective starts from when its curve type is first set: its usual window. */
export function defaultTransform(o: Objective): Transform | null {
  const known =
    o.type === "native"
      ? NATIVE_TRANSFORMS[o.component ?? ""]
      : o.service === "admet"
        ? ADMET_TRANSFORMS[String(o.service_params?.endpoint ?? "")]
        : o.service
          ? serviceObjective({}, o.service, typeof o.service_params?.interaction_type === "string" ? o.service_params.interaction_type : undefined).transform
          : null;
  if (known) return { ...known };
  const unit = UNIT_WINDOWS[o.component ?? o.service ?? ""];
  return unit ? { type: "sigmoid", low: unit[0], high: unit[1], k: 0.5 } : null;
}

/** The double sigmoid's coef_div that keeps its steepness in scale with the window. */
const scaleDiv = (t: Transform) => Math.max(Math.abs(num(t.low) ?? 0), Math.abs(num(t.high) ?? 0)) || 1;

/** The objective's transform as another curve type, keeping its window; null for "score as is". */
export function retypeObjective(o: Objective, type: string | null): Transform | null {
  if (!type) return null;
  const seed = o.transform ?? defaultTransform(o);
  const next = retype(seed, type);
  if (normaliseType(type) === "double_sigmoid" && typeof seed?.coef_div !== "number") next.coef_div = scaleDiv(next);
  return next;
}

/** A transform with a new window; a double sigmoid's coef_div follows when it was in scale. */
export function setRange(t: Transform, patch: Partial<Pick<Transform, "low" | "high" | "k">>): Transform {
  const next = { ...t, ...patch };
  if (normaliseType(t.type) === "double_sigmoid" && (typeof t.coef_div !== "number" || t.coef_div === scaleDiv(t))) next.coef_div = scaleDiv(next);
  return next;
}

export const hasCofolding = (list: Objective[]) => list.some((o) => o.service === "boltz");

/**
 * Objectives with cofolding switched on (Boltz-2 affinity, then PoseBusters on
 * its pose) or off (those and anything that runs after Boltz removed).
 */
export function withCofolding(list: Objective[], services: Record<string, ServiceInfo>, on: boolean): Objective[] {
  if (on) {
    if (hasCofolding(list)) return list;
    const taken = list.map((o) => o.name);
    const boltz = { ...serviceObjective(services, "boltz"), name: uniqueName(BOLTZ_NAME, taken) };
    const pose = { ...serviceObjective(services, "posebusters"), weight: 0.6, depends_on: boltz.name };
    return [boltz, { ...pose, name: uniqueName(pose.name, [...taken, boltz.name]) }, ...list];
  }
  const boltzNames = new Set(list.filter((o) => o.service === "boltz").map((o) => o.name));
  return list.filter((o) => o.service !== "boltz" && o.service !== "posebusters" && !(o.depends_on && boltzNames.has(o.depends_on)));
}

/** An objective the picker offers: one checkbox, for the objectives with its objectiveKey. */
export interface CatalogueEntry {
  /** The objectiveKey of what `make` returns. */
  key: string;
  group: string;
  label: string;
  make: () => Objective;
  /** Shown once the group is expanded: the less common native scorers. */
  extra?: boolean;
  /** The key of the objective it runs after (a Boltz pose); it cannot be added without it. */
  after?: string;
}

export interface ObjectiveGroup {
  name: string;
  entries: CatalogueEntry[];
}

const PREFERRED_NATIVE = ["QED", "MolecularWeight", "SlogP", "TPSA", "HBondDonors", "HBondAcceptors", "NumRotBond", "SAScore", "TanimotoSimilarity"];
const CORE_NATIVE = new Set(["QED", "MolecularWeight", "SlogP", "TPSA", "SAScore"]);
const BOLTZ_KEY = "boltz:";

/** Starting weights, as the presets have them; anything else starts at 0.5. */
const WEIGHTS: Record<string, number> = {
  [BOLTZ_KEY]: 1,
  "posebusters:": 0.6,
  "native:QED": 1,
  "native:MolecularWeight": 0.6,
  "native:SlogP": 0.6,
  "native:TPSA": 0.4,
  "native:SAScore": 0.6,
  "admet:logd": 0.8,
  "admet:aqueous_solubility": 0.6,
  "admet:caco2_permeability": 0.6,
  "admet:efflux_mdck": 0.4,
};

function catalogueEntry(group: string, label: string, make: () => Objective, flags: Pick<CatalogueEntry, "extra" | "after"> = {}): CatalogueEntry {
  const key = objectiveKey(make());
  return { key, group, label, make: () => ({ ...make(), weight: WEIGHTS[key] ?? 0.5 }), ...flags };
}

/** Every objective the builder offers, in picker order: Cofold, Drug-likeness, ADMET, RetroDMTA. */
export function objectiveCatalogue(services: Record<string, ServiceInfo>, components: string[]): CatalogueEntry[] {
  const entries: CatalogueEntry[] = [];
  const has = (name: string) => name in services || Object.keys(services).length === 0;
  if (has("boltz")) {
    entries.push(catalogueEntry("Cofold", "Binding affinity", () => serviceObjective(services, "boltz")));
    if (has("posebusters")) entries.push(catalogueEntry("Cofold", "Pose quality", () => serviceObjective(services, "posebusters"), { after: BOLTZ_KEY }));
    if (has("interaction_profiler")) entries.push(catalogueEntry("Cofold", "H-bond contacts", () => serviceObjective(services, "interaction_profiler", "hbond"), { after: BOLTZ_KEY }));
  }
  const lower = (names: string[]) => new Set(names.map((c) => c.toLowerCase()));
  const available = lower(components.length ? components : PREFERRED_NATIVE);
  const preferred = lower(PREFERRED_NATIVE);
  const native = [
    ...PREFERRED_NATIVE.filter((c) => available.has(c.toLowerCase()) || CORE_NATIVE.has(c) || c === "TanimotoSimilarity"),
    ...components.filter((c) => !preferred.has(c.toLowerCase())),
  ];
  for (const component of native) {
    entries.push(
      catalogueEntry("Drug-likeness", COMPONENT_NAMES[component] ?? component, () => nativeObjective(component), {
        extra: !PREFERRED_NATIVE.includes(component),
      }),
    );
  }
  const endpoints = services.admet?.endpoints.length ? services.admet.endpoints : has("admet") ? Object.keys(ADMET_TRANSFORMS) : [];
  for (const endpoint of endpoints) entries.push(catalogueEntry("ADMET", ENDPOINT_NAMES[endpoint] ?? endpoint, () => serviceObjective(services, "admet", endpoint)));
  for (const [target, list] of Object.entries(services.retrodmta?.targets ?? {})) {
    for (const endpoint of list.length ? list : [""]) {
      entries.push(catalogueEntry("RetroDMTA", endpoint || target, () => serviceObjective(services, "retrodmta", endpoint || undefined, target)));
    }
  }
  return entries;
}

export function objectiveGroups(catalogue: CatalogueEntry[]): ObjectiveGroup[] {
  const groups: ObjectiveGroup[] = [];
  for (const e of catalogue) {
    const group = groups.find((g) => g.name === e.group);
    if (group) group.entries.push(e);
    else groups.push({ name: e.group, entries: [e] });
  }
  return groups;
}

/** What a new profile starts with: cofolding, pose checks, QED, synthetic accessibility and two ADMET endpoints, as far as the catalogue offers them. */
const DEFAULT_KEYS = [BOLTZ_KEY, "posebusters:", "native:QED", "native:SAScore", "admet:logd", "admet:aqueous_solubility"];

export function defaultObjectives(catalogue: CatalogueEntry[]): Objective[] {
  return DEFAULT_KEYS.reduce<Objective[]>((list, key) => {
    const entry = catalogue.find((e) => e.key === key);
    return entry ? toggleObjective(list, entry, true) : list;
  }, []);
}

/** One hue per picker group, from the theme tokens so it follows light and dark. */
export const GROUP_COLOURS: Record<string, string> = {
  Cofold: "var(--color-structure)",
  "Drug-likeness": "var(--color-ok)",
  ADMET: "var(--color-table)",
  RetroDMTA: "var(--color-molecule)",
};

/** The picker group an objective belongs to, or null for a custom endpoint. */
export function objectiveGroup(o: Pick<Objective, "type" | "service">): string | null {
  if (o.type === "native") return "Drug-likeness";
  switch (o.service) {
    case "boltz":
    case "posebusters":
    case "interaction_profiler":
      return "Cofold";
    case "admet":
      return "ADMET";
    case "retrodmta":
      return "RetroDMTA";
    default:
      return null;
  }
}

export const groupColour = (group: string | null) => (group && GROUP_COLOURS[group]) || "var(--text-faint)";

/** The list without the objectives *drop* picks, and without whatever ran after them. */
export function dropObjectives(list: Objective[], drop: (o: Objective) => boolean): Objective[] {
  const gone = new Set(list.filter(drop).map((o) => o.name));
  for (let size = -1; size !== gone.size; ) {
    size = gone.size;
    for (const o of list) if (o.depends_on && gone.has(o.depends_on)) gone.add(o.name);
  }
  return gone.size ? list.filter((o) => !gone.has(o.name)) : list;
}

/**
 * The list with one catalogue objective added at the end, or every objective
 * with its key removed (with what ran after them). Something that runs after
 * a Boltz pose is not added without Boltz.
 */
export function toggleObjective(list: Objective[], entry: CatalogueEntry, on: boolean): Objective[] {
  const present = list.some((o) => objectiveKey(o) === entry.key);
  if (!on) return present ? dropObjectives(list, (o) => objectiveKey(o) === entry.key) : list;
  if (present) return list;
  const objective = entry.make();
  if (entry.after) {
    const producer = list.find((o) => objectiveKey(o) === entry.after);
    if (!producer) return list;
    objective.depends_on = producer.name;
  }
  return [...list, { ...objective, name: uniqueName(objective.name, list.map((o) => o.name)) }];
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
  /** Carried over when a stored TTP is duplicated: its scoring comment. */
  description: string;
  sequence: string;
  /**
   * A structure on the shared volume the server reads the protein from
   * (its chain structureChain); the sequence is then not sent.
   */
  structurePath?: string;
  structureChain?: string | null;
  pdbId: string;
  /** A precomputed MSA, relative to the shared volume; Boltz folds its query sequence. */
  msaPath?: string;
  /**
   * Carried over when a stored TTP is duplicated, never set from a picked
   * structure: PoseBusters checks poses against this protein instead of the
   * Boltz complex, so it must be in the frame of the predicted poses.
   */
  proteinPdbPath?: string | null;
  /** Template structures, relative to the shared volume. */
  templates?: TemplateConfig[];
  aggregation: "geometric_mean" | "arithmetic_mean";
  objectives: Objective[];
  /** Carried over when a stored TTP is duplicated; the builder has no editor for them. */
  constraints?: Constraint[] | null;
}

/** A form that recreates a stored TTP under a new id; pre-scoring components become objectives. */
export function formFromTtp(ttp: Ttp, existingIds: string[] = []): TtpForm {
  let id = `${ttp.id}-copy`.slice(0, 60);
  for (let n = 2; existingIds.includes(id); n++) id = `${ttp.id}-copy-${n}`;
  return {
    name: `${ttp.name} (copy)`,
    id,
    description: ttp.scoring.comment ?? "",
    sequence: ttp.target?.protein_sequence ?? "",
    structurePath: ttp.target?.structure_path ?? "",
    structureChain: ttp.target?.structure_chain ?? null,
    pdbId: ttp.target?.pdb_id ?? "",
    msaPath: ttp.target?.msa_path ?? "",
    proteinPdbPath: ttp.target?.protein_pdb_path ?? null,
    templates: (ttp.target?.templates ?? []).map((t) => ({ ...t })),
    aggregation: ttp.scoring.aggregation ?? "geometric_mean",
    objectives: [...objectivesOf(ttp, "pre_scoring").map((o) => ({ ...o, type: "native" as const })), ...ttp.scoring.objectives].map((o) => ({ ...o })),
    constraints: ttp.constraints ?? null,
  };
}

/** What stops the form from making a TTP the server accepts. */
export function formProblems(form: TtpForm, existingIds: string[] = []): string[] {
  const problems: string[] = [];
  if (!form.name.trim()) problems.push("Give the profile a name.");
  if (!form.id) {
    if (form.name.trim()) problems.push("The id is empty.");
  }
  else if (!TTP_ID.test(form.id)) problems.push("The id may only hold lowercase letters, digits, - and _, starting with a letter or digit.");
  else if (existingIds.includes(form.id)) problems.push(`A TTP with id "${form.id}" already exists.`);
  const structure = form.structurePath?.trim() ?? "";
  const sequence = structure ? "" : normaliseSequence(form.sequence);
  const seqProblem = sequenceProblem(sequence);
  if (seqProblem) problems.push(seqProblem);
  if (structure && !TEMPLATE_FILE.test(structure)) problems.push(`${structure.split("/").pop()} is not a .cif or .pdb file.`);
  const protein = Boolean(structure || sequence);
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
  if (form.objectives.some((o) => o.service === "boltz") && !protein) problems.push("Cofolding needs the protein: a structure or a sequence.");
  else if (form.msaPath?.trim() && !protein) problems.push("An MSA needs the protein it was built for.");
  for (const t of form.templates ?? []) {
    const name = t.path.split("/").pop() ?? t.path;
    if (!TEMPLATE_FILE.test(t.path.trim())) problems.push(`The template ${name} is not a .cif or .pdb file.`);
    if (t.threshold != null && !(t.threshold > 0)) problems.push(`The template ${name} needs a threshold above 0 Å.`);
    else if (t.force && t.threshold == null) problems.push(`Forcing the template ${name} needs a threshold in Å.`);
  }
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
  const structure = form.structurePath?.trim();
  if (structure || sequence) {
    const target: Record<string, unknown> = structure
      ? { structure_path: structure, ...(form.structureChain?.trim() ? { structure_chain: form.structureChain.trim() } : {}) }
      : { protein_sequence: sequence };
    if (form.pdbId.trim()) target.pdb_id = form.pdbId.trim().toUpperCase();
    if (form.msaPath?.trim()) target.msa_path = form.msaPath.trim();
    if (form.proteinPdbPath?.trim()) target.protein_pdb_path = form.proteinPdbPath.trim();
    if (form.templates?.length) {
      target.templates = form.templates.map((t) => ({
        path: t.path.trim(),
        ...(t.chain_id ? { chain_id: t.chain_id } : {}),
        ...(t.force ? { force: true } : {}),
        ...(t.threshold != null ? { threshold: t.threshold } : {}),
      }));
    }
    ttp.target = target;
  }
  if (form.constraints?.length) ttp.constraints = form.constraints;
  return ttp;
}

/** What "Use in chat" puts in the composer. */
export const chatPrompt = (ttp: Pick<Ttp, "id" | "name">) => `Use the TTP "${ttp.name}" (${ttp.id})`;

export const ttpFileName = (ttp: Pick<Ttp, "id" | "version">) => `${ttp.id}-v${ttp.version}.json`;

/** Stored TTPs whose name or id holds every word of the query. */
export function filterTtps<T extends TtpSummary>(items: T[], query: string): T[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  return items.filter((t) => words.every((w) => t.name.toLowerCase().includes(w) || t.id.toLowerCase().includes(w)));
}

/**
 * Most recently used or created first; list_ttps has no dates, so `activity`
 * (ISO times by id: runs, local creations) orders them, then the name.
 */
export function sortTtps<T extends TtpSummary>(items: T[], activity: Record<string, string>): T[] {
  return [...items].sort((a, b) => (activity[b.id] ?? "").localeCompare(activity[a.id] ?? "") || a.name.localeCompare(b.name));
}

const CREATED_KEY = "cofold.ttp-created";

/** When this browser created each TTP, by id. */
export function createdTimes(storage: KeyValueStore | null = browserStorage()): Record<string, string> {
  try {
    const value: unknown = JSON.parse(storage?.getItem(CREATED_KEY) ?? "{}");
    return isRecord(value) ? (Object.fromEntries(Object.entries(value).filter(([, v]) => typeof v === "string")) as Record<string, string>) : {};
  } catch {
    return {};
  }
}

export function rememberCreated(id: string, at = new Date().toISOString(), storage: KeyValueStore | null = browserStorage()): void {
  try {
    storage?.setItem(CREATED_KEY, JSON.stringify({ ...createdTimes(storage), [id]: at }));
  } catch {
    // Storage full or disabled: the list falls back to name order.
  }
}

export interface VersionDiff {
  added: string[];
  removed: string[];
  /** Per objective: "weight 0.6 → 0.8", "low -6 → -5", "curve Higher is better → Within a range". */
  changed: { name: string; parts: string[] }[];
  /** Other changes in a phrase each: target, aggregation, constraints, name. */
  other: string[];
}

const short = (v: unknown) => (typeof v === "number" ? String(+v.toPrecision(4)) : v == null ? "—" : String(v));

/** What changed between two versions of a TTP, objectives matched by name. */
export function versionDiff(from: Ttp, to: Ttp): VersionDiff {
  const all = (t: Ttp) => new Map([...objectivesOf(t, "pre_scoring"), ...objectivesOf(t, "scoring")].map((o) => [o.name, o]));
  const before = all(from);
  const after = all(to);
  const changed: VersionDiff["changed"] = [];
  for (const [name, b] of after) {
    const a = before.get(name);
    if (!a) continue;
    const parts: string[] = [];
    if (a.weight !== b.weight) parts.push(`weight ${short(a.weight)} → ${short(b.weight)}`);
    const ta = a.transform ?? null;
    const tb = b.transform ?? null;
    if ((ta?.type ?? null) !== (tb?.type ?? null)) parts.push(`curve ${curveLabel(ta?.type)} → ${curveLabel(tb?.type)}`);
    else if (ta && tb) {
      for (const key of transformParams(tb.type)) if (ta[key] !== tb[key]) parts.push(`${key} ${short(ta[key])} → ${short(tb[key])}`);
    }
    if (!same(a.params, b.params)) parts.push("parameters");
    if (!same(a.service_params, b.service_params) || a.service_url !== b.service_url) parts.push("service settings");
    if ((a.depends_on ?? null) !== (b.depends_on ?? null)) parts.push(`runs after ${short(b.depends_on)}`);
    if (parts.length) changed.push({ name, parts });
  }
  const other: string[] = [];
  if (from.name !== to.name) other.push(`renamed to "${to.name}"`);
  const sa = from.target?.protein_sequence ?? "";
  const sb = to.target?.protein_sequence ?? "";
  if (sa !== sb && sa && sb) other.push(`target sequence ${sa.length} → ${sb.length} aa`);
  const pa = from.target?.structure_path ? `${from.target.structure_path}#${from.target.structure_chain ?? ""}` : "";
  const pb = to.target?.structure_path ? `${to.target.structure_path}#${to.target.structure_chain ?? ""}` : "";
  if (pa !== pb && pb) other.push(`target from ${to.target?.structure_path?.split("/").pop()}${to.target?.structure_chain ? ` chain ${to.target.structure_chain}` : ""}`);
  else if (sa !== sb && sb && !sa) other.push(`target added (${sb.length} aa)`);
  else if ((sa || pa) && !sb && !pb) other.push("target removed");
  if ((from.target?.pdb_id ?? null) !== (to.target?.pdb_id ?? null)) other.push(`PDB ${short(from.target?.pdb_id)} → ${short(to.target?.pdb_id)}`);
  const ga = from.scoring.aggregation ?? "geometric_mean";
  const gb = to.scoring.aggregation ?? "geometric_mean";
  if (ga !== gb) other.push(`${ga.replace("_", " ")} → ${gb.replace("_", " ")}`);
  if (!same(from.constraints, to.constraints)) other.push("constraints changed");
  return {
    added: [...after.keys()].filter((n) => !before.has(n)),
    removed: [...before.keys()].filter((n) => !after.has(n)),
    changed,
    other,
  };
}

export const isSameVersion = (d: VersionDiff) => !d.added.length && !d.removed.length && !d.changed.length && !d.other.length;

/** "1,362 aa · PDB 3MXF", "1m17.cif chain A · MSA", or null without a target. */
export function targetSummary(ttp: Ttp): string | null {
  const target = ttp.target;
  const protein = target?.protein_sequence
    ? `${target.protein_sequence.length.toLocaleString("en")} aa`
    : target?.structure_path
      ? [target.structure_path.split("/").pop(), target.structure_chain && `chain ${target.structure_chain}`].filter(Boolean).join(" ")
      : null;
  if (!protein) return null;
  return [protein, target?.pdb_id && `PDB ${target.pdb_id}`, target?.msa_path && "MSA", target?.templates?.length && "template"].filter(Boolean).join(" · ");
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
  /** The canonical sequence from UniProt, which residue numbers refer to; `sequence` may be a construct. */
  uniprotSequence: string | null;
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
  uniprotSequence: null,
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
      uniprotSequence: sequence ?? facts.uniprotSequence,
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
