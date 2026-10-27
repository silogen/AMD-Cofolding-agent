// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * The contract with the agent gateway.
 *
 * Server side: reinvent4_agent/web/server.py (endpoints) and
 * reinvent4_agent/web/events.py (event shapes). Keep the two in step.
 */

// ---------------------------------------------------------------------------
// panels — the four kinds of data one prompt can produce
// ---------------------------------------------------------------------------

/** A protein-ligand complex written by Boltz, shown in the 3D viewer. */
export interface StructurePanel {
  kind: "structure";
  tool: string;
  /** File name, for the card heading. */
  name: string;
  /** Path relative to the shared volume; also the panel identity. */
  path: string;
  /** Where to fetch the file from. */
  url: string;
  format: "cif" | "pdb";
  /** The ligand in this complex, when the tool result says which it is. */
  smiles?: string;
  label?: string | null;
  metrics?: StructureMetrics;
}

/** Boltz and PoseBusters values of one predicted complex. */
export interface StructureMetrics {
  confidence?: number;
  ligandIptm?: number;
  iptm?: number;
  plddt?: number;
  /** log10(IC50 / 1 µM); lower binds tighter. */
  affinity?: number;
  affinity1?: number;
  affinity2?: number;
  binderProbability?: number;
  posebustersPassed?: number;
  posebustersTotal?: number;
  failedChecks?: string[];
}

/** One predicted endpoint: an ensemble mean, its spread and its unit. */
export interface PropertyValue {
  mean: number;
  std?: number;
  unit?: string;
  condition?: string;
}

/** ADMET or RetroDMTA predictions, endpoints by molecule. */
export interface PropertiesPanel {
  kind: "properties";
  tool: string;
  service: string | null;
  target: string | null;
  endpoints: string[];
  molecules: {
    smiles: string | null;
    standardized: string | null;
    label: string | null;
    ok: boolean;
    error: string | null;
    /** Inside the models' applicability domain. */
    adPass: boolean | null;
    values: Record<string, PropertyValue>;
  }[];
}

/** One protein-ligand contact from the interaction profiler. */
export interface Contact {
  type: string;
  subtype: string | null;
  chain: string | null;
  resname: string;
  resnum: number | null;
  /** "ASN140". */
  label: string;
  receptorAtom: string | null;
  ligandAtom: string | null;
  /** 0-based index into the ligand's atoms, -1 or null when not attributable. */
  ligandAtomIndex: number | null;
  distance: number | null;
  fragment: string | null;
  confidence: unknown;
}

/** The contacts of one complex. */
export interface InteractionsPanel {
  kind: "interactions";
  tool: string;
  name: string | null;
  /** The structure these contacts were measured on, as a structure panel path. */
  structurePath: string | null;
  smiles: string | null;
  error: string | null;
  counts: Record<string, number>;
  contacts: Contact[];
  totalContacts: number;
}

/** A ligand as SMILES, drawn as a 2D graph. */
export interface MoleculePanel {
  kind: "molecule";
  tool: string;
  smiles: string;
  label: string | null;
}

/** Predictions and measurements per molecule. */
export interface TablePanel {
  kind: "table";
  tool: string;
  columns: string[];
  rows: Record<string, string | number | boolean | null>[];
  totalRows: number;
  truncated: boolean;
}

export type Panel = StructurePanel | MoleculePanel | TablePanel | PropertiesPanel | InteractionsPanel;

// ---------------------------------------------------------------------------
// events — what the gateway pushes down the WebSocket
// ---------------------------------------------------------------------------

/**
 * Fields the gateway adds to every event it records in a session. Events
 * that concern one browser only (ready, session, notice) have neither.
 */
export interface Recorded {
  seq: number;
  /** ISO 8601, UTC. */
  ts: string;
}

/** Sent first on every connection; not part of the conversation. */
export interface ReadyEvent {
  type: "ready";
  model: string;
  backend: string;
  reasoningEffort: string | null;
  tools: string[];
}

/** Which conversation this socket is attached to. Sent after ready. */
export interface SessionEvent extends SessionSummary {
  type: "session";
  /** Events up to this number are replayed right after this one. */
  lastSeq: number;
}

/** A message for this browser only, such as a refused second question. */
export interface NoticeEvent {
  type: "notice";
  message: string;
}

export interface UserMessageEvent extends Recorded {
  type: "user_message";
  content: string;
  effort?: Effort;
}

export interface ThinkingEvent extends Recorded {
  type: "thinking";
  content: string;
}

export interface ToolCallEvent extends Recorded {
  type: "tool_call";
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

export interface ToolResultEvent extends Recorded {
  type: "tool_result";
  id: string | null;
  name: string;
  result: string;
  truncated: boolean;
  isError: boolean;
  panels: Panel[];
  elapsedMs: number | null;
}

export interface AssistantMessageEvent extends Recorded {
  type: "assistant_message";
  content: string;
  inputTokens: number;
  outputTokens: number;
}

export interface StatusEvent extends Recorded {
  type: "status";
  state: "busy" | "idle";
}

export interface CancelledEvent extends Recorded {
  type: "cancelled";
}

export interface ErrorEvent extends Recorded {
  type: "error";
  message: string;
}

export interface MonitorAlertEvent extends Recorded {
  type: "monitor_alert";
  alert: Record<string, unknown>;
}

export interface MonitorResponseEvent extends Recorded {
  type: "monitor_response";
  content: string;
}

/** Events that belong to the conversation and are replayed on reconnect. */
export type ConversationEvent =
  | UserMessageEvent
  | ThinkingEvent
  | ToolCallEvent
  | ToolResultEvent
  | AssistantMessageEvent
  | StatusEvent
  | CancelledEvent
  | ErrorEvent
  | MonitorAlertEvent
  | MonitorResponseEvent;

export type AgentEvent = ReadyEvent | SessionEvent | NoticeEvent | ConversationEvent;

/** How hard the model reasons on a turn; the gateway sets it on the session's backend. */
export type Effort = "low" | "medium" | "high";

/** What the browser sends. */
export type ClientMessage = { type: "user_message"; content: string; effort?: Effort } | { type: "cancel" };

// ---------------------------------------------------------------------------
// models — which model answers, and switching to the other one
// ---------------------------------------------------------------------------

/** One served model. A model without *efforts* does not reason in levels. */
export interface ModelOption {
  key: string;
  id: string;
  label: string;
  efforts: Effort[];
  defaultEffort: Effort | null;
}

/**
 * GET /api/models, and the answer to a POST to it. Only one model runs at a
 * time, so a switch restarts the server: the POST answers 202 with state
 * "loading" and the browser polls until it is ready or failed.
 */
export interface ModelsState {
  /** The key of the model now serving. */
  current: string;
  state: "ready" | "loading" | "failed";
  /** False where there is no cluster to restart; the model is then fixed. */
  switchable: boolean;
  models: ModelOption[];
}

// ---------------------------------------------------------------------------
// sessions — saved conversations
// ---------------------------------------------------------------------------

export interface SessionSummary {
  id: string;
  title: string;
  /** ISO 8601, UTC. */
  created: string;
  updated: string;
  turns: number;
  busy: boolean;
  inputTokens?: number;
  outputTokens?: number;
}

// ---------------------------------------------------------------------------
// transcript — what the UI keeps and renders
// ---------------------------------------------------------------------------

/**
 * Blocks are appended in the order the agent produced them, so the transcript
 * reads as a record of what was done when. That ordering is the whole point of
 * showing tool activity inline rather than in a separate report.
 */
export type TurnBlock =
  | { kind: "thinking"; id: string; content: string }
  | {
      kind: "tool";
      id: string;
      name: string;
      arguments: Record<string, unknown>;
      /** Epoch ms when the call started. */
      at: number;
      result?: ToolResultEvent;
    }
  | { kind: "text"; id: string; content: string }
  | { kind: "error"; id: string; message: string }
  | { kind: "cancelled"; id: string }
  | { kind: "alert"; id: string; alert: Record<string, unknown> };

export interface UserTurn {
  role: "user";
  id: string;
  content: string;
  at: number;
}

export interface AgentTurn {
  role: "agent";
  id: string;
  /** Epoch ms when the turn started. */
  at: number;
  /** Epoch ms when it ended; unset while running. */
  endedAt?: number;
  blocks: TurnBlock[];
  /** Set once the turn finishes; drives the "working" indicator. */
  done: boolean;
  tokens?: { input: number; output: number };
}

export type Turn = UserTurn | AgentTurn;

export type ConnectionState = "connecting" | "open" | "closed" | "error";

export interface SessionInfo {
  backend: string;
  model: string;
  endpoint: string;
  tools: string[];
  ready: boolean;
}
