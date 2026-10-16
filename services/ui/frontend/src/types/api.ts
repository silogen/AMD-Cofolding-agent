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

export type Panel = StructurePanel | MoleculePanel | TablePanel;

// ---------------------------------------------------------------------------
// events — what the gateway pushes down the WebSocket
// ---------------------------------------------------------------------------

export interface ReadyEvent {
  type: "ready";
  model: string;
  backend: string;
  tools: string[];
}

export interface ThinkingEvent {
  type: "thinking";
  content: string;
}

export interface ToolCallEvent {
  type: "tool_call";
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

export interface ToolResultEvent {
  type: "tool_result";
  id: string | null;
  name: string;
  result: string;
  truncated: boolean;
  isError: boolean;
  panels: Panel[];
}

export interface AssistantMessageEvent {
  type: "assistant_message";
  content: string;
  inputTokens: number;
  outputTokens: number;
}

export interface StatusEvent {
  type: "status";
  state: "busy" | "idle";
}

export interface ErrorEvent {
  type: "error";
  message: string;
}

export interface MonitorAlertEvent {
  type: "monitor_alert";
  alert: Record<string, unknown>;
}

export interface MonitorResponseEvent {
  type: "monitor_response";
  content: string;
}

export type AgentEvent =
  | ReadyEvent
  | ThinkingEvent
  | ToolCallEvent
  | ToolResultEvent
  | AssistantMessageEvent
  | StatusEvent
  | ErrorEvent
  | MonitorAlertEvent
  | MonitorResponseEvent;

/** The only message the browser sends. */
export interface UserMessage {
  type: "user_message";
  content: string;
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
  | { kind: "tool"; id: string; name: string; arguments: Record<string, unknown>; result?: ToolResultEvent }
  | { kind: "text"; id: string; content: string }
  | { kind: "error"; id: string; message: string };

export interface UserTurn {
  role: "user";
  id: string;
  content: string;
  at: number;
}

export interface AgentTurn {
  role: "agent";
  id: string;
  at: number;
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
