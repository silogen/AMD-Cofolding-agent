// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { AgentTurn, ConversationEvent, ToolResultEvent, Turn, TurnBlock } from "../types/api";

/**
 * The conversation as the UI renders it, built from the gateway's events.
 *
 * The same function handles a live event and a replayed one, which is what
 * makes a reloaded page look exactly like the page that was closed.
 */
export interface Transcript {
  turns: Turn[];
  busy: boolean;
  /** Highest event number applied; sent back on reconnect. */
  lastSeq: number;
}

export const EMPTY_TRANSCRIPT: Transcript = { turns: [], busy: false, lastSeq: 0 };

const time = (ts: string | undefined): number => (ts ? Date.parse(ts) : Date.now());

/** Apply one event. Events already applied (by number) are ignored. */
export function applyEvent(state: Transcript, event: ConversationEvent): Transcript {
  if (event.seq <= state.lastSeq) return state;
  const next = { ...state, lastSeq: event.seq };
  const at = time(event.ts);

  switch (event.type) {
    case "user_message":
      return {
        ...next,
        turns: [...closeOpenTurn(next.turns, at), { role: "user", id: `u-${event.seq}`, content: event.content, at }],
      };

    case "status":
      if (event.state === "busy") {
        return { ...next, busy: true, turns: [...next.turns, newAgentTurn(event.seq, at, event.auto === true)] };
      }
      return { ...next, busy: false, turns: closeOpenTurn(next.turns, at) };

    case "thinking":
      return withBlock(next, at, event.seq, { kind: "thinking", id: `t-${event.seq}`, content: event.content });

    case "tool_call":
      return withBlock(next, at, event.seq, {
        kind: "tool",
        id: event.id,
        name: event.name,
        arguments: event.arguments,
        at,
      });

    case "tool_result":
      return updateOpenTurn(next, at, event.seq, (turn) => {
        const index = findOpenToolCall(turn.blocks, event.id, event.name);
        if (index < 0) return turn;
        const blocks = [...turn.blocks];
        blocks[index] = { ...(blocks[index] as Extract<TurnBlock, { kind: "tool" }>), result: event };
        return { ...turn, blocks };
      });

    case "assistant_message":
      return updateOpenTurn(next, at, event.seq, (turn) => ({
        ...turn,
        blocks: [...turn.blocks, { kind: "text", id: `a-${event.seq}`, content: event.content }],
        tokens: { input: event.inputTokens, output: event.outputTokens },
      }));

    case "monitor_response":
      return withBlock(next, at, event.seq, { kind: "text", id: `m-${event.seq}`, content: event.content });

    case "monitor_alert":
      return withBlock(next, at, event.seq, { kind: "alert", id: `ma-${event.seq}`, alert: event.alert });

    case "cancelled":
      return withBlock(next, at, event.seq, { kind: "cancelled", id: `c-${event.seq}` });

    case "error":
      // Not the end of the turn: errors also arrive while the agent keeps
      // working. Only "status: idle" ends a turn.
      return withBlock(next, at, event.seq, { kind: "error", id: `e-${event.seq}`, message: event.message });
  }
}

export function applyEvents(state: Transcript, events: ConversationEvent[]): Transcript {
  return events.reduce(applyEvent, state);
}

function newAgentTurn(seq: number, at: number, auto = false): AgentTurn {
  return { role: "agent", id: `turn-${seq}`, at, blocks: [], done: false, ...(auto ? { auto: true } : {}) };
}

function closeOpenTurn(turns: Turn[], at: number): Turn[] {
  const last = turns[turns.length - 1];
  if (last && last.role === "agent" && !last.done) {
    return [...turns.slice(0, -1), { ...last, done: true, endedAt: at }];
  }
  return turns;
}

/** Change the open agent turn, opening one when none is (an event outside a turn). */
function updateOpenTurn(state: Transcript, at: number, seq: number, change: (turn: AgentTurn) => AgentTurn): Transcript {
  const last = state.turns[state.turns.length - 1];
  if (last && last.role === "agent" && !last.done) {
    return { ...state, turns: [...state.turns.slice(0, -1), change(last)] };
  }
  const opened = change(newAgentTurn(seq, at));
  // Outside a turn there is no "idle" to come, so the block stands alone.
  return { ...state, turns: [...state.turns, { ...opened, done: !state.busy, endedAt: state.busy ? undefined : at }] };
}

function withBlock(state: Transcript, at: number, seq: number, block: TurnBlock): Transcript {
  return updateOpenTurn(state, at, seq, (turn) => ({ ...turn, blocks: [...turn.blocks, block] }));
}

/**
 * Find the tool block a result belongs to: by id, or failing that the last
 * unanswered call of the same name.
 */
function findOpenToolCall(blocks: TurnBlock[], id: string | null, name: string): number {
  if (id) {
    const byId = blocks.findIndex((block) => block.kind === "tool" && block.id === id);
    if (byId >= 0) return byId;
  }
  for (let index = blocks.length - 1; index >= 0; index -= 1) {
    const block = blocks[index];
    if (block.kind === "tool" && block.name === name && !block.result) return index;
  }
  return -1;
}

type ToolStep = Extract<TurnBlock, { kind: "tool" }>;

/** The tool steps of a turn, in order. */
export function toolSteps(turn: AgentTurn): ToolStep[] {
  return turn.blocks.filter((block): block is ToolStep => block.kind === "tool");
}

/** What the agent did, in order: its reasoning, its tool calls and the errors it hit. */
export type TraceEntry = Extract<TurnBlock, { kind: "thinking" | "tool" | "error" }>;

export function turnTrace(turn: AgentTurn): TraceEntry[] {
  return turn.blocks.filter((block): block is TraceEntry => block.kind === "thinking" || block.kind === "tool" || block.kind === "error");
}

export type StepState = "running" | "ok" | "failed";

export function stepState(step: ToolStep): StepState {
  if (!step.result) return "running";
  return stepError(step) === null ? "ok" : "failed";
}

const ERROR_CHARS = 200;
const errors = new WeakMap<ToolResultEvent, string | null>();

/**
 * Why a tool call failed, in one short line, or null when it did not.
 * Tools also fail by answering normally with an `{"error": ...}` payload
 * (run_optimization does), so those count too. Results never change once
 * received, so the answer is kept per result.
 */
export function stepError(step: ToolStep): string | null {
  const result = step.result;
  if (!result) return null;
  if (!errors.has(result)) errors.set(result, resultError(result));
  return errors.get(result) ?? null;
}

function resultError(result: ToolResultEvent): string | null {
  const text = result.result ?? "";
  const reported = payloadMessage(text);
  if (reported === null && !result.isError) return null;
  return short(reported ?? text) || "Failed";
}

/** The error a JSON payload reports, also from output cut for display. */
function payloadMessage(text: string): string | null {
  const start = text.trimStart();
  if (!start.startsWith("{")) return null;
  try {
    const parsed: unknown = JSON.parse(start);
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return null;
    const payload = parsed as Record<string, unknown>;
    const error = payload.error;
    if (typeof error === "string" && error.trim()) return error;
    if (typeof error === "object" && error !== null && typeof (error as { message?: unknown }).message === "string") {
      return (error as { message: string }).message;
    }
    if (payload.ok === false || payload.success === false) {
      const message = payload.message ?? payload.detail;
      return typeof message === "string" && message.trim() ? message : "Failed";
    }
    return null;
  } catch {
    const match = /"error"\s*:\s*"((?:[^"\\]|\\.)*)/.exec(start);
    return match ? match[1].replace(/\\n/g, "\n").replace(/\\(.)/g, "$1") : null;
  }
}

function short(text: string): string {
  const line = text.split("\n").find((part) => part.trim()) ?? "";
  const trimmed = line.trim();
  return trimmed.length > ERROR_CHARS ? `${trimmed.slice(0, ERROR_CHARS - 1)}…` : trimmed;
}
