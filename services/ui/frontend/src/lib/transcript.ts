// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { AgentTurn, ConversationEvent, Turn, TurnBlock } from "../types/api";

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
        return { ...next, busy: true, turns: [...next.turns, newAgentTurn(event.seq, at)] };
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
      return withBlock(next, at, event.seq, {
        kind: "text",
        id: `ma-${event.seq}`,
        content: `**Monitor alert**\n\n\`\`\`json\n${JSON.stringify(event.alert, null, 2)}\n\`\`\``,
      });

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

function newAgentTurn(seq: number, at: number): AgentTurn {
  return { role: "agent", id: `turn-${seq}`, at, blocks: [], done: false };
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

/** The tool steps of a turn, in order. */
export function toolSteps(turn: AgentTurn): Extract<TurnBlock, { kind: "tool" }>[] {
  return turn.blocks.filter((block): block is Extract<TurnBlock, { kind: "tool" }> => block.kind === "tool");
}
