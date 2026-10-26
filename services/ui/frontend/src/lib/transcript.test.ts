// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import type { AgentTurn, ConversationEvent } from "../types/api";
import { applyEvents, EMPTY_TRANSCRIPT, toolSteps } from "./transcript";

type Unnumbered = ConversationEvent extends infer E ? (E extends ConversationEvent ? Omit<E, "seq" | "ts"> : never) : never;

let seq = 0;
const ts = (s: number) => new Date(Date.UTC(2026, 9, 26, 12, 0, s)).toISOString();

/** Number events in order, the way the gateway does. */
function events(...partial: Unnumbered[]): ConversationEvent[] {
  return partial.map((event) => {
    seq += 1;
    return { ...event, seq, ts: ts(seq) } as ConversationEvent;
  });
}

const turn = (...partial: Unnumbered[]) => {
  seq = 0;
  return events(
    { type: "user_message", content: "Score JQ1" },
    { type: "status", state: "busy" },
    ...partial,
  );
};

const result = (id: string, name: string, extra: object = {}) => ({
  type: "tool_result" as const,
  id,
  name,
  result: "{}",
  truncated: false,
  isError: false,
  panels: [],
  elapsedMs: 1200,
  ...extra,
});

describe("applyEvents", () => {
  it("builds a question and a finished answer", () => {
    const state = applyEvents(
      EMPTY_TRANSCRIPT,
      turn(
        { type: "thinking", content: "Need Boltz." },
        { type: "tool_call", id: "c1", name: "score_molecules", arguments: { smiles: ["C"] } },
        result("c1", "score_molecules"),
        { type: "assistant_message", content: "Done.", inputTokens: 10, outputTokens: 5 },
        { type: "status", state: "idle" },
      ),
    );

    expect(state.busy).toBe(false);
    expect(state.lastSeq).toBe(7);
    expect(state.turns.map((t) => t.role)).toEqual(["user", "agent"]);
    const agent = state.turns[1] as AgentTurn;
    expect(agent.done).toBe(true);
    expect(agent.endedAt).toBe(Date.parse(ts(7)));
    expect(agent.tokens).toEqual({ input: 10, output: 5 });
    expect(agent.blocks.map((b) => b.kind)).toEqual(["thinking", "tool", "text"]);
    expect(toolSteps(agent)[0].result?.elapsedMs).toBe(1200);
  });

  it("keeps a turn open until idle, also after an error", () => {
    const state = applyEvents(EMPTY_TRANSCRIPT, turn({ type: "error", message: "Boltz timed out" }));
    expect(state.busy).toBe(true);
    const agent = state.turns[1] as AgentTurn;
    expect(agent.done).toBe(false);
    expect(agent.blocks[0]).toMatchObject({ kind: "error", message: "Boltz timed out" });
  });

  it("ignores events it has already applied, so a replay is harmless", () => {
    const all = turn(
      { type: "tool_call", id: "c1", name: "list_jobs", arguments: {} },
      result("c1", "list_jobs"),
      { type: "status", state: "idle" },
    );
    const once = applyEvents(EMPTY_TRANSCRIPT, all);
    const twice = applyEvents(once, all);
    expect(twice).toBe(once);

    const resumed = applyEvents(applyEvents(EMPTY_TRANSCRIPT, all.slice(0, 3)), all);
    expect(resumed).toEqual(once);
  });

  it("matches a result to its call by name when the id is missing", () => {
    const state = applyEvents(
      EMPTY_TRANSCRIPT,
      turn(
        { type: "tool_call", id: "c1", name: "get_ttp", arguments: {} },
        { type: "tool_call", id: "c2", name: "list_jobs", arguments: {} },
        result("c2", "list_jobs"),
        { ...result("x", "get_ttp"), id: null },
      ),
    );
    const steps = toolSteps(state.turns[1] as AgentTurn);
    expect(steps.map((step) => Boolean(step.result))).toEqual([true, true]);
  });

  it("records a cancelled turn", () => {
    const state = applyEvents(EMPTY_TRANSCRIPT, turn({ type: "cancelled" }, { type: "status", state: "idle" }));
    const agent = state.turns[1] as AgentTurn;
    expect(agent.done).toBe(true);
    expect(agent.blocks).toEqual([{ kind: "cancelled", id: "c-3" }]);
  });

  it("starts a new agent turn for every question", () => {
    const first = turn({ type: "status", state: "idle" });
    const second = events(
      { type: "user_message", content: "Again" },
      { type: "status", state: "busy" },
      { type: "assistant_message", content: "Ok", inputTokens: 1, outputTokens: 1 },
      { type: "status", state: "idle" },
    );
    const state = applyEvents(EMPTY_TRANSCRIPT, [...first, ...second]);
    expect(state.turns.map((t) => t.role)).toEqual(["user", "agent", "user", "agent"]);
    expect(state.turns.every((t) => t.role === "user" || t.done)).toBe(true);
  });

  it("shows a monitor alert outside a turn as its own finished block", () => {
    seq = 0;
    const state = applyEvents(EMPTY_TRANSCRIPT, events({ type: "monitor_alert", alert: { job: "ab12" } }));
    const agent = state.turns[0] as AgentTurn;
    expect(agent.done).toBe(true);
    expect(agent.blocks[0].kind).toBe("alert");
  });
});
