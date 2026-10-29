// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import type { AgentTurn, ConversationEvent } from "../types/api";
import { applyEvents, EMPTY_TRANSCRIPT, stepError, stepState, toolSteps, turnTrace } from "./transcript";

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

  it("traces reasoning, tool calls and errors in the order they happened", () => {
    const state = applyEvents(
      EMPTY_TRANSCRIPT,
      turn(
        { type: "thinking", content: "Plan" },
        { type: "tool_call", id: "c1", name: "list_jobs", arguments: {} },
        { type: "thinking", content: "Next" },
        { type: "error", message: "Boltz timed out" },
        result("c1", "list_jobs"),
        { type: "assistant_message", content: "Done.", inputTokens: 1, outputTokens: 1 },
      ),
    );
    const trace = turnTrace(state.turns[1] as AgentTurn);
    expect(trace.map((entry) => entry.kind)).toEqual(["thinking", "tool", "thinking", "error"]);
    expect(trace[1]).toMatchObject({ kind: "tool", result: { name: "list_jobs" } });
  });

  it("keeps the campaign log out of the chat, inside a turn and between turns", () => {
    const log = (next: string) => ({ goal: "Beat JQ1", findings: [{ ts: ts(1), text: "Methyl helps" }], next, updated: ts(1) });
    const plain = turn(
      { type: "tool_call", id: "c1", name: "score_molecules", arguments: {} },
      result("c1", "score_molecules"),
      { type: "assistant_message", content: "Done.", inputTokens: 1, outputTokens: 1 },
      { type: "status", state: "idle" },
    );
    seq = 0;
    const logged = events(
      { type: "user_message", content: "Score JQ1" },
      { type: "status", state: "busy" },
      { type: "tool_call", id: "c1", name: "score_molecules", arguments: {} },
      { type: "campaign", campaign: log("Dock it") },
      result("c1", "score_molecules"),
      { type: "assistant_message", content: "Done.", inputTokens: 1, outputTokens: 1 },
      { type: "status", state: "idle" },
      { type: "campaign", campaign: log("Cofold it") },
    );

    const without = applyEvents(EMPTY_TRANSCRIPT, plain);
    const withLog = applyEvents(EMPTY_TRANSCRIPT, logged);
    const ids = (state: typeof without) => state.turns.map((t) => (t.role === "agent" ? t.blocks.map((b) => b.kind) : t.role));
    expect(ids(withLog)).toEqual(ids(without));
    expect(withLog.turns).toHaveLength(2);
    expect((withLog.turns[1] as AgentTurn).done).toBe(true);
    expect(toolSteps(withLog.turns[1] as AgentTurn)[0].result).toBeDefined();
    expect(withLog.busy).toBe(false);
    expect(withLog.lastSeq).toBe(8);
    expect(withLog.campaign?.next).toBe("Cofold it");
    expect(without.campaign).toBeNull();
  });

  it("does not open a turn for a campaign event on its own", () => {
    seq = 0;
    const state = applyEvents(EMPTY_TRANSCRIPT, events({ type: "campaign", campaign: { goal: "x", findings: [], next: null, updated: null } }));
    expect(state.turns).toEqual([]);
    expect(state.campaign?.goal).toBe("x");
  });

  it("ignores an event type it does not know", () => {
    seq = 0;
    const state = applyEvents(EMPTY_TRANSCRIPT, events({ type: "something_new" } as unknown as Unnumbered));
    expect(state.turns).toEqual([]);
    expect(state.lastSeq).toBe(1);
  });

  it("shows a monitor alert outside a turn as its own finished block", () => {
    seq = 0;
    const state = applyEvents(EMPTY_TRANSCRIPT, events({ type: "monitor_alert", alert: { job: "ab12" } }));
    const agent = state.turns[0] as AgentTurn;
    expect(agent.done).toBe(true);
    expect(agent.blocks[0].kind).toBe("alert");
  });
});

describe("step failures", () => {
  const step = (extra?: object) => {
    const state = applyEvents(
      EMPTY_TRANSCRIPT,
      turn(
        { type: "tool_call", id: "c1", name: "run_optimization", arguments: {} },
        ...(extra ? [result("c1", "run_optimization", extra)] : []),
      ),
    );
    return toolSteps(state.turns[1] as AgentTurn)[0];
  };

  it("tells running, done and failed apart", () => {
    expect(stepState(step())).toBe("running");
    expect(stepError(step())).toBeNull();
    expect(stepState(step({ result: '{"job_id": "ab12"}' }))).toBe("ok");
    expect(stepState(step({ result: "Tool crashed\nTraceback ...", isError: true }))).toBe("failed");
  });

  it("gives the first line of a failed result", () => {
    expect(stepError(step({ result: "\nTool crashed\nTraceback ...", isError: true }))).toBe("Tool crashed");
    expect(stepError(step({ result: "", isError: true }))).toBe("Failed");
  });

  it("counts an error payload as a failure even when the call itself succeeded", () => {
    const failed = step({ result: JSON.stringify({ error: "TTP 'x' not found", job_id: null }) });
    expect(stepState(failed)).toBe("failed");
    expect(stepError(failed)).toBe("TTP 'x' not found");
    expect(stepError(step({ result: JSON.stringify({ error: { message: "bad config" } }) }))).toBe("bad config");
    expect(stepError(step({ result: JSON.stringify({ ok: false, detail: "queue full" }) }))).toBe("queue full");
    expect(stepState(step({ result: JSON.stringify({ error: null, ok: true }) }))).toBe("ok");
  });

  it("reads the error from output cut for display", () => {
    const cut = `{"error": "config \\"x\\" invalid", "log": "${"a".repeat(50)}`;
    expect(stepError(step({ result: cut, truncated: true }))).toBe('config "x" invalid');
  });

  it("keeps the message short", () => {
    const message = stepError(step({ result: "x".repeat(500), isError: true }));
    expect(message?.length).toBe(200);
    expect(message?.endsWith("…")).toBe(true);
  });
});
