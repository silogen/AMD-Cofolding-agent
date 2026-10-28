// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import type { TurnBlock } from "../types/api";
import {
  clock,
  COFOLD_POLL_MS,
  COFOLD_PROBE_MS,
  cofoldCall,
  cofoldPollDelay,
  cofoldShown,
  cofoldView,
  EMPTY_MEMORY,
  inFlightCofold,
  learn,
  ownRequests,
  ownStatus,
  runUsesBoltz,
  ttpUsesBoltz,
  type CofoldCall,
  type CofoldStatus,
} from "./cofold";

type ToolStep = Extract<TurnBlock, { kind: "tool" }>;

function step(name: string, args: Record<string, unknown>, at: number, result?: string): ToolStep {
  return {
    kind: "tool",
    id: `c${at}`,
    name,
    arguments: args,
    at,
    result:
      result === undefined
        ? undefined
        : { type: "tool_result", seq: at, ts: "", id: `c${at}`, name, result, truncated: false, isError: false, panels: [], elapsedMs: 1 },
  };
}

const boltzTtp = step("create_ttp", { name: "BRD4", objectives: [{ name: "Binding", type: "external", service: "boltz" }] }, 1, '{"ttp_id": "ttp-boltz"}');
const qedTtp = step("create_ttp", { name: "QED only", objectives: [{ name: "QED", type: "native" }] }, 2, '{"ttp_id": "ttp-qed"}');

function status(over: Partial<Extract<CofoldStatus, { available: true }>> = {}): CofoldStatus {
  return {
    available: true,
    status: "ok",
    mode: "pool",
    workers: { total: 16, busy: 0, loading: 0, failed: 0 },
    gpus: { total: 8, busy: 0 },
    queueLength: 0,
    requests: [],
    active: false,
    ...over,
  };
}

const request = (over: Partial<Extract<CofoldStatus, { available: true }>["requests"][number]> = {}) => ({
  id: "r1",
  molecules: 2,
  done: 0,
  queued: 0,
  running: 2,
  state: "running" as const,
  elapsedSeconds: 40,
  ...over,
});

describe("ttpUsesBoltz", () => {
  it("reads the TTP from the step that created it", () => {
    expect(ttpUsesBoltz([boltzTtp, qedTtp], "ttp-boltz")).toBe(true);
    expect(ttpUsesBoltz([boltzTtp, qedTtp], "ttp-qed")).toBe(false);
  });

  it("is unknown for a TTP the conversation never showed", () => {
    expect(ttpUsesBoltz([boltzTtp], "ttp-other")).toBeNull();
    expect(ttpUsesBoltz([boltzTtp], null)).toBeNull();
  });

  it("uses the newest step that shows the TTP", () => {
    const updated = step("update_ttp", { ttp_id: "ttp-boltz", objectives: [{ name: "QED", type: "native" }] }, 5, '{"ttp_id": "ttp-boltz", "version": 2}');
    expect(ttpUsesBoltz([boltzTtp, updated], "ttp-boltz")).toBe(false);
  });
});

describe("cofoldCall", () => {
  it("takes score_molecules with a Boltz TTP as a cofold", () => {
    const call = step("score_molecules", { smiles: ["CCO", "CCN"], ttp_id: "ttp-boltz" }, 10);
    expect(cofoldCall(call, [boltzTtp, call])).toEqual({ id: "c10", molecules: 2, at: 10, certain: true });
  });

  it("ignores scoring without Boltz and other tools", () => {
    expect(cofoldCall(step("score_molecules", { smiles: ["CCO"], ttp_id: "ttp-qed" }, 10), [qedTtp])).toBeNull();
    expect(cofoldCall(step("score_molecules", { smiles: ["CCO"], components: [{ type: "QED" }] }, 10), [])).toBeNull();
    expect(cofoldCall(step("predict_admet", { smiles: ["CCO"] }, 10), [])).toBeNull();
  });

  it("reads an inline TTP", () => {
    const call = step("score_molecules", { smiles: ["CCO"], ttp: { scoring: { objectives: [{ service: "boltz" }] } } }, 10);
    expect(cofoldCall(call, [])?.certain).toBe(true);
  });

  it("marks a TTP it cannot see as a possible cofold", () => {
    expect(cofoldCall(step("score_molecules", { smiles: ["CCO"], ttp_id: "ttp-x" }, 10), [])).toMatchObject({ certain: false, molecules: 1 });
  });
});

describe("inFlightCofold", () => {
  const open = step("score_molecules", { smiles: ["CCO", "CCN"], ttp_id: "ttp-boltz" }, 20);
  const answered = step("score_molecules", { smiles: ["CCO"], ttp_id: "ttp-boltz" }, 15, "{}");
  const stale = step("score_molecules", { smiles: ["CCO"], ttp_id: "ttp-boltz" }, 5);

  it("keeps the unanswered calls of the turn in progress", () => {
    expect(inFlightCofold([open, answered, stale, boltzTtp], 12).map((c) => c.id)).toEqual(["c20"]);
  });

  it("is empty without a turn in progress", () => {
    expect(inFlightCofold([open], null)).toEqual([]);
  });
});

describe("runUsesBoltz", () => {
  it("prefers the TTP, then the run's component names", () => {
    expect(runUsesBoltz([boltzTtp], "ttp-boltz", [{ name: "QED" }])).toBe(true);
    expect(runUsesBoltz([], "ttp-x", [{ name: "Boltz affinity" }])).toBe(true);
    expect(runUsesBoltz([], "ttp-x", [{ name: "QED" }])).toBe(false);
    expect(runUsesBoltz([], "ttp-x", undefined)).toBeNull();
  });
});

describe("cofoldPollDelay", () => {
  const sure: CofoldCall = { id: "c", molecules: 1, at: 0, certain: true };
  const base = { visible: true, calls: [] as CofoldCall[], boltzRun: false as boolean | null, running: false, lastActive: false };

  it("does not poll when nothing cofolds", () => {
    expect(cofoldPollDelay(base)).toBeNull();
    expect(cofoldPollDelay({ ...base, running: true, boltzRun: false })).toBeNull();
  });

  it("polls fast for a call, a Boltz run or an active service", () => {
    expect(cofoldPollDelay({ ...base, calls: [sure] })).toBe(COFOLD_POLL_MS);
    expect(cofoldPollDelay({ ...base, running: true, boltzRun: true })).toBe(COFOLD_POLL_MS);
    expect(cofoldPollDelay({ ...base, lastActive: true })).toBe(COFOLD_POLL_MS);
  });

  it("only probes a run that may score with Boltz", () => {
    expect(cofoldPollDelay({ ...base, running: true, boltzRun: null })).toBe(COFOLD_PROBE_MS);
  });

  it("never polls a hidden page", () => {
    expect(cofoldPollDelay({ ...base, visible: false, calls: [sure], lastActive: true })).toBeNull();
  });
});

describe("clock", () => {
  it("formats minutes and hours", () => {
    expect(clock(41_000)).toBe("0:41");
    expect(clock(725_400)).toBe("12:05");
    expect(clock(3_725_000)).toBe("1:02:05");
    expect(clock(-5)).toBe("0:00");
  });
});

describe("cofoldView", () => {
  const sure: CofoldCall = { id: "c", molecules: 2, at: 1_000, certain: true };

  it("shows a call at once, before the service answers", () => {
    expect(cofoldView(null, 0, [sure], 6_000)).toMatchObject({ label: "Cofolding", detail: "2 molecules", done: 0, total: 2, elapsedMs: 5_000 });
    expect(cofoldShown(null, [sure])).toBe(true);
  });

  it("shows nothing for a possible cofold until the service confirms it", () => {
    const maybe = { ...sure, certain: false };
    expect(cofoldView(null, 0, [maybe], 6_000)).toBeNull();
    expect(cofoldView(status(), 0, [maybe], 6_000)).toBeNull();
    expect(cofoldShown(status(), [maybe])).toBe(false);
  });

  it("reports one request with its progress and a running clock", () => {
    const view = cofoldView(status({ active: true, workers: { total: 16, busy: 2, loading: 0, failed: 0 }, gpus: { total: 8, busy: 1 }, requests: [request({ done: 1, running: 1 })] }), 10_000, [sure], 11_000);
    expect(view).toMatchObject({ label: "Cofolding", detail: "2 molecules", done: 1, total: 2, note: "", elapsedMs: 41_000 });
  });

  it("summarises several requests", () => {
    const view = cofoldView(
      status({
        active: true,
        workers: { total: 16, busy: 3, loading: 0, failed: 0 },
        gpus: { total: 8, busy: 3 },
        requests: [request({ molecules: 3, done: 1, queued: 1, running: 1 }), request({ id: "r2", molecules: 1, running: 1, elapsedSeconds: 5 })],
      }),
      0,
      [],
      0,
    );
    expect(view).toMatchObject({ detail: "4 molecules on 3 GPUs", note: "1 queued", done: 1, total: 4, elapsedMs: 40_000 });
    expect(view?.title).toBe("running: 1/3 done, 1 queued\nrunning: 0/1 done");
  });

  it("says when molecules wait for Boltz to load or for the MSA", () => {
    const loading = status({ active: true, status: "loading", workers: { total: 16, busy: 0, loading: 16, failed: 0 }, requests: [request({ queued: 2, running: 0, state: "queued" })] });
    expect(cofoldView(loading, 0, [], 0)?.note).toBe("Boltz is loading");
    const preparing = status({ active: true, requests: [request({ running: 0, state: "preparing" })] });
    expect(cofoldView(preparing, 0, [], 0)?.label).toBe("Preparing");
  });

  it("falls back to busy workers for a service without request tracking", () => {
    const view = cofoldView(status({ active: true, workers: { total: 16, busy: 2, loading: 0, failed: 0 }, gpus: { total: 8, busy: 2 }, queueLength: 3 }), 0, [], 0);
    expect(view).toMatchObject({ detail: "2 GPUs busy", note: "3 queued", total: 0, elapsedMs: null });
  });

  it("shows a call while the service is unreachable", () => {
    expect(cofoldView({ available: false }, 0, [sure], 1_000)?.detail).toBe("2 molecules");
  });

  it("counts running molecules for how far they likely got, and estimates the time left", () => {
    // 10 molecules, 6 on workers for 40 s, 4 queued; a molecule takes about 80 s.
    const busy = status({
      active: true,
      moleculeSeconds: 80,
      requests: [request({ molecules: 10, queued: 4, running: 6, elapsedSeconds: 40, runningSeconds: [40, 40, 40, 40, 40, 40] })],
    });
    const now = cofoldView(busy, 0, [], 0)!;
    expect(now.done).toBe(0);
    expect(now.fraction).toBeCloseTo(0.3);
    expect(now.remainingMs).toBe((40 + 80) * 1000);
    // Ten seconds later, without a new answer, the bar has moved on.
    const later = cofoldView(busy, 0, [], 10_000)!;
    expect(later.fraction).toBeCloseTo(0.375);
    expect(later.elapsedMs).toBe(50_000);
  });

  it("never counts a running molecule as done", () => {
    const slow = status({ active: true, moleculeSeconds: 10, requests: [request({ molecules: 1, running: 1, runningSeconds: [500] })] });
    expect(cofoldView(slow, 0, [], 0)?.fraction).toBeCloseTo(0.9);
  });

  it("shows a finished batch full and still", () => {
    const busy = status({ active: true, moleculeSeconds: 80, requests: [request({ molecules: 10, done: 8, running: 2, runningSeconds: [5, 5] })] });
    expect(cofoldView(busy, 0, [], 60_000, { finished: true })).toMatchObject({ done: 10, fraction: 1, note: "", remainingMs: null, elapsedMs: 40_000 });
  });
});

describe("ownRequests", () => {
  const call: CofoldCall = { id: "c", molecules: 2, at: 1_000_000, certain: true };
  const r = (id: string, molecules: number, startedAt: number | null) => request({ id, molecules, startedAt });

  it("pairs a call with the first request that started after it", () => {
    const requests = [r("before", 2, 900), r("run", 10, 1_001), r("mine", 2, 1_002), r("later", 2, 1_003)];
    expect(ownRequests(requests, [call], false).map((x) => x.id)).toEqual(["mine"]);
  });

  it("gives a run of the conversation every request", () => {
    expect(ownRequests([r("a", 10, 1), r("b", 2, 2)], [], true).map((x) => x.id)).toEqual(["a", "b"]);
  });

  it("claims nothing for another conversation", () => {
    expect(ownRequests([r("a", 10, 1_001)], [], false)).toEqual([]);
    expect(ownStatus(status({ active: true, requests: [r("a", 10, 1_001)] }), [], false)).toMatchObject({ active: false, requests: [] });
  });

  it("pairs by size alone when the gateway does not say when a request started", () => {
    expect(ownRequests([r("big", 10, null), r("small", 2, null)], [call], false).map((x) => x.id)).toEqual(["small"]);
  });
});

describe("learn", () => {
  const read = (done: number, elapsed: number, running = 6) =>
    status({ active: true, requests: [request({ molecules: 10, done, running, queued: 10 - done - running, elapsedSeconds: elapsed })] });

  it("learns the molecule time from when molecules finish, for services that do not report it", () => {
    let memory = learn(EMPTY_MEMORY, read(0, 20));
    expect(memory.typicalSeconds).toBeNull();
    memory = learn(memory, read(6, 85, 4));
    expect(memory.typicalSeconds).toBe(85);
    expect(memory.requests.r1).toEqual({ done: 6, at: 85 });
    // The next wave started when the first finished.
    const view = cofoldView(read(6, 125, 4), 0, [], 0, { memory })!;
    expect(view.fraction).toBeCloseTo((6 + 4 * (40 / 85)) / 10);
  });

  it("prefers the service's molecule time and forgets finished requests", () => {
    const memory = learn({ typicalSeconds: 10, requests: { gone: { done: 1, at: 5 } } }, status({ active: true, moleculeSeconds: 70, requests: [] }));
    expect(memory).toEqual({ typicalSeconds: 70, requests: {} });
  });
});
