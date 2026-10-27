// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import {
  conversationJobIds,
  countDown,
  etaLabel,
  parseStepTimes,
  runProgress,
  runEventText,
  runsToShow,
  RunWatch,
  secondsPerStep,
  shortDuration,
} from "./runProgress";

describe("RunWatch", () => {
  it("announces a run that ends after being seen running, once", () => {
    const watch = new RunWatch();
    expect(watch.jobs([{ id: "a", state: "completed" }, { id: "b", state: "running" }, { id: "c", state: "queued" }])).toEqual([]);
    const events = watch.jobs([{ id: "a", state: "completed" }, { id: "b", state: "completed" }, { id: "c", state: "failed", error: "No GPU" }]);
    expect(events.map((e) => [e.jobId, e.kind, e.message])).toEqual([
      ["b", "completed", null],
      ["c", "failed", "No GPU"],
    ]);
    expect(watch.jobs([{ id: "b", state: "running" }])).toEqual([]);
    expect(watch.jobs([{ id: "b", state: "completed" }])).toEqual([]);
  });

  it("records the first alerts quietly and announces new ones", () => {
    const watch = new RunWatch();
    expect(watch.alertsOf([{ jobId: "a", message: "Plateau" }])).toEqual([]);
    expect(watch.alertsOf([{ jobId: "a", message: " plateau " }])).toEqual([]);
    const [event] = watch.alertsOf([{ jobId: "a", message: "Plateau" }, { jobId: "a", message: "Validity dropped to 40%" }]);
    expect(event).toMatchObject({ jobId: "a", kind: "alert", message: "Validity dropped to 40%" });
  });

  it("words each event", () => {
    expect(runEventText({ key: "k", jobId: "a", kind: "completed", message: null }, "sol v1")).toEqual({ title: "Design finished · sol v1", body: "Results are ready." });
    expect(runEventText({ key: "k", jobId: "a", kind: "failed", message: "boom" }, "sol v1").body).toBe("boom");
  });
});

const LOG = [
  "09:24:19 <INFO> Started REINVENT 4.8.24 (C) AstraZeneca 2017, 2023 on 2026-10-27",
  " | Agent Prior Target Score SMILES SMILES_state",
  "09:41:32 <INFO> Score: 0.82 Agent NLL: 14.44 Valid:  98% Step: 212",
  " | 12.9 47.0 80.9 0.99 CCO 1",
  "09:41:37 <INFO> Calling admet at http://localhost:8084/score with 60 SMILES",
  "09:41:37 <INFO> Score: 0.90 Agent NLL: 14.98 Valid:  97% Step: 213",
  "09:41:42 <INFO> Score: 0.93 Agent NLL: 15.00 Valid: 100% Step: 214",
  "09:41:47 <INFO> Score: 0.96 Agent NLL: 15.48 Valid:  98% Step: 215",
].join("\n");

describe("parseStepTimes", () => {
  it("reads the stamped step lines only", () => {
    const marks = parseStepTimes(LOG);
    expect(marks.map((m) => m.step)).toEqual([212, 213, 214, 215]);
    expect(marks[3].seconds - marks[0].seconds).toBe(15);
  });

  it("keeps counting across midnight", () => {
    const marks = parseStepTimes("23:59:58 <INFO> Score: 1 Step: 9\n00:00:03 <INFO> Score: 1 Step: 10\r\n");
    expect(marks[1].seconds - marks[0].seconds).toBe(5);
  });

  it("tolerates a tail cut mid-line and repeated steps", () => {
    const marks = parseStepTimes("e: 0.5 Step: 3\n10:00:00 <INFO> Step: 4\n10:00:02 <INFO> Step: 4\n");
    expect(marks).toEqual([{ step: 4, seconds: 36_002 }]);
  });
});

describe("secondsPerStep", () => {
  it("averages over the recent window", () => {
    expect(secondsPerStep(parseStepTimes(LOG))).toBe(5);
    const marks = [0, 10, 20, 22, 24].map((seconds, i) => ({ step: i + 1, seconds }));
    expect(secondsPerStep(marks, 3)).toBe(2);
  });

  it("needs a few steps", () => {
    expect(secondsPerStep([{ step: 1, seconds: 0 }, { step: 2, seconds: 5 }])).toBeNull();
    expect(secondsPerStep([])).toBeNull();
  });
});

describe("runProgress", () => {
  const started = "2026-10-27T07:24:04+00:00";
  const at = (minutes: number) => new Date(started).getTime() + minutes * 60_000;

  it("counts the steps done: current_step is the last step plus one", () => {
    const p = runProgress({ state: "running", currentStep: 216, maxSteps: 300, startedAt: started, now: at(20) });
    expect(p.done).toBe(215);
    expect(p.fraction).toBeCloseTo(215 / 300);
    expect(p.elapsedMs).toBe(20 * 60_000);
  });

  it("estimates from the log rate, else from the start", () => {
    const fromLog = runProgress({ state: "running", currentStep: 216, logStep: 215, maxSteps: 300, startedAt: started, secondsPerStep: 5, now: at(20) });
    expect(fromLog.remainingMs).toBe(85 * 5000);
    const fromStart = runProgress({ state: "running", currentStep: 101, maxSteps: 300, startedAt: started, now: at(10) });
    expect(fromStart.remainingMs).toBe(200 * 6000);
  });

  it("says nothing about the finish before a few steps", () => {
    const p = runProgress({ state: "running", currentStep: 2, maxSteps: 300, startedAt: started, now: at(1) });
    expect(p.remainingMs).toBeNull();
    expect(etaLabel(p.remainingMs)).toBe("estimating");
  });

  it("prefers the log step when it is ahead of the CSV", () => {
    expect(runProgress({ state: "running", currentStep: 10, logStep: 12, maxSteps: 50, startedAt: null, now: 0 }).done).toBe(12);
  });

  it("is full when completed and stops the clock at the finish", () => {
    const p = runProgress({ state: "completed", currentStep: 299, maxSteps: 300, startedAt: started, finishedAt: "2026-10-27T07:54:04+00:00", now: at(90) });
    expect(p).toMatchObject({ done: 300, fraction: 1, remainingMs: 0, elapsedMs: 30 * 60_000 });
  });

  it("copes with missing numbers", () => {
    const p = runProgress({ state: "queued", currentStep: null, maxSteps: null, startedAt: null, now: 0 });
    expect(p).toEqual({ done: 0, max: null, fraction: null, elapsedMs: null, remainingMs: null });
  });
});

describe("formatting", () => {
  it("counts down without going negative", () => {
    expect(countDown(60_000, 20_000)).toBe(40_000);
    expect(countDown(10_000, 20_000)).toBe(0);
    expect(countDown(null, 5)).toBeNull();
  });

  it("writes coarse durations", () => {
    expect(shortDuration(44_600)).toBe("45 s");
    expect(shortDuration(7 * 60_000 + 10_000)).toBe("7 min");
    expect(shortDuration(65 * 60_000)).toBe("1 h 05 min");
    expect(etaLabel(425_000)).toBe("~7 min left");
    expect(etaLabel(5_000)).toBe("finishing");
  });
});

describe("which runs", () => {
  const result = (job_id: string) => ({ result: JSON.stringify({ job_id, state: "running" }), isError: false });

  it("collects the jobs the conversation started", () => {
    const steps = [
      { name: "run_optimization", at: 2, result: result("bbbb2222") },
      { name: "get_optimization_status", at: 3, result: result("bbbb2222") },
      { name: "run_optimization", at: 1, result: result("aaaa1111") },
      { name: "run_optimization", at: 4, result: { result: "boom", isError: true } },
      { name: "run_optimization", at: 5 },
    ];
    expect(conversationJobIds(steps)).toEqual(["aaaa1111", "bbbb2222"]);
  });

  it("shows the conversation's running jobs, else every running one", () => {
    const jobs = [
      { id: "a", state: "running" as const },
      { id: "b", state: "completed" as const },
      { id: "c", state: "queued" as const },
    ];
    expect(runsToShow(jobs, ["c", "b"]).map((j) => j.id)).toEqual(["c"]);
    expect(runsToShow(jobs, ["b"]).map((j) => j.id)).toEqual(["a", "c"]);
    expect(runsToShow(jobs, []).map((j) => j.id)).toEqual(["a", "c"]);
  });
});
