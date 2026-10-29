// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { JobSummary } from "../../lib/jobs";
import { JobPickerView } from "./JobPicker";
import { alertCounts, jobOptions } from "./useJobScope";

const job = (id: string, state: JobSummary["state"] = "completed"): JobSummary => ({
  id,
  ttpId: "egfr",
  ttpVersion: 1,
  model: null,
  type: "optimization",
  state,
  createdAt: null,
  startedAt: null,
  finishedAt: null,
  error: null,
  maxSteps: null,
  numEpochs: null,
  csvPath: null,
  checkpointPath: null,
  hasCheckpoint: false,
  workingDir: null,
});

const a = job("aaaa1111");
const b = job("bbbb2222", "running");
const c = job("cccc3333", "failed");
const d = job("dddd4444");

describe("jobOptions", () => {
  it("lists the conversation's jobs first, then the others", () => {
    expect(jobOptions([a, b, c], [c]).map(({ job, group }) => [job.id, group])).toEqual([
      ["cccc3333", "own"],
      ["aaaa1111", "other"],
      ["bbbb2222", "other"],
    ]);
  });

  it("puts the jobs seen in the conversation between its own and the rest", () => {
    expect(jobOptions([a, b, c, d], [d], [b, d]).map(({ job, group }) => [job.id, group])).toEqual([
      ["dddd4444", "own"],
      ["bbbb2222", "seen"],
      ["aaaa1111", "other"],
      ["cccc3333", "other"],
    ]);
  });
});

describe("alertCounts", () => {
  it("counts distinct alerts per job", () => {
    const counts = alertCounts([
      { jobId: "a", severity: "low", type: "alert", message: "Plateau" },
      { jobId: "a", severity: "high", type: "alert", message: "plateau " },
      { jobId: "a", severity: "medium", type: "alert", message: "Low validity" },
      { jobId: "b", severity: "medium", type: "alert", message: "Plateau" },
    ]);
    expect([...counts]).toEqual([
      ["a", 2],
      ["b", 1],
    ]);
  });
});

describe("JobPickerView", () => {
  it("names the job with its id and state", () => {
    const html = renderToStaticMarkup(<JobPickerView runs={[a, b]} own={[a]} job={a} error={null} onPick={() => undefined} />);
    expect(html).toContain("egfr v1");
    expect(html).toContain("aaaa1111 · completed");
    expect(html).toContain("Other jobs");
    expect(html).not.toContain('role="menu"');
  });

  it("says so when the conversation has no job", () => {
    const html = renderToStaticMarkup(<JobPickerView runs={[a]} own={[]} job={null} error={null} onPick={() => undefined} />);
    expect(html).toContain("No job in this conversation yet.");
    expect(html).toContain("Other jobs");
  });

  it("marks this chat's jobs in the list and checks the shown one", () => {
    const html = renderToStaticMarkup(<JobPickerView runs={[a, b]} own={[b]} job={a} error={null} onPick={() => undefined} initialOpen />);
    expect(html.match(/this chat/g)).toHaveLength(1);
    expect(html.indexOf("bbbb2222")).toBeLessThan(html.lastIndexOf("aaaa1111"));
    expect(html.match(/aria-checked="true"/g)).toHaveLength(1);
  });

  it("shows the three groups, apart, with their hints", () => {
    const html = renderToStaticMarkup(<JobPickerView runs={[a, b, c]} own={[c]} seen={[b]} job={c} error={null} onPick={() => undefined} initialOpen />);
    const order = ["cccc3333", "bbbb2222", "aaaa1111"].map((id) => html.lastIndexOf(id));
    expect(order).toEqual([...order].sort((x, y) => x - y));
    expect(html.match(/this chat/g)).toHaveLength(1);
    expect(html.match(/seen here/g)).toHaveLength(1);
    expect(html.match(/border-t pt-1/g)).toHaveLength(2);
  });

  it("counts a job's alerts quietly, on the row and in the list, and only when there are some", () => {
    const alerts = new Map([
      ["aaaa1111", 2],
      ["bbbb2222", 1],
    ]);
    const closed = renderToStaticMarkup(<JobPickerView runs={[a, b, c]} own={[a]} job={a} alerts={alerts} error={null} onPick={() => undefined} />);
    expect(closed).toContain('aria-label="2 alerts"');
    const none = renderToStaticMarkup(<JobPickerView runs={[c]} own={[c]} job={c} alerts={alerts} error={null} onPick={() => undefined} />);
    expect(none).not.toContain("alert");
    const listed = renderToStaticMarkup(<JobPickerView runs={[a, b, c]} own={[a]} job={a} alerts={alerts} error={null} onPick={() => undefined} initialOpen />);
    expect(listed.match(/aria-label="2 alerts"/g)).toHaveLength(2);
    expect(listed.match(/aria-label="1 alert"/g)).toHaveLength(1);
  });
});
