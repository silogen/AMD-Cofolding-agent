// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { callTool, clearToolCache } from "./api";

/** Answer every tool call with its name, after a tick, and report the calls. */
function gateway() {
  const fetcher = vi.fn<(path: string, init?: RequestInit) => Promise<Response>>(async (path) => {
    await new Promise((resolve) => setTimeout(resolve, 0));
    return new Response(JSON.stringify({ name: path, result: [], panels: [] }), { status: 200 });
  });
  vi.stubGlobal("fetch", fetcher);
  return fetcher;
}

beforeEach(() => clearToolCache());
afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("callTool", () => {
  it("sends identical read-only calls in flight at the same time once", async () => {
    const fetcher = gateway();
    await Promise.all([callTool("get_top_candidates", { job_id: "a" }), callTool("get_top_candidates", { job_id: "a" })]);
    await callTool("get_top_candidates", { job_id: "b" });
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it("reuses a recent job list, and reads it again once it is old", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    const fetcher = gateway();
    await callTool("list_jobs");
    await callTool("list_jobs");
    expect(fetcher).toHaveBeenCalledTimes(1);
    vi.setSystemTime(Date.now() + 9_000);
    await callTool("list_jobs");
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it("never shares a call that changes something, and forgets what was read before it", async () => {
    const fetcher = gateway();
    await callTool("list_jobs");
    await Promise.all([callTool("stop_optimization", { job_id: "a" }), callTool("stop_optimization", { job_id: "a" })]);
    await callTool("list_jobs");
    const names = fetcher.mock.calls.map(([path]) => path.split("/").pop());
    expect(names).toEqual(["list_jobs", "stop_optimization", "stop_optimization", "list_jobs"]);
  });

  it("does not keep an answer read while a change was being made", async () => {
    const fetcher = gateway();
    const read = callTool("get_optimization_status", { job_id: "a" });
    await callTool("stop_optimization", { job_id: "a" });
    await read;
    await callTool("get_optimization_status", { job_id: "a" });
    expect(fetcher.mock.calls.filter(([path]) => path.endsWith("get_optimization_status"))).toHaveLength(2);
  });
});
