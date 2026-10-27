// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ConversationEvent, SessionSummary } from "../types/api";

const api = vi.hoisted(() => ({ sessionEvents: vi.fn(), callTool: vi.fn(), download: vi.fn() }));
vi.mock("./api", () => api);
vi.mock("./rdkit", () => ({
  loadRDKit: () => Promise.reject(new Error("no WebAssembly in tests")),
  canonical: () => null,
  depict: () => null,
}));

const { sessionReport, sessionToMarkdown, slug } = await import("./exportSession");

const SESSION: SessionSummary = { id: "s-20261027-071613-711b5a", title: "Solubility run", created: "2026-10-27T07:16:13Z", updated: "2026-10-27T07:30:00Z", turns: 1, busy: true };

let seq = 0;
const at = () => {
  seq += 1;
  return { seq, ts: new Date(Date.UTC(2026, 9, 27, 7, 0, seq)).toISOString() };
};

const tool = (name: string, args: Record<string, unknown>, result: unknown): ConversationEvent[] => [
  { type: "tool_call", id: `c${seq}`, name, arguments: args, ...at() },
  { type: "tool_result", id: `c${seq - 1}`, name, result: JSON.stringify(result), truncated: false, isError: false, panels: [], elapsedMs: 5, ...at() },
];

function conversation(): ConversationEvent[] {
  seq = 0;
  return [
    { type: "user_message", content: "Optimise O=C(O)c1ccc(-c2ccncc2)cc1", ...at() },
    { type: "status", state: "busy", ...at() },
    ...tool("run_optimization", { ttp_id: "sol", seed_smiles: ["O=C(O)c1ccc(-c2ccncc2)cc1"] }, { job_id: "3814ad67", state: "running" }),
    ...tool("get_optimization_status", { job_id: "3814ad67" }, { ...STATUS, trajectory: { ...STATUS.trajectory, current_step: 4 } }),
    { type: "assistant_message", content: "The run is going. Solubility is already improving.", inputTokens: 1, outputTokens: 1, ...at() },
    { type: "status", state: "idle", ...at() },
  ];
}

const TTP = {
  id: "sol",
  name: "solubility",
  version: 1,
  scoring: { objectives: [{ name: "Aqueous solubility", type: "external", service: "admet", weight: 0.6, transform: { type: "sigmoid", low: -6, high: -3, k: 0.5 } }] },
};

const STATUS = {
  job_id: "3814ad67",
  ttp_id: "sol",
  ttp_version: 1,
  state: "running",
  max_steps: 300,
  trajectory: {
    current_step: 280,
    per_component: { Score: { mean: 0.86, max: 1, min: 0 } },
    per_component_trajectory: { Score: [{ step: 1, mean_score: 0.5 }, { step: 279, mean_score: 0.9 }] },
  },
};

const CANDIDATES = {
  candidates: [{ smiles: "O=C(O)c1ccc(Cc2ccncc2)cc1", components: { Score: { transformed: 0.998 }, "Aqueous solubility": { transformed: 0.998, raw: -2.89 } } }],
};

beforeEach(() => {
  api.sessionEvents.mockResolvedValue({ session: SESSION, events: conversation() });
  api.callTool.mockImplementation((name: string) => {
    if (name === "get_ttp") return Promise.resolve({ name, result: TTP, panels: [] });
    if (name === "get_optimization_status") return Promise.resolve({ name, result: STATUS, panels: [] });
    if (name === "get_top_candidates") return Promise.resolve({ name, result: CANDIDATES, panels: [] });
    return Promise.reject(new Error(`unexpected ${name}`));
  });
});

describe("sessionReport", () => {
  it("fetches the profile and the run the conversation names", async () => {
    const { name, html } = await sessionReport(SESSION.id, new Date("2026-10-27T08:00:00Z"));
    expect(name).toBe("solubility-run-report.html");
    expect(api.callTool).toHaveBeenCalledWith("get_ttp", { ttp_id: "sol" });
    expect(api.callTool).toHaveBeenCalledWith("get_optimization_status", { job_id: "3814ad67" });
    expect(html).toContain('<section id="ttp">');
    expect(html).toContain('<section id="runs">');
    expect(html).toContain("279 / 300");
    expect(html).toContain("O=C(O)c1ccc(Cc2ccncc2)cc1");
    expect(html).toContain("Solubility is already improving.");
    expect(html).not.toContain('<section id="structures">');
    expect(html).not.toContain('<section id="target">');
  });

  it("falls back to the last status the conversation saw when the job manager fails", async () => {
    api.callTool.mockRejectedValue(new Error("gateway down"));
    const { html } = await sessionReport(SESSION.id);
    expect(html).toContain('<section id="runs">');
    expect(html).toContain("3 / 300");
    expect(html).not.toContain('<section id="ttp">');
    expect(html).toContain('<section id="answers">');
  });
});

describe("sessionToMarkdown", () => {
  it("writes questions, tool calls and answers", () => {
    const markdown = sessionToMarkdown(SESSION, conversation());
    expect(markdown).toContain("# Solubility run");
    expect(markdown).toContain("- `run_optimization` (ok, 0.0 s)");
    expect(markdown).toContain("- `get_optimization_status` (ok, 0.0 s)");
    expect(markdown).toContain("## Answer\n\nThe run is going.");
  });

  it("makes file names from titles", () => {
    expect(slug("Cofold JQ1 into BRD4 BD1!", "x")).toBe("cofold-jq1-into-brd4-bd1");
    expect(slug("???", "s-1")).toBe("s-1");
  });
});
