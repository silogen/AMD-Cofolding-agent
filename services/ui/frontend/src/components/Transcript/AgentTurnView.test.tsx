// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import type { ToolResultEvent } from "../../types/api";
import type { TraceEntry } from "../../lib/transcript";
import { TraceList } from "./AgentTurnView";

const result = (name: string, extra: Partial<ToolResultEvent> = {}): ToolResultEvent => ({
  type: "tool_result",
  seq: 1,
  ts: "",
  id: name,
  name,
  result: "{}",
  truncated: false,
  isError: false,
  panels: [],
  elapsedMs: 1000,
  ...extra,
});

const tool = (name: string, done?: Partial<ToolResultEvent>): TraceEntry => ({
  kind: "tool",
  id: name,
  name,
  arguments: {},
  at: 0,
  result: done ? result(name, done) : undefined,
});

describe("TraceList", () => {
  it("shows reasoning and steps in order, with each step's state", () => {
    const html = renderToStaticMarkup(
      <TraceList
        follow
        entries={[
          { kind: "thinking", id: "t1", content: "Check the target first." },
          tool("fetch_target_info", {}),
          tool("run_optimization", { result: JSON.stringify({ error: "TTP 'x' not found" }) }),
          { kind: "thinking", id: "t2", content: "Retry with a valid TTP." },
          tool("get_ttp"),
          { kind: "error", id: "e1", message: "Boltz timed out\nTraceback ..." },
        ]}
      />,
    );

    const order = ["Check the target first.", "fetch_target_info", "run_optimization", "Retry with a valid TTP.", "get_ttp", "Boltz timed out"];
    const positions = order.map((text) => html.indexOf(text));
    expect(positions.every((position) => position >= 0)).toBe(true);
    expect(positions).toEqual([...positions].sort((a, b) => a - b));

    expect(html).toContain('aria-label="Done"');
    expect(html).toContain('aria-label="Failed"');
    expect(html).toContain('aria-label="Running"');
    expect(html).toContain("TTP &#x27;x&#x27; not found");
    expect(html).not.toContain("Traceback");
    // Consecutive steps share one box.
    expect(html.match(/<ol/g)).toHaveLength(2);
  });

  it("says so when there is nothing yet", () => {
    expect(renderToStaticMarkup(<TraceList follow entries={[]} />)).toContain("Nothing yet.");
  });
});
