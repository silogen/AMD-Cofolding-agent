// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it, vi } from "vitest";
import { focusedRun, notifyRunsChanged, onRunsChanged, onWorkbenchRequest, openRun, openWorkbench } from "./workbenchBus";

describe("workbenchBus", () => {
  it("delivers requests to listeners until they unsubscribe", () => {
    const listener = vi.fn();
    const off = onWorkbenchRequest(listener);
    openWorkbench("jobs");
    expect(listener).toHaveBeenCalledWith("jobs");
    off();
    openWorkbench("ttp");
    expect(listener).toHaveBeenCalledTimes(1);
  });

  it("focuses a run and opens its tab", () => {
    const listener = vi.fn();
    const off = onWorkbenchRequest(listener);
    openRun("abc123", "results");
    expect(focusedRun()).toBe("abc123");
    expect(listener).toHaveBeenCalledWith("results");
    off();
  });

  it("tells listeners the runs changed", () => {
    const listener = vi.fn();
    const off = onRunsChanged(listener);
    notifyRunsChanged();
    off();
    notifyRunsChanged();
    expect(listener).toHaveBeenCalledTimes(1);
  });
});
