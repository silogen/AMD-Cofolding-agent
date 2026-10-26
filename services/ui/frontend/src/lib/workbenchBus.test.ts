// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it, vi } from "vitest";
import { onWorkbenchRequest, openWorkbench } from "./workbenchBus";

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
});
