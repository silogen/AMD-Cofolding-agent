// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { scopedJob } from "./jobScope";

const job = (id: string, state = "completed") => ({ id, state });

describe("scopedJob", () => {
  const jobs = [job("new", "failed"), job("mine"), job("other")];

  it("shows the picked job, from any conversation", () => {
    expect(scopedJob(jobs, [jobs[1]], "other")?.id).toBe("other");
  });

  it("defaults to the conversation's latest job that started", () => {
    expect(scopedJob(jobs, [jobs[0], jobs[1]], null)?.id).toBe("mine");
  });

  it("shows nothing for a conversation without jobs", () => {
    expect(scopedJob(jobs, [], null)).toBeNull();
  });
});
