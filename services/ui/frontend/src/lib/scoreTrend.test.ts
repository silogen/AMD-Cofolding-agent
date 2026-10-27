// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { parseStatus } from "./jobs";
import { downsample, MAX_POINTS, scoreSeries, sparkPath, trendTitle } from "./scoreTrend";

const job = { job_id: "3814ad67", state: "completed", max_steps: 300 };

describe("scoreSeries", () => {
  it("prefers score_trajectory, with its max", () => {
    const s = parseStatus({
      ...job,
      trajectory: {
        score_trajectory: [{ step: 2, mean_score: 0.5, max_score: 0.9 }, { step: 1, mean_score: 0.4 }],
        per_component_trajectory: { Score: [{ step: 1, mean_score: 0.1 }] },
      },
    });
    expect(scoreSeries(s)).toEqual([
      { step: 1, mean: 0.4, max: null },
      { step: 2, mean: 0.5, max: 0.9 },
    ]);
  });

  it("falls back to the Score component, sorted by step", () => {
    const s = parseStatus({
      ...job,
      trajectory: {
        per_component_trajectory: {
          Agent: [{ step: 1, mean_score: 6.2 }],
          Score: [{ step: 11, mean_score: 0.76 }, { step: 1, mean_score: 0.71 }],
        },
      },
    });
    expect(scoreSeries(s)).toEqual([
      { step: 1, mean: 0.71, max: null },
      { step: 11, mean: 0.76, max: null },
    ]);
  });

  it("is empty without a status or a score history", () => {
    expect(scoreSeries(null)).toEqual([]);
    expect(scoreSeries(parseStatus({ ...job, trajectory: { per_component_trajectory: { QED: [{ step: 1, mean_score: 0.5 }] } } }))).toEqual([]);
  });
});

describe("downsample", () => {
  it("keeps short series as they are", () => {
    expect(downsample([1, 2, 3])).toEqual([1, 2, 3]);
  });

  it("thins long ones evenly, keeping both ends", () => {
    const points = Array.from({ length: 301 }, (_, i) => i);
    const out = downsample(points);
    expect(out).toHaveLength(MAX_POINTS);
    expect(out[0]).toBe(0);
    expect(out.at(-1)).toBe(300);
    expect(out).toEqual([...out].sort((a, b) => a - b));
    expect(downsample(points, 1)).toEqual([300]);
  });
});

describe("sparkPath", () => {
  it("spans the width and the values' own range", () => {
    expect(sparkPath([0.2, 0.6, 0.4], 10, 6, 1)).toBe("M1.0,5.0L5.0,1.0L9.0,3.0");
  });

  it("draws flat values in the middle and nothing for fewer than two", () => {
    expect(sparkPath([0.5, 0.5], 10, 6, 1)).toBe("M1.0,3.0L9.0,3.0");
    expect(sparkPath([0.5], 10, 6)).toBe("");
    expect(sparkPath([NaN, 0.5], 10, 6)).toBe("");
  });
});

describe("trendTitle", () => {
  it("says where the mean went and over how many steps", () => {
    expect(trendTitle([{ step: 1, mean: 0.4123, max: null }, { step: 120, mean: 0.631, max: null }])).toBe("Mean score 0.41 → 0.63 over 120 steps");
    expect(trendTitle([{ step: 1, mean: 0.4, max: null }])).toBe("");
  });
});
