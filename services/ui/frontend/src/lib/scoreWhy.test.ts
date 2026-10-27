// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { explainScore } from "./scoreWhy";
import type { Objective, Ttp } from "./ttp";

const objective = (name: string, weight: number, transform: Objective["transform"] = null): Objective => ({ name, type: "native", weight, transform });

const ttp = (objectives: Objective[], aggregation: Ttp["scoring"]["aggregation"] = "geometric_mean"): Ttp => ({
  id: "t",
  name: "t",
  version: 1,
  scoring: { objectives, aggregation },
});

describe("explainScore", () => {
  it("matches REINVENT's total for one objective and reads the raw value", () => {
    const why = explainScore(ttp([objective("Aqueous solubility", 0.6, { type: "sigmoid", low: -6, high: -3, k: 0.5 })]), {
      "Aqueous solubility": 0.591354,
      "Aqueous solubility (raw)": -4.4037,
    })!;
    expect(why.total).toBeCloseTo(0.591354, 6);
    expect(why.items[0]).toMatchObject({ name: "Aqueous solubility", raw: -4.4037, share: 1 });
    expect(why.verdict).toBe("Held back by Aqueous solubility");
  });

  it("sorts by drag on a weighted geometric mean and names the worst", () => {
    const why = explainScore(ttp([objective("QED", 1), objective("SA score", 1), objective("Affinity", 2)]), { QED: 0.9, "SA score": 0.3, Affinity: 0.8 })!;
    expect(why.items.map((i) => i.name)).toEqual(["SA score", "Affinity", "QED"]);
    expect(why.total).toBeCloseTo(0.9 ** 0.25 * 0.3 ** 0.25 * 0.8 ** 0.5, 6);
    expect(why.verdict).toBe("Held back by SA score");
  });

  it("names two objectives when they pull about equally", () => {
    const why = explainScore(ttp([objective("A", 1), objective("B", 1), objective("C", 1)]), { A: 0.4, B: 0.45, C: 1 })!;
    expect(why.verdict).toBe("Held back by A and B");
  });

  it("computes the desirability from the raw value when only that is known", () => {
    const why = explainScore(ttp([objective("MW", 1, { type: "step", low: 200, high: 500 })]), { "MW (raw)": 650 })!;
    expect(why.items[0].desirability).toBe(0);
    expect(why.total).toBe(0);
    expect(why.verdict).toBe("Zeroed by MW");
  });

  it("uses the arithmetic mean when the TTP asks for it", () => {
    const why = explainScore(ttp([objective("A", 3), objective("B", 1)], "arithmetic_mean"), { A: 1, B: 0 })!;
    expect(why.total).toBeCloseTo(0.75);
    expect(why.items[0].name).toBe("B");
    expect(why.items[0].drag).toBeCloseTo(0.25);
  });

  it("says when nothing holds the score back, and flags a missing value", () => {
    expect(explainScore(ttp([objective("A", 1), objective("B", 1)]), { A: 0.97, B: 0.95 })!.verdict).toBe("Meets every objective");
    expect(explainScore(ttp([objective("A", 1), objective("B", 1)]), { A: 0.9 })!.verdict).toBe("B reported no value");
  });

  it("returns null when no objective has a value or weight", () => {
    expect(explainScore(ttp([objective("A", 1)]), { other: 1 })).toBeNull();
    expect(explainScore(ttp([objective("A", 0)]), { A: 1 })).toBeNull();
  });
});
