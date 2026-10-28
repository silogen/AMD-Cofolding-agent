// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";

import { structureFacts, verdictLabel } from "./structureFacts";

describe("structureFacts", () => {
  it("gives the affinity as an IC50 with its unit, and neither ligand ipTM nor binder probability", () => {
    const facts = structureFacts({ confidence: 0.98, ligandIptm: 0.99, affinity: -1.85, binderProbability: 0.99, posebustersPassed: 18, posebustersTotal: 22 });
    expect(facts.map((fact) => fact.label)).toEqual(["Affinity", "Confidence", "PoseBusters"]);
    expect(facts[0]).toMatchObject({ value: "14", unit: "nM" });
    expect(facts[2].tone).toBe("warn");
  });
});

describe("verdictLabel", () => {
  it("says Valid or Invalid, and nothing when the verdict is unknown", () => {
    expect([verdictLabel("pass"), verdictLabel("fail"), verdictLabel("unknown")]).toEqual(["Valid", "Invalid", null]);
  });
});
