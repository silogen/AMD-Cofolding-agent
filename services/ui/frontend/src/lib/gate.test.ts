// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { affinityConcentration, affinityText, gate, pIC50 } from "./gate";

describe("gate", () => {
  it("passes a confident pose with clean PoseBusters", () => {
    const g = gate({ confidence: 0.82, ligandIptm: 0.71, posebustersPassed: 20, posebustersTotal: 20 });
    expect(g.status).toBe("pass");
    expect(g.reasons).toEqual([]);
  });

  it("fails on any check and says why", () => {
    const g = gate({ confidence: 0.82, ligandIptm: 0.41, posebustersPassed: 18, posebustersTotal: 20, failedChecks: ["clash"] });
    expect(g.status).toBe("fail");
    expect(g.reasons).toEqual(["Ligand ipTM 0.41 (needs > 0.5)", "PoseBusters 18/20 (needs all checks pass)", "failed: clash"]);
  });

  it("passes on confidence alone when PoseBusters did not run", () => {
    expect(gate({ confidence: 0.7, ligandIptm: 0.6 }).status).toBe("pass");
  });

  it("is unknown without confidence values", () => {
    expect(gate(undefined).status).toBe("unknown");
    expect(gate({ confidence: 0.7 }).status).toBe("unknown");
  });
});

describe("affinity", () => {
  it("brackets disagreeing ensemble members", () => {
    expect(affinityText({ affinity: -0.5, affinity1: -1.2, affinity2: 0.3 })).toBe("-1.20 to 0.30");
    expect(affinityText({ affinity: -0.5, affinity1: -0.7, affinity2: -0.3 })).toBe("-0.50");
    expect(affinityText({})).toBeNull();
  });

  it("converts to concentration and pIC50", () => {
    expect(affinityConcentration(-1.2)).toBe("≈ 63 nM");
    expect(affinityConcentration(0.5)).toBe("≈ 3.2 µM");
    expect(pIC50(-1)).toBe(7);
  });
});
