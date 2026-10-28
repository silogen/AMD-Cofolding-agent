// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { cofoldBars, predictionBars, propertyBars } from "./detailFacts";

describe("cofoldBars", () => {
  it("draws scores as they are and marks those under the gate", () => {
    const facts = cofoldBars({ confidence: 0.82, ligandIptm: 0.31 });
    expect(facts.find((f) => f.label === "Confidence")?.bar).toEqual({ fraction: 0.82, tone: "ok" });
    expect(facts.find((f) => f.label === "Ligand ipTM")?.bar).toEqual({ fraction: 0.31, tone: "warn" });
  });

  it("puts affinity on pIC50 4–9 and PoseBusters as checks passed", () => {
    const facts = cofoldBars({ affinity: -1.5, posebustersPassed: 18, posebustersTotal: 20 });
    expect(facts.find((f) => f.label === "Affinity")?.bar).toEqual({ fraction: 0.7, tone: "ok" });
    expect(facts.find((f) => f.label === "PoseBusters")?.bar).toEqual({ fraction: 0.9, tone: "warn" });
    expect(cofoldBars({ affinity: 2 })[0].bar).toEqual({ fraction: 0, tone: "warn" });
  });
});

describe("propertyBars", () => {
  it("scales known properties and marks values outside the drug-like range", () => {
    const [mw, logp, qed] = propertyBars([
      { label: "MW", value: "612.4", unit: "Da" },
      { label: "cLogP", value: "−0.20" },
      { label: "QED", value: "0.71" },
    ]);
    expect(mw.tone).toBe("warn");
    expect(mw.bar?.tone).toBe("warn");
    expect(mw.bar?.fraction).toBeCloseTo(612.4 / 700);
    expect(logp.bar?.tone).toBe("ok");
    expect(logp.bar?.fraction).toBeCloseTo(0.2);
    expect(qed.bar).toEqual({ fraction: 0.71, tone: "ok" });
  });

  it("leaves unknown properties without a bar", () => {
    expect(propertyBars([{ label: "Charge", value: "1" }])[0].bar).toBeUndefined();
  });
});

describe("predictionBars", () => {
  it("draws unitless 0–1 values only", () => {
    const [group] = predictionBars([
      {
        title: "ADMET",
        facts: [
          { label: "hERG", value: "0.12" },
          { label: "LogD", value: "2.10" },
          { label: "Solubility", value: "0.50", unit: "log mol/L" },
        ],
      },
    ]);
    expect(group.facts.map((f) => f.bar?.fraction)).toEqual([0.12, undefined, undefined]);
  });
});
