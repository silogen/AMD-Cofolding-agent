// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import type { PropertiesPanel } from "../types/api";
import { cofoldFacts, componentFacts, ic50, metricsFromValues, otherRawValues, predictionGroups, propertyFacts, propertyKey } from "./moleculeFacts";

describe("ic50", () => {
  it("reads log10(IC50 / µM) as a concentration", () => {
    expect(ic50(-1.66)).toEqual({ value: "22", unit: "nM" });
    expect(ic50(0.5)).toEqual({ value: "3.2", unit: "µM" });
    expect(ic50(3.5)).toEqual({ value: "3.2", unit: "mM" });
    expect(ic50(-3)).toEqual({ value: "1.0", unit: "nM" });
  });
});

describe("cofoldFacts", () => {
  it("lists affinity, ligand ipTM, confidence and PoseBusters, marking what fails the gate", () => {
    const facts = cofoldFacts({ affinity: -1.66, ligandIptm: 0.93, confidence: 0.55, posebustersPassed: 19, posebustersTotal: 22 });
    expect(facts.map((f) => [f.label, f.value, f.unit, f.tone])).toEqual([
      ["Affinity", "22", "nM", undefined],
      ["Ligand ipTM", "0.93", undefined, undefined],
      ["Confidence", "0.55", undefined, "warn"],
      ["PoseBusters", "19/22", undefined, "warn"],
    ]);
  });

  it("flags an affinity the two Boltz models disagree on", () => {
    const [affinity] = cofoldFacts({ affinity: -1, affinity1: -2.2, affinity2: 0.1 });
    expect(affinity.value).toBe("100");
    expect(affinity.title).toContain("disagree");
  });

  it("is empty without metrics", () => {
    expect(cofoldFacts({})).toEqual([]);
  });
});

describe("propertyFacts", () => {
  it("prefers the conversation's values and fills the rest from RDKit, in a fixed order", () => {
    const facts = propertyFacts(
      { "QED (raw)": 0.678, "Molecular weight": 412.559, "Synthetic accessibility": 2.39, Unrelated: 3 },
      { mw: 400, clogp: 4.3046, tpsa: 66.71, hbd: 1, hba: 6, rotb: 4, rings: 4, heavyAtoms: 30 },
    );
    expect(facts.map((f) => `${f.label} ${f.value}${f.unit ? ` ${f.unit}` : ""}`)).toEqual([
      "QED 0.68",
      "MW 412.6 Da",
      "cLogP 4.30",
      "TPSA 66.7 Å²",
      "SA 2.4",
      "HBD 1",
      "HBA 6",
      "RotB 4",
    ]);
  });

  it("knows the usual names", () => {
    expect(["qed", "MW", "mol_wt", "cLogP (raw)", "TPSA", "sa_score", "SAScore"].map(propertyKey)).toEqual(["qed", "mw", "mw", "clogp", "tpsa", "sa", "sa"]);
    expect(propertyKey("Boltz affinity")).toBeNull();
  });
});

describe("component facts", () => {
  const components = [
    { name: "Boltz affinity", value: 0.98, raw: -1.66, cofold: true },
    { name: "QED", value: 0.29, raw: 0.29 },
    { name: "Aqueous solubility", value: 0.5, raw: -4.12 },
  ];

  it("shows desirabilities with the raw value on hover", () => {
    expect(componentFacts(components)[0]).toEqual({ label: "Boltz affinity", value: "0.98", title: "Raw −1.66 → desirability" });
  });

  it("keeps as properties only the raw values that are neither known properties nor cofolding", () => {
    expect(otherRawValues(components)).toEqual([{ label: "Aqueous solubility", value: "−4.12" }]);
  });
});

describe("metricsFromValues", () => {
  it("reads Boltz columns of a scoring table", () => {
    expect(metricsFromValues({ ligand_iptm: 0.8, confidence_score: 0.9, n_checks_passed: 20, qed: 0.5 })).toEqual({ ligandIptm: 0.8, confidence: 0.9, posebustersPassed: 20 });
    expect(metricsFromValues({ qed: 0.5 })).toBeUndefined();
  });
});

describe("predictionGroups", () => {
  const panel = (service: string, smiles: string): PropertiesPanel => ({
    kind: "properties",
    tool: service === "retrodmta" ? "predict_retrodmta" : "predict_admet",
    service,
    target: null,
    endpoints: ["logd", "aqueous_solubility"],
    molecules: [
      { smiles, standardized: null, label: null, ok: true, error: null, adPass: true, values: { logd: { mean: 2.13 }, aqueous_solubility: { mean: -4.3, std: 0.2, unit: "log10(mol/L)" } } },
    ],
  });

  it("groups the molecule's predictions by service, newest first", () => {
    const groups = predictionGroups("CCO", [
      { item: panel("admet", "CCO"), tool: "predict_admet", at: 2 },
      { item: panel("admet", "CCO"), tool: "predict_admet", at: 1 },
      { item: panel("admet", "CCN"), tool: "predict_admet", at: 0 },
    ]);
    expect(groups).toEqual([
      {
        title: "ADMET",
        facts: [
          { label: "LogD", value: "2.13" },
          { label: "Aq. solubility", value: "−4.30 ± 0.20", unit: "log₁₀ M", title: "log10(mol/L)" },
        ],
      },
    ]);
  });
});
