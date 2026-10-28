// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import {
  DEFAULT_RUN,
  RUN_KEY,
  addSeeds,
  loadRunSettings,
  modelLabel,
  mol2molChoices,
  parseMol2MolModels,
  rememberRun,
  removeSeed,
  runCost,
  runProblems,
  runRequest,
  saveRunSettings,
  seedChoices,
  seedList,
  ttpRequest,
  withOfferedModel,
} from "./ttpRun";

const memory = () => {
  const data = new Map<string, string>();
  return { getItem: (k: string) => data.get(k) ?? null, setItem: (k: string, v: string) => void data.set(k, v) };
};

describe("run settings", () => {
  it("remembers the last settings, but never the prior, and drops bad values", () => {
    const store = memory();
    expect(loadRunSettings(store)).toEqual(DEFAULT_RUN);
    expect(DEFAULT_RUN.prior).toBeNull();
    saveRunSettings({ enabled: true, maxSteps: 100, batchSize: 32, prior: "mol2mol", model: ".m2m_high", seeds: "CCO" }, store);
    expect(loadRunSettings(store)).toEqual({ enabled: true, maxSteps: 100, batchSize: 32, prior: null, model: ".m2m_high", seeds: "CCO" });
    store.setItem(RUN_KEY, JSON.stringify({ maxSteps: -3, batchSize: 1.5, prior: "x", model: "/etc/passwd" }));
    expect(loadRunSettings(store)).toEqual(DEFAULT_RUN);
    store.setItem(RUN_KEY, "{oops");
    expect(loadRunSettings(store)).toEqual(DEFAULT_RUN);
  });

  it("checks the prior, steps, batch size and seeds", () => {
    expect(runProblems(DEFAULT_RUN)).toEqual([expect.stringMatching(/from scratch or analogues/)]);
    expect(runProblems({ ...DEFAULT_RUN, prior: "reinvent" })).toEqual([]);
    expect(runProblems({ ...DEFAULT_RUN, prior: "reinvent", maxSteps: 0 })[0]).toMatch(/Steps/);
    expect(runProblems({ ...DEFAULT_RUN, prior: "reinvent", batchSize: 10_000 })[0]).toMatch(/Batch/);
    expect(runProblems({ ...DEFAULT_RUN, prior: "mol2mol" })).toEqual(["Mol2Mol needs at least one seed SMILES."]);
    expect(runProblems({ ...DEFAULT_RUN, prior: "mol2mol", seeds: "CCO" })).toEqual([]);
  });

  it("adds and removes seeds without duplicates", () => {
    expect(addSeeds("CCO\nCCN", ["CCN", " c1ccccc1 ", "c1ccccc1", ""])).toBe("CCO\nCCN\nc1ccccc1");
    expect(addSeeds("", ["CCO"])).toBe("CCO");
    expect(removeSeed("CCO, CCN c1ccccc1", "CCN")).toBe("CCO\nc1ccccc1");
  });

  it("offers each molecule of the conversation once", () => {
    expect(seedChoices([{ smiles: "CCO", label: "ethanol" }], undefined, [{ smiles: " CCO " }, { smiles: "CCN" }, { smiles: " " }])).toEqual([
      { smiles: "CCO", label: "ethanol" },
      { smiles: "CCN" },
    ]);
  });

  it("writes the chat request, with the run only when asked", () => {
    const ttp = { id: "sol", name: "Solubility" };
    expect(ttpRequest(ttp, DEFAULT_RUN)).toBe('Use the TTP "Solubility" (sol)');
    expect(ttpRequest(ttp, { ...DEFAULT_RUN, enabled: true, prior: "reinvent" })).toBe('Run REINVENT on the TTP "Solubility" (sol): Reinvent (de novo) (.reinvent), 300 steps, batch size 16.');
    expect(ttpRequest(ttp, { ...DEFAULT_RUN, enabled: true, prior: "mol2mol", seeds: "CCO\n c1ccccc1 ,CCN" })).toBe(
      'Run REINVENT on the TTP "Solubility" (sol): Mol2Mol medium similarity (.m2m_medium) from CCO c1ccccc1 CCN, 300 steps, batch size 16.',
    );
    expect(seedList("  ")).toEqual([]);
  });

  it("asks for a run on a stored TTP whatever the builder switch says", () => {
    const ttp = { id: "sol", name: "Solubility" };
    expect(runRequest(ttp, { ...DEFAULT_RUN, prior: "reinvent" })).toBe(ttpRequest(ttp, { ...DEFAULT_RUN, prior: "reinvent", enabled: true }));
    expect(runRequest(ttp, { ...DEFAULT_RUN, prior: "mol2mol", model: ".m2m_high", seeds: "CCO" })).toBe(
      'Run REINVENT on the TTP "Solubility" (sol): Mol2Mol high similarity (.m2m_high) from CCO, 300 steps, batch size 16.',
    );
  });

  it("remembers edits from elsewhere without flipping the builder switch", () => {
    const store = memory();
    saveRunSettings({ ...DEFAULT_RUN, enabled: true }, store);
    rememberRun({ enabled: false, maxSteps: 50 }, store);
    expect(loadRunSettings(store)).toEqual({ ...DEFAULT_RUN, enabled: true, maxSteps: 50 });
  });

  it("offers installed Mol2Mol priors, else the default", () => {
    expect(mol2molChoices([{ key: ".m2m_high", label: "h", available: false }]).map((m) => m.key)).toEqual([".m2m_medium"]);
    const choices = mol2molChoices([{ key: ".m2m_high", label: "h", available: true }]);
    expect(withOfferedModel(DEFAULT_RUN, choices).model).toBe(".m2m_high");
    expect(withOfferedModel({ ...DEFAULT_RUN, model: ".m2m_high" }, choices).model).toBe(".m2m_high");
  });

  it("counts molecules and cofolds", () => {
    expect(runCost(DEFAULT_RUN, false)).toEqual({ molecules: 4800, cofolds: 0 });
    expect(runCost({ ...DEFAULT_RUN, maxSteps: 50 }, true)).toEqual({ molecules: 800, cofolds: 800 });
  });
});

describe("models and results", () => {
  it("names priors", () => {
    expect(modelLabel(".reinvent")).toBe("Reinvent (de novo)");
    expect(modelLabel(".m2m_medium")).toBe("Mol2Mol medium similarity");
    expect(modelLabel(".m2m_scaffold_generic")).toBe("Mol2Mol scaffold generic");
    expect(modelLabel(null)).toBe("—");
  });

  it("lists Mol2Mol priors, installed first", () => {
    const models = parseMol2MolModels([
      { key: ".reinvent", exists: true, model_type: "Reinvent (de novo generation)" },
      { key: ".m2m_high", exists: false, model_type: "Mol2Mol (molecule-to-molecule translation)" },
      { key: ".m2m_medium", exists: true, model_type: "Mol2Mol (molecule-to-molecule translation)" },
    ]);
    expect(models.map((m) => [m.key, m.available])).toEqual([
      [".m2m_medium", true],
      [".m2m_high", false],
    ]);
    expect(parseMol2MolModels("nope")).toEqual([]);
  });
});
