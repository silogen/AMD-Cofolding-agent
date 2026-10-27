// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import {
  DEFAULT_RUN,
  RUN_KEY,
  loadRunSettings,
  modelLabel,
  parseMol2MolModels,
  runArguments,
  runCost,
  runProblems,
  saveRunSettings,
  seedList,
  startedRun,
} from "./ttpRun";

const memory = () => {
  const data = new Map<string, string>();
  return { getItem: (k: string) => data.get(k) ?? null, setItem: (k: string, v: string) => void data.set(k, v) };
};

describe("run settings", () => {
  it("remembers the last settings and drops bad values", () => {
    const store = memory();
    expect(loadRunSettings(store)).toEqual(DEFAULT_RUN);
    saveRunSettings({ maxSteps: 100, batchSize: 32, prior: "mol2mol", model: ".m2m_high", seeds: "CCO" }, store);
    expect(loadRunSettings(store)).toEqual({ maxSteps: 100, batchSize: 32, prior: "mol2mol", model: ".m2m_high", seeds: "CCO" });
    store.setItem(RUN_KEY, JSON.stringify({ maxSteps: -3, batchSize: 1.5, prior: "x", model: "/etc/passwd" }));
    expect(loadRunSettings(store)).toEqual(DEFAULT_RUN);
    store.setItem(RUN_KEY, "{oops");
    expect(loadRunSettings(store)).toEqual(DEFAULT_RUN);
  });

  it("checks steps, batch size and seeds", () => {
    expect(runProblems(DEFAULT_RUN)).toEqual([]);
    expect(runProblems({ ...DEFAULT_RUN, maxSteps: 0 })[0]).toMatch(/Steps/);
    expect(runProblems({ ...DEFAULT_RUN, batchSize: 10_000 })[0]).toMatch(/Batch/);
    expect(runProblems({ ...DEFAULT_RUN, prior: "mol2mol" })[0]).toMatch(/seed/);
  });

  it("builds run_optimization arguments for each prior", () => {
    expect(runArguments("t", DEFAULT_RUN)).toEqual({ ttp_id: "t", model: ".reinvent", max_steps: 300, batch_size: 16 });
    expect(runArguments("t", { ...DEFAULT_RUN, prior: "mol2mol", seeds: "CCO\n c1ccccc1 ,CCN" })).toEqual({
      ttp_id: "t",
      model: ".m2m_medium",
      max_steps: 300,
      batch_size: 16,
      seed_smiles: ["CCO", "c1ccccc1", "CCN"],
    });
    expect(seedList("  ")).toEqual([]);
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

  it("reads the started job or the error", () => {
    expect(startedRun({ job_id: "3814ad67", status: "running" })).toEqual({ jobId: "3814ad67" });
    expect(startedRun(JSON.stringify({ error: "Mol2Mol models require 'seed_smiles'" }))).toEqual({ error: "Mol2Mol models require 'seed_smiles'" });
    expect(startedRun("Tool failed")).toEqual({ error: "Tool failed" });
  });
});
