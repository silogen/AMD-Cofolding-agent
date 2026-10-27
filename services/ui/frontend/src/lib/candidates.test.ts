// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";

import type { MoleculePanel, StructurePanel, TablePanel } from "../types/api";
import type { Artifacts, Sourced } from "./artifacts";
import {
  buildCandidates,
  candidateNumbers,
  candidateValue,
  candidatesToCsv,
  candidatesToSdf,
  descriptorValue,
  ro5Violations,
  cofoldPrompt,
  MAX_COFOLD,
  numericColumns,
  scoreColor,
} from "./candidates";

const sourced = <T extends { tool: string }>(item: T, at = 1): Sourced<T> => ({ item, tool: item.tool, at });

const table = (tool: string, columns: string[], rows: TablePanel["rows"]): Sourced<TablePanel> =>
  sourced({ kind: "table", tool, columns, rows, totalRows: rows.length, truncated: false });

const molecule = (smiles: string, label: string | null = null, tool = "draw"): Sourced<MoleculePanel> =>
  sourced({ kind: "molecule", tool, smiles, label });

const structure = (path: string, smiles: string | undefined, metrics?: StructurePanel["metrics"]): Sourced<StructurePanel> =>
  sourced({ kind: "structure", tool: "cofold", name: path, path, url: `/api/artifacts/${path}`, format: "cif", smiles, metrics });

const empty = (): Pick<Artifacts, "tables" | "molecules" | "structures"> => ({ tables: [], molecules: [], structures: [] });

describe("buildCandidates", () => {
  it("reads score_molecules rows: score, name and numeric components", () => {
    const artifacts = empty();
    artifacts.tables.push(
      table(
        "score_molecules",
        ["#", "smiles", "name", "score", "qed", "affinity_pred_value", "note"],
        [
          { "#": 1, smiles: "CCO", name: "ethanol", score: 0.87, qed: 0.71, affinity_pred_value: -9.4, note: "x" },
          { "#": 2, smiles: "c1ccccc1", name: null, score: 0.4, qed: null, affinity_pred_value: -6.1, note: null },
        ],
      ),
    );
    const [a, b] = buildCandidates(artifacts);
    expect(a).toMatchObject({ smiles: "CCO", name: "ethanol", origin: "score_molecules", score: 0.87 });
    expect(a.components).toEqual({ qed: 0.71, affinity_pred_value: -9.4 });
    expect(a.gate.status).toBe("unknown");
    expect(b.name).toBeUndefined();
    expect(b.components).toEqual({ affinity_pred_value: -6.1 });
  });

  it("reads get_top_candidates rows with a total_score and skips tables without SMILES", () => {
    const artifacts = empty();
    artifacts.tables.push(
      table("get_top_candidates", ["#", "smiles", "total_score"], [{ "#": 1, smiles: " CCN ", total_score: 0.5 }]),
      table("list_scoring_components", ["#", "component", "weight"], [{ "#": 1, component: "QED", weight: 1 }]),
    );
    const candidates = buildCandidates(artifacts);
    expect(candidates).toHaveLength(1);
    expect(candidates[0]).toMatchObject({ smiles: "CCN", score: 0.5, components: {} });
  });

  it("merges tables, molecules and structures by SMILES, newest value first", () => {
    const artifacts = empty();
    artifacts.tables.push(
      table("score_molecules", ["smiles", "score", "qed"], [{ smiles: "CCO", score: 0.9, qed: 0.8 }]),
      table("get_top_candidates", ["smiles", "total_score", "qed", "sa"], [{ smiles: "CCO", total_score: 0.1, qed: 0.2, sa: 0.3 }]),
    );
    artifacts.molecules.push(molecule("CCO", "Ethanol"), molecule("CCC", "Propane"));
    artifacts.structures.push(
      structure("new/CCO.cif", "CCO", { confidence: 0.8, ligandIptm: 0.7 }),
      structure("old/CCO.cif", "CCO", { confidence: 0.2, ligandIptm: 0.1 }),
      structure("other.cif", undefined),
    );
    const candidates = buildCandidates(artifacts);
    expect(candidates.map((c) => c.smiles)).toEqual(["CCO", "CCC"]);
    const [ethanol, propane] = candidates;
    expect(ethanol).toMatchObject({
      name: "Ethanol",
      origin: "score_molecules",
      origins: ["score_molecules", "get_top_candidates", "draw", "cofold"],
      score: 0.9,
      components: { qed: 0.8, sa: 0.3 },
      structurePath: "new/CCO.cif",
    });
    expect(ethanol.gate.status).toBe("pass");
    expect(propane).toMatchObject({ origin: "draw", name: "Propane", components: {} });
  });

  it("gates on a failing complex", () => {
    const artifacts = empty();
    artifacts.structures.push(structure("a.cif", "CCO", { confidence: 0.3, ligandIptm: 0.7 }));
    const [c] = buildCandidates(artifacts);
    expect(c.gate.status).toBe("fail");
    expect(c.origin).toBe("cofold");
  });
});

describe("numericColumns and candidateValue", () => {
  it("lists score, components and metrics that are present", () => {
    const artifacts = empty();
    artifacts.tables.push(table("t", ["smiles", "score", "qed"], [{ smiles: "CCO", score: 0.5, qed: 0.6 }]));
    artifacts.structures.push(structure("a.cif", "CCO", { confidence: 0.9 }));
    const candidates = buildCandidates(artifacts);
    expect(numericColumns(candidates).map((c) => c.key)).toEqual(["score", "qed", "metric:confidence"]);
    expect(candidateValue(candidates[0], "metric:confidence")).toBe(0.9);
    expect(candidateValue(candidates[0], "qed")).toBe(0.6);
    expect(candidateValue(candidates[0], "metric:plddt")).toBeUndefined();
  });
});

describe("candidateNumbers", () => {
  it("keeps the components and names the total Score", () => {
    const [c] = buildCandidates({ ...empty(), tables: [table("t", ["smiles", "score", "QED"], [{ smiles: "CCO", score: 0.7, QED: 0.4 }])] });
    expect(candidateNumbers(c)).toEqual({ ...c.components, Score: 0.7 });
    expect(candidateNumbers({ ...c, score: undefined, components: { QED: 0.4 } })).toEqual({ QED: 0.4 });
  });
});

describe("scoreColor", () => {
  it("uses the three bands", () => {
    expect(scoreColor(0.7)).toBe("#3ccf91");
    expect(scoreColor(0.4)).toBe("#f5a524");
    expect(scoreColor(0.39)).toBe("#ff4d4f");
  });
});

describe("exports", () => {
  it("writes CSV with quoting", () => {
    const artifacts = empty();
    artifacts.tables.push(table("t", ["smiles", "name", "score"], [{ smiles: "CCO", name: 'a, "b"', score: 1 }]));
    const csv = candidatesToCsv(buildCandidates(artifacts)).split("\n");
    expect(csv[0]).toBe("smiles,name,origin,score,confidence,ligandIptm,iptm,plddt,affinity,binderProbability,gate,structure");
    expect(csv[1]).toBe('CCO,"a, ""b""",t,1,,,,,,,unknown,');
  });

  it("names at most two molecules in a cofold request", () => {
    const artifacts = empty();
    artifacts.molecules.push(molecule("CCO", "A"), molecule("CCC"), molecule("CCN"));
    const prompt = cofoldPrompt(buildCandidates(artifacts));
    expect(MAX_COFOLD).toBe(2);
    expect(prompt).toContain("- A: CCO");
    expect(prompt).toContain("- CCC");
    expect(prompt).not.toContain("CCN");
    expect(prompt).toContain("3 were selected");
  });
});

describe("descriptors", () => {
  const props = { mw: 520, clogp: 5.5, tpsa: 80, hbd: 2, hba: 11, rotb: 4, ro5: 0 };
  it("counts rule-of-five violations", () => {
    expect(ro5Violations(props)).toBe(3);
    expect(ro5Violations({ mw: 300, clogp: 2, hbd: 1, hba: 4 })).toBe(0);
  });
  it("reads a descriptor by key", () => {
    expect(descriptorValue(props, "desc:tpsa")).toBe(80);
    expect(descriptorValue(props, "score")).toBeUndefined();
    expect(descriptorValue(undefined, "desc:mw")).toBeUndefined();
  });
  it("writes descriptors to CSV when given", () => {
    const [c] = buildCandidates({ ...empty(), molecules: [molecule("CCO", "ethanol")] });
    const csv = candidatesToCsv([c], () => props).split("\n");
    expect(csv[0]).toContain("MW,cLogP,TPSA,HBD,HBA,RotB,Ro5");
    expect(csv[1]).toContain("520,5.5,80,2,11,4,0");
    expect(candidatesToCsv([c]).split("\n")[0]).not.toContain("MW");
  });
  it("writes an SD file, named, with properties, skipping unparsable molecules", () => {
    const candidates = buildCandidates({ ...empty(), molecules: [molecule("CCO", "ethanol"), molecule("bad")] });
    const sdf = candidatesToSdf(candidates, (s) => (s === "CCO" ? "\n  RDKit\n\n  3  2  0  0  0  0  0  0  0  0999 V2000\nM  END\n" : null), () => props);
    expect(sdf.startsWith("ethanol\n")).toBe(true);
    expect(sdf).toContain(">  <SMILES>\nCCO\n\n");
    expect(sdf).toContain(">  <MW>\n520\n\n");
    expect(sdf.match(/\$\$\$\$/g)).toHaveLength(1);
    expect(candidatesToSdf([], () => null)).toBe("");
  });
});
