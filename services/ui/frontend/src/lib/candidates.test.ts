// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";

import type { MoleculePanel, StructurePanel, TablePanel } from "../types/api";
import type { Artifacts, Sourced } from "./artifacts";
import { buildCandidates, cofoldPrompt, MAX_COFOLD } from "./candidates";

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

describe("cofoldPrompt", () => {
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
