// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { afterEach, describe, expect, it, vi } from "vitest";
import { bySmiles, loadJobStructures, rowMetrics, rowMolecule, runStructurePanel, type JobStructure } from "./jobStructures";
import { parseRunResults } from "./runResults";

const PDB = "/shared/predictions/out_2440b07ce52f/boltz_results_batch_2440b07ce52f/predictions/mol_0/mol_0_model_0.pdb";

const RESULTS = [
  "Score,SMILES,SMILES_state,Boltz affinity,Boltz affinity (raw),structure_path (Boltz affinity),ligand_iptm (Boltz affinity),affinity_pred_value (Boltz affinity),skip_reason (Boltz affinity),n_checks_passed (Pose quality),QED,step",
  `0.37,CCO,1,0.08,-0.58,${PDB},0.87,-0.58,,19,0.4,3`,
  "0.10,CCN,1,,,,,,,,0.5,3",
].join("\n");

const entry = (smiles: string, path = PDB): JobStructure => ({ smiles, step: 3, score: 0.5, path, metrics: { ligandIptm: 0.8 } });

describe("rowMetrics", () => {
  it("reads the Boltz and PoseBusters values of a row", () => {
    const table = parseRunResults(RESULTS);
    expect(rowMetrics(table, table.rows[0])).toEqual({ ligandIptm: 0.87, affinity: -0.58, posebustersPassed: 19 });
    expect(rowMetrics(table, table.rows[1])).toBeUndefined();
  });
});

describe("rowMolecule", () => {
  // As REINVENT writes a TTP whose Boltz objective is named "Binding affinity", with PoseBusters as "Pose quality".
  const CSV = [
    "Score,SMILES,SMILES_state,Binding affinity,Binding affinity (raw),QED,QED (raw),Pose quality,Pose quality (raw),structure_path (Binding affinity),ligand_iptm (Binding affinity),affinity_pred_value (Binding affinity),bond_lengths (Pose quality),n_checks_passed (Pose quality),n_checks_total (Pose quality),step",
    `0.49,CCO,1,0.98,-1.66,0.29,0.29,0.86,0.86,${PDB},0.93,-1.66,True,19,22,5`,
    `0.40,CCN,1,0.50,-0.20,0.70,0.70,0.90,0.90,,0.80,-0.20,False,20,22,6`,
  ].join("\n");

  it("finds the complex in any structure_path column, not in the PoseBusters checks", () => {
    const table = parseRunResults(CSV);
    expect(table.columns[table.structureColumn].key).toBe("structure_path (Binding affinity)");
    expect(table.columns.filter((c) => c.role === "structure")).toHaveLength(1);
    expect(table.rows[0].structure).toBe(PDB);
  });

  it("gives the page the run, step, score, components and complex", () => {
    const table = parseRunResults(CSV);
    expect(rowMolecule(table, table.rows[0], "e0dceaaa", { ttpId: "egfr", ttpVersion: 1 })).toEqual({
      run: "e0dceaaa",
      step: 5,
      score: 0.49,
      components: [
        { name: "Binding affinity", value: 0.98, raw: -1.66, cofold: true },
        { name: "QED", value: 0.29, raw: 0.29 },
        { name: "Pose quality", value: 0.86, raw: 0.86, cofold: true },
      ],
      structure: { path: PDB, metrics: { ligandIptm: 0.93, affinity: -1.66, posebustersPassed: 19, posebustersTotal: 22 } },
      ttpId: "egfr",
      ttpVersion: 1,
    });
    expect(rowMolecule(table, table.rows[1], "e0dceaaa").structure).toBeNull();
  });
});

describe("runStructurePanel", () => {
  it("titles the complex by run and step and keeps its metrics", () => {
    const panel = runStructurePanel(PDB, "CCO", "3f041625", 3, { ligandIptm: 0.87 });
    expect(panel).toMatchObject({
      path: "predictions/out_2440b07ce52f/boltz_results_batch_2440b07ce52f/predictions/mol_0/mol_0_model_0.pdb",
      format: "pdb",
      smiles: "CCO",
      label: "Run 3f041625 · step 3",
      metrics: { ligandIptm: 0.87 },
    });
    expect(panel?.url.startsWith("/api/artifacts/predictions/out_2440b07ce52f/")).toBe(true);
  });

  it("leaves out empty metrics and refuses files off the shared volume", () => {
    expect(runStructurePanel(PDB, null, "abc", null, {})).not.toHaveProperty("metrics");
    expect(runStructurePanel(PDB, null, "abc", null)?.label).toBe("Run abc");
    expect(runStructurePanel("/etc/passwd", null, "abc", 1)).toBeNull();
  });
});

describe("loadJobStructures", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("asks the gateway once per run unless the run is still going", async () => {
    const fetch = vi.fn(async () => new Response(JSON.stringify([entry("CCO")]), { status: 200 }));
    vi.stubGlobal("fetch", fetch);
    const first = await loadJobStructures("aaa111");
    expect(first.get("CCO")?.path).toBe(PDB);
    await loadJobStructures("aaa111");
    expect(fetch).toHaveBeenCalledTimes(1);
    await loadJobStructures("aaa111", true);
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it("tries again after a failure", async () => {
    const fetch = vi.fn(async () => new Response("{}", { status: 503 }));
    vi.stubGlobal("fetch", fetch);
    await expect(loadJobStructures("bbb222")).rejects.toThrow();
    await expect(loadJobStructures("bbb222")).rejects.toThrow();
    expect(fetch).toHaveBeenCalledTimes(2);
  });
});

describe("bySmiles", () => {
  it("indexes the entries by SMILES", () => {
    expect([...bySmiles([entry("CCO"), entry("CCN")]).keys()]).toEqual(["CCO", "CCN"]);
  });
});
