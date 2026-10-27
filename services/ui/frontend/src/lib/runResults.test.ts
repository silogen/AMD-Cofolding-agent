// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import {
  defaultColumns,
  filterRows,
  formatCell,
  parseCell,
  parseCsv,
  parseRunResults,
  resultsSdf,
  resultsSheet,
  rowValues,
  runSeeds,
  sortRows,
  structurePanel,
  type Filter,
} from "./runResults";

const MOL2MOL = [
  "Agent,Prior,Target,Score,SMILES,SMILES_state,Input_SMILES,Scaffold,Aqueous solubility,Aqueous solubility (raw),std (Aqueous solubility),ad_pass (Aqueous solubility),step",
  "4.1601,4.1601,71.5332,0.5913540,O=C(O)c1ccc(-c2ccc(Cl)cc2)cc1,1,O=C(O)c1ccc(-c2ccncc2)cc1,c1ccc(-c2ccccc2)cc1,0.5913540,-4.4037,0.3816,True,1",
  "4.5387,4.5387,123.1935,0.9979079,O=C(O)c1ccc(Cc2ccncc2)cc1,1,O=C(O)c1ccc(-c2ccncc2)cc1,c1ccc(Cc2ccncc2)cc1,0.9979079,-2.8929,0.1627,True,1",
  "16.7649,40.6634,-40.6634,0.0000000,COC1Nc2nccc(Cc3ccc(C(=O)O)cc3)n2C1,0,O=C(O)c1ccc(-c2ccncc2)cc1,--,0.0000000,0.0000,None,None,2",
  "5.1881,5.1881,98.6096,0.8109202,O=C(O)c1ccc(Cc2ccncc2)cc1,1,O=C(O)c1ccc(-c2ccncc2)cc1,c1ccc(Cc2ccncc2)cc1,0.8109202,-4.1206,0.3054,True,2",
  "5.1881,5.1881,98.6",
].join("\n");

const BOLTZ = [
  'SMILES,total_score,SMILES_state,Boltz affinity,Boltz affinity (raw),structure_path (Boltz affinity),QED,Step',
  'CCO,"0.71",1,0.8,-1.2,/shared/boltz2/run_1/mol_0_model_0.cif,0.4,0',
  'CCN,nan,1,NaN,,/shared/boltz2/run_1/mol_1_model_0.cif,0.5,0',
  'c1ccccc1,0.33,1,,,"",0.44,1',
].join("\r\n");

const ALL: Filter = { query: "", minScore: null, showFailed: true, unique: false };

describe("parseCsv", () => {
  it("handles quotes, escaped quotes, CRLF and embedded line breaks", () => {
    expect(parseCsv('a,b\r\n"x, y","say ""hi"""\n"two\nlines",3\n')).toEqual([
      ["a", "b"],
      ["x, y", 'say "hi"'],
      ["two\nlines", "3"],
    ]);
  });
  it("skips blank lines", () => {
    expect(parseCsv("a,b\n\n1,2")).toEqual([
      ["a", "b"],
      ["1", "2"],
    ]);
  });
});

describe("parseCell", () => {
  it("reads numbers written as text, NaN variants and booleans", () => {
    expect(parseCell("0.5913540")).toBe(0.591354);
    expect(parseCell(" -4.4 ")).toBe(-4.4);
    expect(parseCell("1e-3")).toBe(0.001);
    for (const missing of ["", "nan", "NaN", "None", "--", "null"]) expect(parseCell(missing)).toBeNull();
    expect(parseCell("True")).toBe(true);
    expect(parseCell("False")).toBe(false);
    expect(parseCell("CCO")).toBe("CCO");
    expect(parseCell("Infinity")).toBe("Infinity");
  });
});

describe("parseRunResults (REINVENT 4, Mol2Mol)", () => {
  const table = parseRunResults(MOL2MOL);

  it("drops a half-written last line", () => {
    expect(table.rows).toHaveLength(4);
  });

  it("classifies the columns", () => {
    const role = (key: string) => table.columns.find((c) => c.key === key);
    expect(role("Score")).toMatchObject({ role: "total", label: "Score" });
    expect(role("Agent")).toMatchObject({ role: "likelihood", label: "Agent NLL" });
    expect(role("SMILES_state")?.role).toBe("state");
    expect(role("step")).toMatchObject({ role: "step", integer: true });
    expect(role("Score")?.integer).toBe(false);
    expect(role("Input_SMILES")?.role).toBe("text");
    expect(role("Scaffold")?.role).toBe("text");
    expect(role("Aqueous solubility")?.role).toBe("component");
    expect(role("Aqueous solubility (raw)")).toMatchObject({ role: "raw", component: "Aqueous solubility" });
    expect(role("std (Aqueous solubility)")).toMatchObject({ role: "meta", component: "Aqueous solubility" });
    expect(role("ad_pass (Aqueous solubility)")).toMatchObject({ role: "meta", numeric: false });
    expect(table.components).toEqual(["Aqueous solubility"]);
    expect(table.structureColumn).toBe(-1);
  });

  it("marks invalid SMILES as failed", () => {
    expect(table.failed).toBe(1);
    expect(table.rows[2]).toMatchObject({ failure: "invalid SMILES", total: 0, step: 2, state: 0 });
    expect(table.rows[0]).toMatchObject({ failure: null, total: 0.591354, step: 1, smiles: "O=C(O)c1ccc(-c2ccc(Cl)cc2)cc1" });
  });

  it("shows score, step and components by default", () => {
    expect([...defaultColumns(table)]).toEqual(["Score", "Aqueous solubility", "Aqueous solubility (raw)", "step"]);
  });
});

describe("seeds and row values", () => {
  const table = parseRunResults(MOL2MOL);

  it("reads the Mol2Mol seeds once each", () => {
    expect(runSeeds(table)).toEqual(["O=C(O)c1ccc(-c2ccncc2)cc1"]);
    expect(runSeeds(parseRunResults(BOLTZ))).toEqual([]);
  });

  it("maps a row's scores by header", () => {
    expect(rowValues(table, table.rows[0])).toEqual({ Score: 0.591354, "Aqueous solubility": 0.591354, "Aqueous solubility (raw)": -4.4037 });
  });
});

describe("parseRunResults (cofolding, older headers)", () => {
  const table = parseRunResults(BOLTZ);

  it("finds the total, the step and the structure paths", () => {
    expect(table.columns[1].role).toBe("total");
    expect(table.columns[7].role).toBe("step");
    expect(table.structureColumn).toBe(5);
    expect(table.columns[5]).toMatchObject({ role: "structure", component: "Boltz affinity" });
    expect(table.components).toEqual(["Boltz affinity", "QED"]);
    expect(table.rows[0]).toMatchObject({ total: 0.71, structure: "/shared/boltz2/run_1/mol_0_model_0.cif" });
    expect(table.rows[2].structure).toBeNull();
  });

  it("marks NaN scores and failed components", () => {
    expect(table.rows[1].failure).toBe("no score");
    expect(table.rows[2].failure).toBe("Boltz affinity failed");
    expect(table.failed).toBe(2);
  });

  it("tolerates an empty file", () => {
    expect(parseRunResults("")).toMatchObject({ rows: [], columns: [], failed: 0, structureColumn: -1 });
  });
});

describe("filter and sort", () => {
  const table = parseRunResults(MOL2MOL);
  const score = table.columns.findIndex((c) => c.key === "Score");

  it("hides failures, applies a minimum score and a text query", () => {
    expect(filterRows(table, { ...ALL, showFailed: false })).toHaveLength(3);
    expect(filterRows(table, { ...ALL, minScore: 0.8 }).map((r) => r.id)).toEqual([1, 3]);
    expect(filterRows(table, { ...ALL, query: "CL)" }).map((r) => r.id)).toEqual([0]);
    expect(filterRows(table, { ...ALL, query: "invalid" }).map((r) => r.id)).toEqual([2]);
  });

  it("keeps the best row per SMILES", () => {
    expect(filterRows(table, { ...ALL, unique: true }).map((r) => r.id)).toEqual([0, 1, 2]);
  });

  it("sorts either way with empty values last", () => {
    const boltz = parseRunResults(BOLTZ);
    expect(sortRows(boltz.rows, 1, true).map((r) => r.id)).toEqual([0, 2, 1]);
    expect(sortRows(boltz.rows, 1, false).map((r) => r.id)).toEqual([2, 0, 1]);
    expect(sortRows(table.rows, score, true).map((r) => r.id)).toEqual([1, 3, 0, 2]);
    expect(sortRows(table.rows, "smiles", false)[0].smiles).toBe("COC1Nc2nccc(Cc3ccc(C(=O)O)cc3)n2C1");
  });
});

describe("files", () => {
  const table = parseRunResults(BOLTZ);

  it("exports every column and the failure", () => {
    const sheet = resultsSheet(table, table.rows.slice(1, 2));
    expect(sheet.columns.at(-1)).toBe("failure");
    expect(sheet.rows[0]).toMatchObject({ SMILES: "CCN", total_score: null, failure: "no score" });
  });

  it("writes SD records with the scores", () => {
    const sdf = resultsSdf(table, table.rows.slice(0, 1), () => "x\n  RDKit\n\n  0  0  0  0  0  0  0  0  0  0999 V2000\nM  END\n");
    expect(sdf).toContain("molecule 1");
    expect(sdf).toContain("> <total_score>\n0.71");
    expect(sdf.trim().endsWith("$$$$")).toBe(true);
  });

  it("turns a shared structure path into a viewer panel", () => {
    expect(structurePanel("/shared/boltz2/run 1/a.cif", "CCO")).toEqual({
      kind: "structure",
      tool: "reinvent",
      name: "a.cif",
      path: "boltz2/run 1/a.cif",
      url: "/api/artifacts/boltz2/run%201/a.cif",
      format: "cif",
      smiles: "CCO",
    });
    expect(structurePanel("/etc/passwd.pdb", null)).toBeNull();
    expect(structurePanel("/shared/x.txt", null)).toBeNull();
  });

  it("formats numbers compactly", () => {
    expect(formatCell(0.591354)).toBe("0.591");
    expect(formatCell(123.1935)).toBe("123.2");
    expect(formatCell(0.00001)).toBe("1.00e-5");
    expect(formatCell(3, true)).toBe("3");
    expect(formatCell(1)).toBe("1.000");
    expect(formatCell(0)).toBe("0.000");
    expect(formatCell(null)).toBe("—");
    expect(formatCell(true)).toBe("yes");
  });
});
