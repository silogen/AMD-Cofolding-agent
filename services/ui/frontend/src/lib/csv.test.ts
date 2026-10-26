// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import type { PropertiesPanel } from "../types/api";
import { candidatesSdf, candidatesSheet, csvCell, fileName, propertiesSheet, toCsv, toSdf, toTsv } from "./csv";

describe("csvCell", () => {
  it("leaves plain values alone and blanks missing ones", () => {
    expect(csvCell("CCO")).toBe("CCO");
    expect(csvCell(1.5)).toBe("1.5");
    expect(csvCell(-2)).toBe("-2");
    expect(csvCell(true)).toBe("true");
    expect(csvCell(null)).toBe("");
    expect(csvCell(undefined)).toBe("");
    expect(csvCell(Number.NaN)).toBe("");
  });

  it("quotes commas, quotes, line breaks and edge spaces", () => {
    expect(csvCell("a,b")).toBe('"a,b"');
    expect(csvCell('say "hi"')).toBe('"say ""hi"""');
    expect(csvCell("two\nlines")).toBe('"two\nlines"');
    expect(csvCell(" padded")).toBe('" padded"');
  });
});

describe("toCsv", () => {
  it("writes a header and the rows in column order, CRLF-separated", () => {
    const csv = toCsv(["smiles", "score"], [
      { score: 0.5, smiles: "CCO" },
      { smiles: "C(=O)O,Na", score: null },
    ]);
    expect(csv).toBe('smiles,score\r\nCCO,0.5\r\n"C(=O)O,Na",\r\n');
  });

  it("quotes awkward column names too", () => {
    expect(toCsv(["a,b"], [])).toBe('"a,b"\r\n');
  });
});

describe("toTsv", () => {
  it("separates cells with tabs and flattens tabs and line breaks", () => {
    expect(toTsv(["a", "b"], [{ a: "x\ty", b: "1\n2" }, { a: null, b: 3 }])).toBe("a\tb\nx y\t1 2\n\t3");
  });
});

describe("fileName", () => {
  it("slugs the base name", () => {
    expect(fileName("BRD4 top candidates", "csv")).toBe("brd4-top-candidates.csv");
    expect(fileName("  ", "sdf")).toBe("table.sdf");
  });
});

const panel: PropertiesPanel = {
  kind: "properties",
  tool: "predict_admet",
  service: "admet",
  target: null,
  endpoints: ["logp", "caco2_permeability"],
  molecules: [
    {
      smiles: "CCO",
      standardized: "CCO",
      label: "ethanol",
      ok: true,
      error: null,
      adPass: true,
      values: { logp: { mean: -0.1, std: 0.2, unit: "logP (dimensionless)" }, caco2_permeability: { mean: -4.6, unit: "log10(cm/s)" } },
    },
    { smiles: "XX", standardized: null, label: null, ok: false, error: "invalid SMILES", adPass: null, values: {} },
  ],
};

describe("propertiesSheet", () => {
  it("has mean and std per endpoint, with units in the header", () => {
    const sheet = propertiesSheet(panel);
    expect(sheet.columns).toEqual([
      "smiles",
      "label",
      "logp (logP (dimensionless))",
      "logp_std",
      "caco2_permeability (log10(cm/s))",
      "caco2_permeability_std",
      "ad_pass",
      "error",
    ]);
    expect(sheet.rows[0]).toMatchObject({ smiles: "CCO", "logp (logP (dimensionless))": -0.1, logp_std: 0.2, "caco2_permeability (log10(cm/s))": -4.6, ad_pass: true });
    expect(sheet.rows[1]).toMatchObject({ smiles: "XX", error: "invalid SMILES" });
    expect(toCsv(sheet.columns, sheet.rows).split("\r\n")[2]).toBe("XX,,,,,,,invalid SMILES");
  });
});

describe("candidatesSheet", () => {
  it("ranks candidates and lists transformed and raw component scores", () => {
    const sheet = candidatesSheet([
      { smiles: "CCN", total: 0.8, components: [{ name: "QED", transformed: 0.7, raw: 0.7 }, { name: "MW", transformed: 1, raw: 45 }] },
      { smiles: "CCC", total: 0.5, components: [{ name: "QED", transformed: 0.4, raw: 0.4 }] },
    ]);
    expect(sheet.columns).toEqual(["rank", "smiles", "total_score", "QED", "QED (raw)", "MW", "MW (raw)"]);
    expect(sheet.rows[0]).toEqual({ rank: 1, smiles: "CCN", total_score: 0.8, QED: 0.7, "QED (raw)": 0.7, MW: 1, "MW (raw)": 45 });
    expect(sheet.rows[1].MW).toBeUndefined();
  });
});

describe("toSdf", () => {
  const molblock = "\n     RDKit          2D\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\n    0.0000    0.0000    0.0000 C   0  0\nM  END\n";

  it("titles each record, appends data fields and terminates with $$$$", () => {
    const sdf = toSdf([{ molblock, title: "CCO", fields: { total_score: 0.8, note: null } }]);
    const lines = sdf.split("\n");
    expect(lines[0]).toBe("CCO");
    expect(sdf).toContain("M  END\n> <total_score>\n0.8\n\n$$$$\n");
    expect(sdf).not.toContain("note");
  });

  it("skips molecules without a molblock", () => {
    expect(toSdf([{ molblock: null, fields: {} }, { molblock, fields: {} }]).match(/\$\$\$\$/g)).toHaveLength(1);
  });

  it("writes candidates with their rank and scores", () => {
    const sdf = candidatesSdf(
      [
        { smiles: "C", total: 0.9, components: [{ name: "QED", transformed: 0.5, raw: 0.5 }] },
        { smiles: "bad", total: 0.1, components: [] },
      ],
      (smiles) => (smiles === "C" ? molblock : null),
    );
    expect(sdf.startsWith("candidate 1\n")).toBe(true);
    expect(sdf).toContain("> <rank>\n1\n\n> <smiles>\nC\n\n> <total_score>\n0.9\n\n> <QED>\n0.5\n");
    expect(sdf.match(/\$\$\$\$/g)).toHaveLength(1);
  });
});
