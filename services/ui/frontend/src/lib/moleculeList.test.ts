// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";

import type { PropertiesPanel } from "../types/api";
import type { Sourced } from "./artifacts";
import { gate } from "./gate";
import type { Candidate } from "./candidates";
import {
  columnRange,
  filterMolecules,
  listColumns,
  listMolecules,
  moleculeSheet,
  originWord,
  sortMolecules,
  starValues,
  tidyNumber,
} from "./moleculeList";

const candidate = (smiles: string, extra: Partial<Candidate> = {}): Candidate => ({
  smiles,
  origin: "score_molecules",
  origins: ["score_molecules"],
  components: {},
  gate: gate(undefined),
  ...extra,
});

const admet = (molecules: { smiles: string; label?: string; logp: number }[], service: string | null = "admet"): Sourced<PropertiesPanel> => ({
  tool: "predict_admet",
  at: 1,
  item: {
    kind: "properties",
    tool: "predict_admet",
    service,
    target: null,
    endpoints: ["logp"],
    molecules: molecules.map((m) => ({ smiles: m.smiles, standardized: null, label: m.label ?? null, ok: true, error: null, adPass: true, values: { logp: { mean: m.logp } } })),
  },
});

/** Stands in for RDKit: "OCC" and "CCO" are ethanol. */
const key = (smiles: string) => (smiles === "OCC" ? "CCO" : smiles);

describe("listMolecules", () => {
  it("lists a molecule reported by several tools once, by canonical SMILES", () => {
    const rows = listMolecules({
      candidates: [candidate("CCO", { name: "ethanol", score: 0.8, metrics: { confidence: 0.9, affinity: -1.2 }, origins: ["score_molecules", "boltz_cofold"] })],
      properties: [admet([{ smiles: "OCC", logp: -0.3 }])],
      starred: [{ smiles: "OCC", at: 1 }],
      key,
    });
    expect(rows).toHaveLength(1);
    const [row] = rows;
    expect(row.smiles).toBe("CCO");
    expect(row.name).toBe("ethanol");
    expect(row.starred).toBe(true);
    expect(row.origins).toEqual(["Cofolded", "Scored", "ADMET"]);
    expect(row.tools).toEqual(["score_molecules", "boltz_cofold", "predict_admet"]);
    expect(row.values).toEqual({ score: 0.8, "metric:confidence": 0.9, "metric:affinity": -1.2, "p:ADMET:logp": -0.3 });
  });

  it("keeps the newest value: candidates and predictions come newest first, stars fill the gaps", () => {
    const [row] = listMolecules({
      candidates: [candidate("CCO", { components: { QED: 0.4 } })],
      properties: [admet([{ smiles: "CCO", logp: 1 }]), admet([{ smiles: "CCO", logp: 2 }])],
      starred: [{ smiles: "CCO", at: 1, values: { Score: 0.5, QED: 0.9, docking: -7 } }],
    });
    expect(row.values.score).toBe(0.5);
    expect(row.values["p:ADMET:logp"]).toBe(1);
    expect(row.values["c:docking"]).toBe(-7);
    // QED is a property, not a score component.
    expect(row.values["c:QED"]).toBeUndefined();
  });

  it("lists starred molecules nothing else reported, with the run they came from", () => {
    const rows = listMolecules({ candidates: [], properties: [], starred: [{ smiles: "c1ccccc1", label: "benzene", source: "e0dceaaa", at: 1 }] });
    expect(rows).toEqual([
      { key: "c1ccccc1", smiles: "c1ccccc1", name: "benzene", origins: ["Run e0dceaaa"], tools: [], starred: true, run: "e0dceaaa", values: {} },
    ]);
  });

  it("adds descriptors and skips failed predictions", () => {
    const failed = admet([{ smiles: "CCN", logp: 3 }]);
    failed.item.molecules[0].ok = false;
    const rows = listMolecules({ candidates: [candidate("CCO")], properties: [failed], starred: [], descriptors: () => ({ mw: 46.07, clogp: -0.0014, tpsa: 20.23 }) });
    expect(rows.map((r) => r.smiles)).toEqual(["CCO"]);
    expect(rows[0].values["desc:mw"]).toBe(46.07);
  });
});

describe("listColumns", () => {
  it("has a column only for what some row knows, score and cofolding first", () => {
    const properties = [admet([{ smiles: "CCO", logp: 1 }])];
    const rows = listMolecules({ candidates: [candidate("CCO", { score: 0.7, metrics: { confidence: 0.8 } }), candidate("CCN")], properties, starred: [] });
    expect(listColumns(rows, properties).map((c) => c.key)).toEqual(["score", "metric:confidence", "p:ADMET:logp"]);
  });

  it("keeps the three most common score components", () => {
    const rows = listMolecules({
      candidates: [candidate("A", { components: { a: 1, b: 1, c: 1, d: 1 } }), candidate("B", { components: { b: 1, c: 1, d: 1 } }), candidate("C", { components: { d: 1 } })],
      properties: [],
      starred: [],
    });
    expect(listColumns(rows).map((c) => c.key)).toEqual(["c:d", "c:b", "c:c"]);
  });
});

describe("filtering and sorting", () => {
  const rows = listMolecules({
    candidates: [candidate("CCO", { name: "ethanol", score: 0.2 }), candidate("CCN", { score: 0.9 }), candidate("CCC")],
    properties: [],
    starred: [{ smiles: "CCN", at: 1 }],
  });

  it("searches name, SMILES and origin, and keeps the starred ones on request", () => {
    expect(filterMolecules(rows, "ETHA", false).map((r) => r.smiles)).toEqual(["CCO"]);
    expect(filterMolecules(rows, "scored", false)).toHaveLength(3);
    expect(filterMolecules(rows, "", true).map((r) => r.smiles)).toEqual(["CCN"]);
  });

  it("sorts by a column with missing values last either way", () => {
    expect(sortMolecules(rows, "score", true).map((r) => r.smiles)).toEqual(["CCN", "CCO", "CCC"]);
    expect(sortMolecules(rows, "score", false).map((r) => r.smiles)).toEqual(["CCO", "CCN", "CCC"]);
    expect(sortMolecules(rows, null, true).map((r) => r.smiles)).toEqual(["CCO", "CCN", "CCC"]);
    expect(columnRange(rows, "score")).toEqual([0.2, 0.9]);
    expect(columnRange(rows.slice(2), "score")).toBeNull();
  });
});

describe("helpers", () => {
  it("names origins in a word", () => {
    expect(originWord("boltz_cofold")).toBe("Cofolded");
    expect(originWord("run_reinvent")).toBe("Generated");
    expect(originWord("pubchem_lookup")).toBe("Looked up");
    expect(originWord("my_tool")).toBe("my tool");
  });

  it("formats numbers by their size", () => {
    expect([tidyNumber(0.8234), tidyNumber(12.345), tidyNumber(412.7), tidyNumber(3)]).toEqual(["0.82", "12.3", "413", "3"]);
  });

  it("exports the rows and what a star keeps", () => {
    const [row] = listMolecules({ candidates: [candidate("CCO", { name: "ethanol", score: 0.8, metrics: { ligandIptm: 0.6 }, components: { docking: -7 } })], properties: [], starred: [] });
    const sheet = moleculeSheet([row], listColumns([row]));
    expect(sheet.columns).toEqual(["smiles", "name", "origin", "starred", "Score", "Ligand ipTM", "docking"]);
    expect(sheet.rows[0]).toMatchObject({ smiles: "CCO", name: "ethanol", origin: "Cofolded; Scored", Score: 0.8, "Ligand ipTM": 0.6, docking: -7 });
    expect(starValues(row)).toEqual({ Score: 0.8, ligand_iptm: 0.6, docking: -7 });
  });
});
