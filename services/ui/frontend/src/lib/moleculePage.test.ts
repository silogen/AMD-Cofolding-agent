// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it, vi } from "vitest";
import { gate } from "./gate";
import { moleculeView, onMoleculeRequest, openMolecule, originLabel, runRowContext } from "./moleculePage";
import { parseRunResults } from "./runResults";

describe("openMolecule", () => {
  it("hands the SMILES and what the caller knows to the workbench", () => {
    const listener = vi.fn();
    const off = onMoleculeRequest(listener);
    openMolecule("CCO", { run: "e0dceaaa" });
    openMolecule("CCN");
    off();
    openMolecule("CCC");
    expect(listener.mock.calls).toEqual([[{ smiles: "CCO", context: { run: "e0dceaaa" } }], [{ smiles: "CCN" }]]);
  });
});

describe("originLabel", () => {
  it("names the run and step, or nothing for the chat", () => {
    expect(originLabel({ run: "e0dceaaa", step: 5 })).toBe("Run e0dceaaa · step 5");
    expect(originLabel({ run: "e0dceaaa", step: null })).toBe("Run e0dceaaa");
    expect(originLabel(undefined)).toBeNull();
  });
});

describe("moleculeView", () => {
  it("shows a run row as it came, with the run's raw values as properties", () => {
    const view = moleculeView({
      smiles: "CCO",
      context: {
        run: "e0dceaaa",
        step: 5,
        score: 0.49,
        components: [
          { name: "Boltz affinity", value: 0.98, raw: -1.66, cofold: true },
          { name: "QED", value: 0.29, raw: 0.31 },
        ],
        structure: { path: "predictions/x.pdb", metrics: { ligandIptm: 0.93 } },
      },
    });
    expect(view).toMatchObject({ origin: "Run e0dceaaa · step 5", score: 0.49, cofold: { ligandIptm: 0.93 }, structurePath: "predictions/x.pdb" });
    expect(view.components.map((f) => f.label)).toEqual(["Boltz affinity", "QED"]);
    expect(view.properties).toEqual([{ label: "QED", value: "0.31" }]);
  });

  it("fills a run molecule from the run's structures when the caller only knew the run", () => {
    const view = moleculeView({
      smiles: "CCO",
      context: { run: "e0dceaaa" },
      fetched: { smiles: "CCO", step: 7, score: 0.6, path: "predictions/y.pdb", metrics: { affinity: -1 } },
    });
    expect(view).toMatchObject({ origin: "Run e0dceaaa · step 7", score: 0.6, cofold: { affinity: -1 }, structurePath: "predictions/y.pdb" });
  });

  it("reads a chat molecule from the conversation's candidates", () => {
    const view = moleculeView({
      smiles: "CCO",
      candidate: {
        smiles: "CCO",
        name: "ethanol",
        origin: "score_molecules",
        origins: ["score_molecules", "fetch_compound"],
        score: 0.7,
        components: { qed: 0.41, ligand_iptm: 0.8, docking: 0.5 },
        gate: gate(undefined),
      },
    });
    expect(view).toMatchObject({ name: "ethanol", origin: "Chat", tools: ["score_molecules", "fetch_compound"], score: 0.7, cofold: { ligandIptm: 0.8 }, structurePath: null });
    expect(view.components).toEqual([{ label: "docking", value: "0.50" }]);
    expect(view.properties).toEqual([{ label: "QED", value: "0.41" }]);
  });

  it("lists the run's endpoint values as predictions, after the conversation's, without repeating one", () => {
    const view = moleculeView({
      smiles: "CCO",
      context: {
        run: "e0dceaaa",
        components: [
          { name: "Aqueous solubility", value: 0.5, raw: -4.12 },
          { name: "hERG", value: 0.9, raw: 0.12 },
        ],
      },
      predictions: [
        {
          item: {
            kind: "properties",
            tool: "predict_admet",
            service: "admet",
            target: null,
            endpoints: ["aqueous_solubility"],
            molecules: [{ smiles: "OCC", standardized: null, label: null, ok: true, error: null, adPass: true, values: { aqueous_solubility: { mean: -4.3 } } }],
          },
          tool: "predict_admet",
          at: 1,
        },
      ],
      key: (s) => (s === "OCC" ? "CCO" : s),
    });
    expect(view.predictions.map((g) => [g.title, g.facts.map((f) => `${f.label} ${f.value}`)])).toEqual([
      ["ADMET", ["Aq. solubility −4.30"]],
      ["Run", ["hERG 0.12"]],
    ]);
    expect(view.properties).toEqual([]);
  });
});

describe("runRowContext", () => {
  const table = parseRunResults(
    [
      "Score,SMILES,SMILES_state,QED,QED (raw),Boltz affinity,Boltz affinity (raw),structure_path (Boltz affinity),ligand_iptm (Boltz affinity),step",
      "0.40,OCC,1,0.3,0.31,0.9,-1.2,/shared/predictions/a.pdb,0.8,1",
      "0.60,OCC,1,0.4,0.41,0.9,-1.5,/shared/predictions/b.pdb,0.9,3",
      "0.70,CCN,1,0.5,0.51,0.9,-1.0,/shared/predictions/c.pdb,0.7,3",
    ].join("\n"),
  );

  it("takes the best scoring row of the molecule, matched by canonical SMILES", () => {
    expect(runRowContext(table, "CCO", "e0dceaaa")).toBeNull();
    const context = runRowContext(table, "CCO", "e0dceaaa", (s) => (s === "OCC" ? "CCO" : s));
    expect(context).toMatchObject({ run: "e0dceaaa", step: 3, score: 0.6, structure: { path: "/shared/predictions/b.pdb", metrics: { ligandIptm: 0.9 } } });
    expect(context?.components?.find((c) => c.name === "QED")).toEqual({ name: "QED", value: 0.4, raw: 0.41 });
  });
});
