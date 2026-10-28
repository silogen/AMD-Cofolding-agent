// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it, vi } from "vitest";
import { gate } from "./gate";
import { moleculeView, onMoleculeRequest, openMolecule, originLabel } from "./moleculePage";

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
});
