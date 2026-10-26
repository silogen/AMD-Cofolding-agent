// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";

import type { InteractionsPanel, StructurePanel } from "../types/api";
import { suggestFollowups } from "./followups";

const pose = (metrics: StructurePanel["metrics"]): StructurePanel => ({
  kind: "structure",
  tool: "score_molecules",
  name: "model_0.pdb",
  path: "predictions/out_1/model_0.pdb",
  url: "/api/artifacts/predictions/out_1/model_0.pdb",
  format: "pdb",
  metrics,
});

const profile: InteractionsPanel = {
  kind: "interactions",
  tool: "profile_interactions",
  name: "mol_0",
  structurePath: "predictions/out_1/model_0.pdb",
  smiles: "CCO",
  error: null,
  counts: { hydrophobic: 3 },
  contacts: ["LEU92", "ASN140", "LEU92"].map((label) => ({
    type: "hydrophobic",
    subtype: null,
    chain: "A",
    resname: label.slice(0, 3),
    resnum: Number(label.slice(3)),
    label,
    receptorAtom: null,
    ligandAtom: null,
    ligandAtomIndex: null,
    distance: 3.5,
    fragment: null,
    confidence: null,
  })),
  totalContacts: 3,
};

describe("suggestFollowups", () => {
  it("asks why a pose failed the gate, then to profile it", () => {
    expect(suggestFollowups([pose({ confidence: 0.3, ligandIptm: 0.2 })], ["score_molecules"])).toEqual([
      "Why did this pose fail the confidence gate?",
      "Profile the interactions in this pose",
    ]);
  });

  it("asks about the residue with the most contacts", () => {
    const followups = suggestFollowups([pose({ confidence: 0.9, ligandIptm: 0.9 }), profile], []);
    expect(followups[0]).toBe("Which ligand fragment makes the LEU92 contact?");
    expect(followups).toHaveLength(2);
  });

  it("suggests nothing for a discovery answer", () => {
    expect(suggestFollowups([], ["list_scoring_components"])).toEqual([]);
  });
});
