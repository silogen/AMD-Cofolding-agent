// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { pickKind, pickQuestion, pickTitle, structureRef, type PickedResidue } from "./pickQuestion";

const pick = (over: Partial<PickedResidue>): PickedResidue => ({ kind: "residue", resname: "LEU", label: "LEU94", chain: "A", file: null, ...over });

describe("pickKind", () => {
  it("tells the ligand from the chain", () => {
    expect(pickKind("MI1", false)).toBe("ligand");
    expect(pickKind("LIG", false)).toBe("ligand");
    expect(pickKind("CY0", true)).toBe("adduct");
    expect(pickKind("PTR", true)).toBe("modified");
    expect(pickKind("LEU", true)).toBe("residue");
    expect(pickKind("HOH", false)).toBe("solvent");
    expect(pickKind("ZN", false)).toBe("solvent");
  });

  it("does not guess before the structure's residues are known", () => {
    expect(pickKind("CY0", null)).toBe("residue");
    expect(pickKind("LIG", null)).toBe("ligand");
  });
});

describe("pickQuestion", () => {
  const path = "uploads/s-20261028-090207-229c4e/2J5E.cif";

  it("names the file", () => {
    expect(structureRef("2J5E", path)).toBe(`2J5E (${path})`);
    expect(structureRef(path, path)).toBe(path);
    expect(pickQuestion("2J5E", path, pick({}))).toBe(
      `In 2J5E (${path}), what role does LEU94 (chain A) play in binding the ligand? Use the interaction profile.`,
    );
  });

  it("asks the ligand question for a ligand group", () => {
    const ligand = pick({ kind: "ligand", resname: "MI1", label: "MI1 1" });
    expect(pickTitle(ligand)).toBe("MI1 1");
    expect(pickQuestion("3EYG", "uploads/s-x/3EYG.cif", ligand)).toBe(
      "In 3EYG (uploads/s-x/3EYG.cif), how is the ligand MI1 1 (chain A) held in the pocket? Profile that ligand and name its key contacts.",
    );
  });

  it("calls a cofolded ligand the ligand", () => {
    const ligand = pick({ kind: "ligand", resname: "LIG", label: "LIG1", chain: "B" });
    expect(pickTitle(ligand)).toBe("the ligand");
    expect(pickQuestion("mol_0", "predictions/x/mol_0.pdb", ligand)).toBe(
      "In mol_0 (predictions/x/mol_0.pdb), how is the ligand held in the pocket? Name its key contacts from the interaction profile.",
    );
  });

  it("words a residue carrying a covalent ligand as such", () => {
    const question = pickQuestion("2J5E", path, pick({ kind: "adduct", resname: "CY0", label: "CY0 797" }));
    expect(question).toContain(`In 2J5E (${path}), CY0 797 (chain A) is a non-standard residue of the protein chain`);
    expect(question).toContain("covalently bound ligand");
    expect(question).not.toContain("CY0797");
  });

  it("asks about a modified residue and gives the file number when the label shows another", () => {
    expect(pickQuestion("3EYG", null, pick({ kind: "modified", resname: "PTR", label: "PTR1034" }))).toBe(
      "In 3EYG, what role does the modified residue PTR1034 (chain A) play near the ligand? Use the interaction profile.",
    );
    expect(pickQuestion("BRD4", null, pick({ file: 51 }))).toContain("LEU94 (chain A, residue 51 in the file)");
  });
});
