// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import {
  alignChain,
  buildNumbering,
  displayNumber,
  numberingFor,
  rememberNumbering,
  residueLabel,
  resolveResidue,
  type FileResidue,
} from "./numbering";

// BRD4 (O60885) residues 1-180.
const BRD4 =
  "MSAESGPGTRLRNLPVMGDGLETSQMSTTQAQAQPQPANAASTNPPPPETSNPNKPKRQTNQLQYLLKVVLKTLWKHQFAWPFQQPVDAVKLNLPDYYKIIKTPMDMGTIKKRLENNYYWNAQECIQDFNTMFTNCYIYNKPGDDIVLMAEALEKLFLQKINELPTEETEIMIVQAKGRG";

const THREE: Record<string, string> = {
  A: "ALA", R: "ARG", N: "ASN", D: "ASP", C: "CYS", Q: "GLN", E: "GLU", G: "GLY", H: "HIS", I: "ILE",
  L: "LEU", K: "LYS", M: "MET", F: "PHE", P: "PRO", S: "SER", T: "THR", W: "TRP", Y: "TYR", V: "VAL",
};

/** Residues of chain A for *sequence*, numbered from *start* unless *numbers* says otherwise. */
function residues(sequence: string, start = 1, numbers?: number[]): FileResidue[] {
  return [...sequence].map((letter, i) => ({ chain: "A", resnum: numbers ? numbers[i] : start + i, resname: THREE[letter] }));
}

describe("alignChain", () => {
  it("places an exact fragment", () => {
    expect(alignChain("NPPPPETSNP", BRD4)).toEqual([...Array(10).keys()].map((i) => 43 + i));
  });

  it("returns null for another protein", () => {
    expect(alignChain("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQ", BRD4)).toBeNull();
  });

  it("keeps point mutations on the diagonal", () => {
    const fragment = BRD4.slice(43, 168);
    const mutated = `${fragment.slice(0, 50)}A${fragment.slice(51)}`;
    const positions = alignChain(mutated, BRD4);
    expect(positions?.[50]).toBe(93);
    expect(positions?.every((p, i) => p === 43 + i)).toBe(true);
  });

  it("leaves a tag that is not in the protein unmapped", () => {
    const tagged = `MHHHHHHSSGVDLGT${BRD4.slice(43, 168)}`;
    const positions = alignChain(tagged, BRD4);
    expect(positions?.[0]).toBeNull();
    expect(positions?.[15]).toBe(43);
    expect(positions?.at(-1)).toBe(167);
  });

  it("follows a deletion", () => {
    const deleted = BRD4.slice(43, 100) + BRD4.slice(110, 168);
    const positions = alignChain(deleted, BRD4);
    expect(positions?.[50]).toBe(93);
    expect(positions?.[60]).toBe(113);
  });
});

describe("buildNumbering", () => {
  it("maps a Boltz file numbered from 1", () => {
    const n = buildNumbering(residues(BRD4.slice(43, 168)), BRD4);
    expect(n?.same).toBe(false);
    expect(displayNumber(n, "A", 51)).toBe(94);
    expect(displayNumber(n, null, 1)).toBe(44);
    expect(residueLabel(n, "LEU", "A", 51)).toEqual({ label: "LEU94", file: 51 });
  });

  it("knows a file already in UniProt numbering", () => {
    const n = buildNumbering(residues(BRD4.slice(43, 168), 44), BRD4);
    expect(n?.same).toBe(true);
    expect(residueLabel(n, "ASN", "A", 140)).toEqual({ label: "ASN140", file: null });
  });

  it("follows the file's own numbers across a missing loop", () => {
    const numbers = [...Array(40).keys()].map((i) => 44 + i).concat([...Array(40).keys()].map((i) => 100 + i));
    const sequence = BRD4.slice(43, 83) + BRD4.slice(99, 139);
    const n = buildNumbering(residues(sequence, 0, numbers), BRD4);
    expect(n?.same).toBe(true);
    expect(displayNumber(n, "A", 100)).toBe(100);
  });

  it("is null without a sequence or a match", () => {
    expect(buildNumbering(residues(BRD4.slice(43, 168)), null)).toBeNull();
    expect(buildNumbering(residues("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKV"), BRD4)).toBeNull();
  });

  it("ignores ligands and maps only the matching chain", () => {
    const list = [...residues(BRD4.slice(43, 168)), { chain: "B", resnum: 1, resname: "LIG" }];
    const n = buildNumbering(list, BRD4);
    expect(displayNumber(n, "B", 1)).toBe(1);
  });
});

describe("resolveResidue", () => {
  const n = buildNumbering(residues(BRD4.slice(43, 168)), BRD4);

  it("takes a name and number that match the file as written", () => {
    expect(resolveResidue(n, { chain: "A", resnum: 51, resname: "LEU" })).toBe(51);
  });

  it("reads a UniProt name and number", () => {
    expect(resolveResidue(n, { chain: null, resnum: 140, resname: "ASN" })).toBe(97);
    expect(resolveResidue(n, { resnum: 94, resname: "Leu" })).toBe(51);
  });

  it("reads a bare number as UniProt", () => {
    expect(resolveResidue(n, { resnum: 140 })).toBe(97);
    expect(resolveResidue(null, { resnum: 140 })).toBe(140);
  });
});

describe("numberingFor", () => {
  it("finds a numbering by either form of the path", () => {
    const n = buildNumbering(residues(BRD4.slice(43, 168)), BRD4);
    rememberNumbering("/shared/predictions/out_x/mol_0_model_0.pdb", n);
    expect(numberingFor("predictions/out_x/mol_0_model_0.pdb")).toBe(n);
    expect(numberingFor("other.pdb")).toBeNull();
  });
});
