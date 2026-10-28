// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import type { FileResidue } from "./numbering";
import { locateResidue } from "./residueTarget";

const RESIDUES: FileResidue[] = [
  { chain: "A", resnum: 140, resname: "ASN" },
  { chain: "A", resnum: 81, resname: "TRP" },
  { chain: "A", resnum: 52, resname: "GLY" },
  { chain: "A", resnum: 52, resname: "SER", insCode: "A" },
  { chain: "B", resnum: 140, resname: "ASN" },
  { chain: "A", resnum: 790, resname: "HIE" },
];

describe("locateResidue", () => {
  it("finds a residue in every chain that has it", () => {
    expect(locateResidue(RESIDUES, null, { chain: null, resnum: 140, resname: "ASN" })).toEqual([
      { chain: "A", resnum: 140, insCode: null },
      { chain: "B", resnum: 140, insCode: null },
    ]);
    expect(locateResidue(RESIDUES, null, { chain: "B", resnum: 140, resname: "ASN" })).toEqual([{ chain: "B", resnum: 140, insCode: null }]);
  });

  it("is null for a residue of another type or number", () => {
    expect(locateResidue(RESIDUES, null, { chain: null, resnum: 81, resname: "LEU" })).toBeNull();
    expect(locateResidue(RESIDUES, null, { chain: null, resnum: 9999, resname: "LEU" })).toBeNull();
  });

  it("drops a chain the file lacks, and falls back from a wrong chain", () => {
    expect(locateResidue(RESIDUES, null, { chain: "C", resnum: 81, resname: "TRP" })).toEqual([{ chain: "A", resnum: 81, insCode: null }]);
    expect(locateResidue(RESIDUES, null, { chain: "B", resnum: 81, resname: "TRP" })).toEqual([{ chain: "A", resnum: 81, insCode: null }]);
  });

  it("reads insertion codes, or a trailing letter as the chain", () => {
    expect(locateResidue(RESIDUES, null, { chain: null, resnum: 52, resname: "SER", insCode: "A" })).toEqual([
      { chain: "A", resnum: 52, insCode: "A" },
    ]);
    expect(locateResidue(RESIDUES, null, { chain: null, resnum: 140, resname: "ASN", insCode: "B" })).toEqual([
      { chain: "B", resnum: 140, insCode: null },
    ]);
  });

  it("accepts variants of the same amino acid", () => {
    expect(locateResidue(RESIDUES, null, { chain: null, resnum: 790, resname: "HIS" })).toEqual([{ chain: "A", resnum: 790, insCode: null }]);
  });

  it("leaves hetero groups to the viewer", () => {
    expect(locateResidue(RESIDUES, null, { chain: null, resnum: 797, resname: "CY0" })).toEqual([{ chain: null, resnum: 797, insCode: null }]);
    expect(locateResidue([], null, { chain: "A", resnum: 5, resname: "ALA" })).toEqual([{ chain: "A", resnum: 5, insCode: null }]);
  });
});
