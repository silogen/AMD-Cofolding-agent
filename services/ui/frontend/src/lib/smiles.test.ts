// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { findSmiles, looksLikeSmiles, replaceToken } from "./smiles";

describe("looksLikeSmiles", () => {
  it.each(["CC(=O)Oc1ccccc1C(=O)O", "CCO", "c1ccccc1", "C[C@H](N)C(=O)O", "[Na+].[Cl-]", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "C%10CCCCC%10"])(
    "accepts %s",
    (token) => expect(looksLikeSmiles(token)).toBe(true),
  );

  it.each(["BRD4", "JQ1", "ADMET", "Boltz-2", "CO2", "ICON", "SOS", "cocoon", "son", "C(C", "CC)C", "[]C", "/shared/x.pdb", "PDB"])(
    "rejects %s",
    (token) => expect(looksLikeSmiles(token)).toBe(false),
  );

  it("keeps an unclosed ring in a longer token, for RDKit to flag", () => {
    expect(looksLikeSmiles("c1ccccc")).toBe(true);
  });
});

describe("findSmiles", () => {
  it("finds SMILES among words and punctuation", () => {
    expect(findSmiles('Cofold "CC(=O)Oc1ccccc1C(=O)O" into BRD4, then compare with (CCO) and c1ccccc1.')).toEqual([
      "CC(=O)Oc1ccccc1C(=O)O",
      "CCO",
      "c1ccccc1",
    ]);
  });

  it("lists each SMILES once and keeps a trailing branch", () => {
    expect(findSmiles("CC(C) and CC(C)\nCC(C)")).toEqual(["CC(C)"]);
  });

  it("ignores slash commands and paths", () => {
    expect(findSmiles("/view pocket\nStructure: /shared/uploads/new/x.pdb")).toEqual([]);
  });
});

describe("replaceToken", () => {
  it("replaces the whole token only", () => {
    expect(replaceToken("score CCO and CCOC", "CCO", "CCN")).toBe("score CCN and CCOC");
    expect(replaceToken("(CC(C)) next", "CC(C)", "c1ccccc1")).toBe("(c1ccccc1) next");
    expect(replaceToken("CCO", "CCO", "CCN")).toBe("CCN");
  });
});
