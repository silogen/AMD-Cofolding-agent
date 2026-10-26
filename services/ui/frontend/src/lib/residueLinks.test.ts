// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { residueFromHref, residueHref, splitResidues } from "./residueLinks";

describe("splitResidues", () => {
  it("finds residues in prose", () => {
    expect(splitResidues("the ASN140 hydrogen bond and a Trp 81.A stack")).toEqual([
      { text: "the " },
      { text: "ASN140", residue: { chain: null, resnum: 140, resname: "ASN" } },
      { text: " hydrogen bond and a " },
      { text: "Trp 81.A", residue: { chain: "A", resnum: 81, resname: "TRP" } },
      { text: " stack" },
    ]);
  });

  it("ignores lower case and words that only look alike", () => {
    expect(splitResidues("asn140, Alanine 5, METHOD 2")).toEqual([{ text: "asn140, Alanine 5, METHOD 2" }]);
  });
});

describe("residue hrefs", () => {
  it("round-trips", () => {
    expect(residueFromHref(residueHref({ chain: "B", resnum: 92 }))).toEqual({ chain: "B", resnum: 92 });
    expect(residueFromHref(residueHref({ chain: null, resnum: 140 }))).toEqual({ chain: null, resnum: 140 });
    expect(residueFromHref(residueHref({ chain: "A", resnum: 140, resname: "ASN" }))).toEqual({ chain: "A", resnum: 140, resname: "ASN" });
    expect(residueFromHref("https://example.org")).toBeNull();
  });
});
