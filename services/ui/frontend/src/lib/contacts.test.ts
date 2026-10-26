// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import type { Contact } from "../types/api";
import { contactLines, matchesTypes, toggleType } from "./contacts";

const contact = (over: Partial<Contact>): Contact => ({
  type: "hydrophobic",
  subtype: null,
  chain: "A",
  resname: "LEU",
  resnum: 49,
  label: "LEU49",
  receptorAtom: "CD1",
  ligandAtom: "C36",
  ligandAtomIndex: 1,
  distance: 3.65,
  fragment: null,
  confidence: null,
  ...over,
});

describe("toggleType", () => {
  it("selects, adds and removes", () => {
    expect(toggleType([], "hbond")).toEqual(["hbond"]);
    expect(toggleType(["hbond"], "hydrophobic")).toEqual(["hbond", "hydrophobic"]);
    expect(toggleType(["hbond", "hydrophobic"], "hbond")).toEqual(["hydrophobic"]);
  });

  it("shows every type when none is selected", () => {
    expect(matchesTypes(contact({}), [])).toBe(true);
    expect(matchesTypes(contact({}), ["hbond"])).toBe(false);
  });
});

describe("contactLines", () => {
  it("draws one line per atom pair, the more specific type first", () => {
    const lines = contactLines([contact({}), contact({ type: "hbond", receptorAtom: "ND2", resnum: 140, distance: 2.8 }), contact({ type: "ch_pi" })], "dark");
    expect(lines).toHaveLength(2);
    expect(lines[0]).toMatchObject({ resnum: 140, strong: true, label: "2.8 Å", color: "#00c2de" });
    expect(lines[1]).toMatchObject({ resnum: 49, strong: false, color: "#a3a3a3" });
  });

  it("labels only specific contacts when there are many", () => {
    const many = Array.from({ length: 12 }, (_, i) => contact({ ligandAtom: `C${i}` }));
    const lines = contactLines([...many, contact({ type: "hbond", ligandAtom: "N30", distance: 3.1 })], "light");
    expect(lines.filter((line) => line.label)).toHaveLength(1);
    expect(lines.find((line) => !line.strong)?.color).toBe("#a38a3e");
  });

  it("skips contacts without a residue", () => {
    expect(contactLines([contact({ resnum: null })], "dark")).toEqual([]);
  });
});
