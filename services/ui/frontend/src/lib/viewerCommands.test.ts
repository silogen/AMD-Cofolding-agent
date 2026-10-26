// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { parsePickLabel } from "./molstar";
import { parseResidues, parseViewerCommand } from "./viewerCommands";

describe("parseViewerCommand", () => {
  it.each([
    ["fit", { kind: "fit" }],
    ["Reset view", { kind: "fit" }],
    ["focus ligand", { kind: "ligand" }],
    ["zoom on the ligand", { kind: "ligand" }],
    ["pocket", { kind: "pocket", radius: 5 }],
    ["show binding site within 8 Å", { kind: "pocket", radius: 8 }],
    ["show hbonds", { kind: "contacts", type: "hbond" }],
    ["hydrogen bonds", { kind: "contacts", type: "hbond" }],
    ["show contacts", { kind: "contacts", type: null }],
    ["salt bridges", { kind: "contacts", type: "ionic" }],
    ["surface", { kind: "style", style: "surface" }],
    ["show as ball and stick", { kind: "style", style: "sticks" }],
    ["color by chain", { kind: "color", color: "chain" }],
    ["colour by pLDDT", { kind: "color", color: "confidence" }],
    ["background light", { kind: "background", background: "light" }],
    ["spin", { kind: "spin", on: null }],
    ["stop", { kind: "spin", on: false }],
    ["screenshot", { kind: "screenshot" }],
    ["download", { kind: "download" }],
    ["quality", { kind: "panel", tab: "quality" }],
    ["next", { kind: "structure", step: 1 }],
  ])("%s", (text, expected) => {
    expect(parseViewerCommand(text)).toEqual(expected);
  });

  it("reads residues in the forms chemists write", () => {
    expect(parseViewerCommand("focus ASN140")).toEqual({
      kind: "residues",
      residues: [{ chain: null, resnum: 140 }],
      mode: "focus",
    });
    expect(parseViewerCommand("highlight Asn 140 and A:92")).toEqual({
      kind: "residues",
      residues: [
        { chain: null, resnum: 140 },
        { chain: "A", resnum: 92 },
      ],
      mode: "highlight",
    });
    expect(parseViewerCommand("ASN140 LEU92.B")).toMatchObject({
      residues: [
        { chain: null, resnum: 140 },
        { chain: "B", resnum: 92 },
      ],
    });
  });

  it("leaves questions for the agent", () => {
    expect(parseViewerCommand("why does the ligand bind so weakly?")).toBeNull();
    expect(parseViewerCommand("")).toBeNull();
  });

  it("parses residue tokens", () => {
    expect(parseResidues("Tyr97, Pro82 and 140.A")).toEqual([
      { chain: null, resnum: 97 },
      { chain: null, resnum: 82 },
      { chain: "A", resnum: 140 },
    ]);
  });
});

describe("parsePickLabel", () => {
  it("reads residue and chain from a Mol* label", () => {
    expect(parsePickLabel("<b>ASN 140</b> | A | ND2")).toEqual({ label: "ASN140", resname: "ASN", resnum: 140, chain: "A" });
  });
  it("keeps a label without a residue", () => {
    expect(parsePickLabel("Model 1")).toMatchObject({ resnum: null });
  });
});
