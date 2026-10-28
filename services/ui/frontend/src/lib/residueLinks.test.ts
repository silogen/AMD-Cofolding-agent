// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { Link, Root } from "mdast";
import { describe, expect, it } from "vitest";
import type { Panel } from "../types/api";
import {
  matchStructure,
  remarkResidueLinks,
  residueFromHref,
  residueHref,
  splitResidues,
  structureFromHref,
  structureMentions,
  turnStructures,
} from "./residueLinks";

const residuesIn = (text: string) => splitResidues(text).flatMap((segment) => ("residue" in segment ? [segment.residue] : []));

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

  it("reads chains in parentheses and insertion codes", () => {
    expect(residuesIn("LEU83 (chain B) and Asn 140A")).toEqual([
      { chain: "B", resnum: 83, resname: "LEU" },
      { chain: null, resnum: 140, resname: "ASN", insCode: "A" },
    ]);
  });

  it("finds modified residues written apart from their number", () => {
    expect(residuesIn("the covalent CY0 797 adduct")).toEqual([{ chain: null, resnum: 797, resname: "CY0" }]);
  });

  it("does not read an atom and a distance as a residue", () => {
    expect(residuesIn("C48 4.4 Å from the ring, N1A 3.1 Å")).toEqual([]);
    expect(residuesIn("H2O 3,5 molecules")).toEqual([]);
  });
});

describe("residue hrefs", () => {
  it("round-trips", () => {
    expect(residueFromHref(residueHref({ chain: "B", resnum: 92 }))).toEqual({ chain: "B", resnum: 92 });
    expect(residueFromHref(residueHref({ chain: null, resnum: 140 }))).toEqual({ chain: null, resnum: 140 });
    expect(residueFromHref(residueHref({ chain: "A", resnum: 140, resname: "ASN" }))).toEqual({ chain: "A", resnum: 140, resname: "ASN" });
    expect(residueFromHref(residueHref({ chain: null, resnum: 52, resname: "GLY", insCode: "B" }))).toEqual({
      chain: null,
      resnum: 52,
      resname: "GLY",
      insCode: "B",
    });
    expect(residueFromHref("https://example.org")).toBeNull();
  });

  it("carries the structure, as a path or a name to look up", () => {
    const path = "uploads/kinases/3MXF_A.cif";
    expect(structureFromHref(residueHref({ chain: null, resnum: 81 }, path))).toEqual({ path, ref: null, near: null });
    expect(structureFromHref(residueHref({ chain: null, resnum: 81 }, "3MXF"))).toEqual({ path: null, ref: "3MXF", near: null });
    expect(structureFromHref(residueHref({ chain: null, resnum: 81 }))).toEqual({ path: null, ref: null, near: null });
  });
});

const PATHS = [
  "uploads/kinases/4ZC9_A.cif",
  "uploads/kinases/A.cif",
  "predictions/out_ab12/boltz_results_batch_ab12/predictions/mol_3/mol_3_model_0.pdb",
  "rcsb/3mxf.cif",
];

describe("structure mentions", () => {
  it("finds PDB ids, Boltz molecules and file names, in order", () => {
    expect(structureMentions("In 3mxf and 4ZC9_A, unlike mol_3 (see mol_0_model_0.pdb), 1000 times")).toEqual([
      "3MXF",
      "4ZC9",
      "mol_3",
      "mol_0_model_0.pdb",
    ]);
  });

  it("matches a mention to a structure", () => {
    expect(matchStructure("4ZC9", PATHS)).toBe(PATHS[0]);
    expect(matchStructure("3MXF", PATHS)).toBe(PATHS[3]);
    expect(matchStructure("mol_3", PATHS)).toBe(PATHS[2]);
    expect(matchStructure("A.cif", PATHS)).toBe(PATHS[1]);
    expect(matchStructure("/shared/uploads/kinases/4ZC9_A.cif", PATHS)).toBe(PATHS[0]);
    expect(matchStructure("2J5E", PATHS)).toBeNull();
  });

  it("finds a Boltz molecule in the run it was said near", () => {
    const run = (id: string, mol: string) => `predictions/out_${id}/boltz_results_batch_${id}/predictions/${mol}/${mol}_model_0.pdb`;
    const paths = [run("aaaa", "mol_3"), run("bbbb", "mol_9"), run("bbbb", "mol_3")];
    expect(matchStructure("mol_3", paths, run("bbbb", "mol_9"))).toBe(paths[2]);
    expect(matchStructure("mol_4", paths, run("bbbb", "mol_9"))).toBeNull();
    expect(matchStructure("mol_3", paths)).toBe(paths[0]);
  });

  it("collects the structures a turn worked on", () => {
    const steps = [
      { name: "list_shared_assets", arguments: { path: "/shared/uploads/x.cif" } },
      { name: "profile_interactions", arguments: { structure_paths: ["/shared/uploads/kinases/4ZC9_A.cif", "rcsb/3mxf.cif"] } },
    ];
    const panels = [{ kind: "structure", path: "rcsb/3mxf.cif" }, { kind: "interactions", structurePath: PATHS[2] }] as Panel[];
    expect(turnStructures(steps, panels)).toEqual([PATHS[0], PATHS[3], PATHS[2]]);
  });
});

describe("remarkResidueLinks", () => {
  const paragraph = (...values: string[]): Root => ({
    type: "root",
    children: values.map((value) => ({ type: "paragraph", children: [{ type: "text", value }] })),
  });
  const links = (tree: Root) =>
    tree.children.flatMap((node) => ("children" in node ? node.children.filter((child): child is Link => child.type === "link") : []));

  it("sends each residue to the structure the answer last named", () => {
    const tree = paragraph("In 4ZC9 the hinge MET793 holds it.", "3MXF instead stacks on TRP81.");
    remarkResidueLinks({ structures: PATHS })(tree);
    expect(links(tree).map((link) => structureFromHref(link.url).path)).toEqual([PATHS[0], PATHS[3]]);
  });

  it("keeps a name it cannot match, and uses the only structure when none is named", () => {
    const named = paragraph("2J5E binds ASP86.");
    remarkResidueLinks({ structures: PATHS })(named);
    expect(structureFromHref(links(named)[0].url)).toEqual({ path: null, ref: "2J5E", near: null });

    const single = paragraph("The pocket around LYS33.");
    remarkResidueLinks({ structures: [PATHS[0]] })(single);
    expect(structureFromHref(links(single)[0].url).path).toBe(PATHS[0]);

    const none = paragraph("The pocket around LYS33.");
    remarkResidueLinks({ structures: PATHS })(none);
    expect(structureFromHref(links(none)[0].url)).toEqual({ path: null, ref: null, near: null });
  });

  it("sends a residue of another molecule of the run there", () => {
    const tree = paragraph("Unlike mol_9, mol_7 holds LEU149.");
    remarkResidueLinks({ structures: [PATHS[2]] })(tree);
    expect(structureFromHref(links(tree)[0].url)).toEqual({ path: null, ref: "mol_7", near: PATHS[2] });
  });
});
