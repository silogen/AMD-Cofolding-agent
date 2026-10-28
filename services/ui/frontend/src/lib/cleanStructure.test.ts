// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { cifValue, cleanChain, templateName } from "./cleanStructure";
import { cifChains, readCif, structureChains } from "./sharedFiles";

const SEQ = ["MSE", "GLY", "SER", "LYS", "ALA", "LEU", "VAL", "ILE", "GLU", "ASP", "PHE", "TRP"];

const site = (fields: (string | number)[]) => fields.join(" ");
/** group id type atom alt comp label_asym entity label_seq x y z occ b auth_seq auth_asym model */
const atom = (id: number, type: string, name: string, alt: string, comp: string, asym: string, entity: string, seq: string, authSeq: string, auth: string, model = 1, group = "ATOM") =>
  site([group, id, type, name, alt, comp, asym, entity, seq, "?", "1.0", "2.0", "3.0", "1.00", "10.0", authSeq, comp, auth, name, model]);

const CIF = `data_1ABC
#
loop_
_entity.id
_entity.type
1 polymer
2 non-polymer
3 water
#
_entity_poly.entity_id 1
_entity_poly.type 'polypeptide(L)'
_entity_poly.pdbx_strand_id A,B
_entity_poly.pdbx_seq_one_letter_code_can MGSKALVIEDFW
#
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
_entity_poly_seq.hetero
${SEQ.map((m, i) => `1 ${i + 1} ${m} n`).join("\n")}
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
${[
  atom(1, "N", "N", ".", "MSE", "C", "1", "1", "671", "A", 1, "HETATM"),
  atom(2, "SE", "SE", ".", "MSE", "C", "1", "1", "671", "A", 1, "HETATM"),
  atom(3, "N", "N", "A", "GLY", "C", "1", "2", "672", "A"),
  atom(4, "N", "N", "B", "GLY", "C", "1", "2", "672", "A"),
  atom(5, "C", "CA", "B", "GLY", "C", "1", "2", "672", "A"),
  atom(6, "C", "CA", "A", "GLY", "C", "1", "2", "672", "A"),
  atom(7, "N", "N", ".", "SER", "C", "1", "3", "673", "A"),
  atom(8, "H", "H", ".", "SER", "C", "1", "3", "673", "A"),
  atom(9, "C", "C1", ".", "ATP", "D", "2", ".", "901", "A", 1, "HETATM"),
  atom(10, "O", "O", ".", "HOH", "E", "3", ".", "1001", "A", 1, "HETATM"),
  atom(11, "N", "N", ".", "LYS", "F", "1", "4", "674", "B"),
  atom(12, "N", "N", ".", "LYS", "C", "1", "4", "674", "A", 2),
].join("\n")}
#
`;

describe("cleanChain", () => {
  it("keeps one chain's amino acids of the first model, first alt-loc, no hydrogens, waters or ligands", () => {
    const out = cleanChain("x/1abc.cif", CIF, "A", "1ABC_A");
    expect(out.atoms).toBe(5);
    expect(out.residues).toBe(3);
    const atoms = readCif(out.text, ["atom_site"]).get("atom_site") ?? [];
    expect(atoms.map((a) => `${a.label_comp_id}:${a.label_atom_id}:${a.label_seq_id}`)).toEqual(["MSE:N:1", "MSE:SE:1", "GLY:N:2", "GLY:CA:2", "SER:N:3"]);
    expect(atoms.every((a) => a.label_asym_id === "A" && a.auth_asym_id === "A" && a.label_entity_id === "1" && a.label_alt_id === null)).toBe(true);
    expect(atoms.map((a) => a.auth_seq_id)).toEqual(["671", "671", "672", "672", "673"]);
    expect(atoms[0].group_pdb).toBe("HETATM");
    expect(atoms[0].cartn_x).toBe("1.0");
  });

  it("writes an mmCIF the chain parser reads back as the full construct", () => {
    const text = cleanChain("x/1abc.cif", CIF, "A", "1ABC_A").text;
    expect(text.startsWith("data_1ABC_A\n")).toBe(true);
    expect(cifChains(text)).toEqual([{ ids: ["A"], sequence: "MGSKALVIEDFW", source: "construct" }]);
    expect(readCif(text, ["struct_asym"]).get("struct_asym")).toEqual([{ id: "A", entity_id: "1" }]);
    expect(readCif(text, ["entity_poly_seq"]).get("entity_poly_seq")).toHaveLength(12);
  });

  it("refuses a chain without protein atoms", () => {
    expect(() => cleanChain("x/1abc.cif", CIF, "Z", "1ABC_Z")).toThrow("Chain Z has no protein atoms.");
  });

  const pdbAtom = (serial: number, name: string, resname: string, chain: string, resnum: number, { alt = " ", record = "ATOM  ", element = name[0] } = {}) =>
    `${record}${String(serial).padStart(5)} ${name.length < 4 ? ` ${name.padEnd(3)}` : name}${alt}${resname.padEnd(3)} ${chain}${String(resnum).padStart(4)}    ${"1.000".padStart(8)}${"2.000".padStart(8)}${"3.000".padStart(8)}  1.00 10.00          ${element.padStart(2)}`;

  it("numbers a PDB chain's observed residues 1..n", () => {
    const residues = SEQ.map((res, i) => pdbAtom(i + 1, "CA", res, "B", i < 6 ? i + 10 : i + 20, { record: res === "MSE" ? "HETATM" : "ATOM  ", element: "C" }));
    const text = [
      pdbAtom(100, "N", "GLY", "A", 1),
      ...residues,
      pdbAtom(30, "CB", "TRP", "B", 31, { alt: "A", element: "C" }),
      pdbAtom(31, "CB", "TRP", "B", 31, { alt: "B", element: "C" }),
      pdbAtom(32, "HA", "TRP", "B", 31, { element: "H" }),
      pdbAtom(40, "C1", "ATP", "B", 401, { record: "HETATM", element: "C" }),
      "ENDMDL",
      pdbAtom(50, "CA", "GLY", "B", 99, { element: "C" }),
    ].join("\n");
    const out = cleanChain("shared/5uf0_protein.pdb", text, "B", "5uf0_protein_B");
    expect(out).toMatchObject({ atoms: 13, residues: 12 });
    const atoms = readCif(out.text, ["atom_site"]).get("atom_site") ?? [];
    expect([...new Set(atoms.map((a) => a.label_seq_id))]).toEqual(SEQ.map((_, i) => String(i + 1)));
    expect(atoms.at(-1)).toMatchObject({ label_atom_id: "CB", auth_seq_id: "31", label_asym_id: "B" });
    expect(structureChains("t.cif", out.text)).toEqual([{ ids: ["B"], sequence: "MGSKALVIEDFW", source: "construct" }]);
  });
});

describe("helpers", () => {
  it("quotes mmCIF values only when needed", () => {
    expect(cifValue("CA")).toBe("CA");
    expect(cifValue(null)).toBe(".");
    expect(cifValue("O5'")).toBe(`"O5'"`);
    expect(cifValue("a b")).toBe("'a b'");
    expect(cifValue("_x")).toBe("'_x'");
    expect(cifValue("data_x")).toBe("'data_x'");
  });

  it("names templates after the source and chain", () => {
    expect(templateName("1M17", "A")).toBe("1M17_A.cif");
    expect(templateName("structures/egfr model.pdb", "B")).toBe("egfr_model_B.cif");
  });
});
