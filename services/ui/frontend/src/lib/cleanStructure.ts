// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { isCif, letter, readCif } from "./sharedFiles";

/**
 * One protein chain of a PDB or mmCIF file as a minimal mmCIF, for Boltz-2
 * to use as a template: the first model, the chain's amino acids (standard
 * and common modified ones), the first alt-loc of each residue, no
 * hydrogens, waters, ligands or ions.
 *
 * Boltz reads templates with gemmi and names chains by label_asym_id (a PDB
 * file's chains become "A1", ...), so the output gives the chain the same
 * author and label id: the chain picked here is the one Boltz finds.
 */

interface Atom {
  record: "ATOM" | "HETATM";
  name: string;
  element: string;
  comp: string;
  /** label_seq_id: position in the chain's sequence, from 1. */
  num: number;
  authSeq: string;
  ins: string | null;
  x: string;
  y: string;
  z: string;
  occupancy: string;
  b: string;
}

interface Chain {
  /** The chain's sequence as residue names, num -> name. */
  sequence: Map<number, string>;
  atoms: Atom[];
}

export interface CleanedChain {
  text: string;
  atoms: number;
  residues: number;
}

const HYDROGEN = new Set(["H", "D"]);

/** Keeps the first residue name and alt-loc seen at each position, and each atom name once. */
function picker() {
  const comps = new Map<string, string>();
  const alts = new Map<string, string>();
  const names = new Set<string>();
  return (key: string, comp: string, alt: string | null, name: string) => {
    if ((comps.get(key) ?? comp) !== comp) return false;
    comps.set(key, comp);
    if (alt) {
      if ((alts.get(key) ?? alt) !== alt) return false;
      alts.set(key, alt);
    }
    if (names.has(`${key}:${name}`)) return false;
    names.add(`${key}:${name}`);
    return true;
  };
}

function cifChain(text: string, chainId: string): Chain {
  const tables = readCif(text, ["atom_site", "entity_poly", "entity_poly_seq"]);
  const rows = tables.get("atom_site") ?? [];
  const polys = tables.get("entity_poly") ?? [];
  const peptides = new Set(polys.filter((p) => !p.type || /polypeptide/i.test(p.type)).map((p) => p.entity_id));
  const model = rows[0]?.pdbx_pdb_model_num ?? null;
  const inChain = rows.filter(
    (r) =>
      (model === null || r.pdbx_pdb_model_num === model) &&
      (r.auth_asym_id ?? r.label_asym_id) === chainId &&
      r.label_seq_id != null &&
      (!polys.length || peptides.has(r.label_entity_id ?? null)),
  );
  const entity = inChain.find((r) => letter(r.label_comp_id ?? r.auth_comp_id ?? ""))?.label_entity_id ?? null;
  const keep = picker();
  const atoms: Atom[] = [];
  for (const r of inChain) {
    const comp = r.label_comp_id ?? r.auth_comp_id ?? "";
    const name = r.label_atom_id ?? r.auth_atom_id ?? "";
    const element = (r.type_symbol ?? name.replace(/[^A-Za-z]/g, "").slice(0, 1)).toUpperCase();
    const num = Number(r.label_seq_id);
    if (r.label_entity_id !== entity || !letter(comp) || HYDROGEN.has(element) || !name || !Number.isInteger(num)) continue;
    if (!keep(String(num), comp, r.label_alt_id ?? null, name)) continue;
    atoms.push({
      record: r.group_pdb === "HETATM" ? "HETATM" : "ATOM",
      name,
      element,
      comp,
      num,
      authSeq: r.auth_seq_id ?? String(num),
      ins: r.pdbx_pdb_ins_code ?? null,
      x: r.cartn_x ?? "0",
      y: r.cartn_y ?? "0",
      z: r.cartn_z ?? "0",
      occupancy: r.occupancy ?? "1",
      b: r.b_iso_or_equiv ?? "0",
    });
  }
  const sequence = new Map<number, string>();
  for (const row of tables.get("entity_poly_seq") ?? []) {
    const num = Number(row.num);
    if (row.entity_id === entity && row.mon_id && Number.isInteger(num) && !sequence.has(num)) sequence.set(num, row.mon_id);
  }
  if (sequence.size && atoms.every((a) => sequence.has(a.num))) return { sequence, atoms };
  // Without the full sequence, number the observed residues 1..n: Boltz indexes the sequence by label_seq_id.
  const order = new Map([...new Set(atoms.map((a) => a.num))].map((num, i) => [num, i + 1]));
  const renumbered = atoms.map((a) => ({ ...a, num: order.get(a.num) ?? 0 }));
  return { sequence: new Map(renumbered.map((a) => [a.num, a.comp])), atoms: renumbered };
}

/** A PDB file's chain, numbered by the residues with coordinates (SEQRES is not aligned to them). */
function pdbChain(text: string, chainId: string): Chain {
  const keep = picker();
  const atoms: Atom[] = [];
  const sequence = new Map<number, string>();
  let lastKey: string | null = null;
  let num = 0;
  for (const line of text.split(/\r?\n/)) {
    const record = line.slice(0, 6);
    if (record === "ENDMDL" && atoms.length) break;
    if (record !== "ATOM  " && record !== "HETATM") continue;
    if ((line[21]?.trim() || "_") !== chainId) continue;
    const comp = line.slice(17, 20).trim();
    const name = line.slice(12, 16).trim();
    const element = (line.slice(76, 78).trim() || name.replace(/[^A-Za-z]/g, "").slice(0, 1)).toUpperCase();
    const key = line.slice(22, 27);
    if (!letter(comp) || HYDROGEN.has(element) || !name || !keep(key, comp, line[16]?.trim() || null, name)) continue;
    if (key !== lastKey) {
      num += 1;
      lastKey = key;
      sequence.set(num, comp);
    }
    atoms.push({
      record: record === "HETATM" ? "HETATM" : "ATOM",
      name,
      element,
      comp,
      num,
      authSeq: line.slice(22, 26).trim(),
      ins: line[26]?.trim() || null,
      x: line.slice(30, 38).trim(),
      y: line.slice(38, 46).trim(),
      z: line.slice(46, 54).trim(),
      occupancy: line.slice(54, 60).trim() || "1.00",
      b: line.slice(60, 66).trim() || "0.00",
    });
  }
  return { sequence, atoms };
}

const RESERVED = /^(data_|loop_|save_|global_|stop_)/i;

/** A value as an mmCIF token, quoted when it would not read back as itself. */
export function cifValue(value: string | null): string {
  if (value === null || value === "") return ".";
  if (/^[^\s'"_#$;[\]][^\s'"]*$/.test(value) && !RESERVED.test(value) && value !== "." && value !== "?") return value;
  return value.includes("'") ? `"${value}"` : `'${value}'`;
}

const loop = (category: string, fields: string[], rows: (string | null)[][]) =>
  ["loop_", ...fields.map((f) => `_${category}.${f}`), ...rows.map((r) => r.map(cifValue).join(" ")), "#"].join("\n");

const ATOM_FIELDS = [
  "group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id", "label_comp_id", "label_asym_id", "label_entity_id", "label_seq_id",
  "pdbx_PDB_ins_code", "Cartn_x", "Cartn_y", "Cartn_z", "occupancy", "B_iso_or_equiv", "auth_seq_id", "auth_comp_id", "auth_asym_id",
  "auth_atom_id", "pdbx_PDB_model_num",
];

function writeCif(name: string, chainId: string, chain: Chain): string {
  const nums = [...new Set([...chain.sequence.keys(), ...chain.atoms.map((a) => a.num)])].sort((a, b) => a - b);
  const monomers = nums.map((n) => [n, chain.sequence.get(n) ?? chain.atoms.find((a) => a.num === n)?.comp ?? "UNK"] as const);
  const oneLetter = monomers.map(([, comp]) => letter(comp) ?? "X").join("");
  const block = name.replace(/[^A-Za-z0-9_-]/g, "_");
  return [
    `data_${block}`,
    "#",
    `_entry.id ${cifValue(block)}`,
    "#",
    loop("entity", ["id", "type"], [["1", "polymer"]]),
    "_entity_poly.entity_id 1",
    "_entity_poly.type 'polypeptide(L)'",
    "_entity_poly.nstd_linkage no",
    `_entity_poly.pdbx_strand_id ${cifValue(chainId)}`,
    "_entity_poly.pdbx_seq_one_letter_code_can",
    `;${oneLetter.replace(/(.{80})/g, "$1\n")}`,
    ";",
    "#",
    loop("entity_poly_seq", ["entity_id", "num", "mon_id", "hetero"], monomers.map(([n, comp]) => ["1", String(n), comp, "n"])),
    loop("struct_asym", ["id", "entity_id"], [[chainId, "1"]]),
    loop(
      "atom_site",
      ATOM_FIELDS,
      chain.atoms.map((a, i) => [
        a.record, String(i + 1), a.element, a.name, null, a.comp, chainId, "1", String(a.num),
        a.ins, a.x, a.y, a.z, a.occupancy, a.b, a.authSeq, a.comp, chainId, a.name, "1",
      ]),
    ),
    "",
  ].join("\n");
}

/** Chain *chainId* (author id) of a PDB or mmCIF file as a template-ready mmCIF named *name*. */
export function cleanChain(path: string, text: string, chainId: string, name: string): CleanedChain {
  const chain = isCif(path, text) ? cifChain(text, chainId) : pdbChain(text, chainId);
  if (!chain.atoms.length) throw new Error(`Chain ${chainId} has no protein atoms.`);
  return { text: writeCif(name, chainId, chain), atoms: chain.atoms.length, residues: new Set(chain.atoms.map((a) => a.num)).size };
}

/** "1M17_A.cif" for chain A of 1M17, "1m17_protein_A.cif" for chain A of 1m17_protein.pdb. */
export function templateName(source: string, chainId: string): string {
  const stem = (source.replace(/\\/g, "/").split("/").pop() ?? source).replace(/\.[^.]+$/, "");
  return `${stem.replace(/[^A-Za-z0-9_-]/g, "_")}_${chainId.replace(/[^A-Za-z0-9]/g, "_")}.cif`;
}
