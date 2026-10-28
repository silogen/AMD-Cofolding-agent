// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { assetUrl, parseSharedAssets, type Asset } from "./assets";
import { fileKind, moleculesFromFile } from "./dropFiles";

/**
 * Target inputs and seed molecules read from files on the shared volume
 * (served at /api/artifacts/<path>): the protein chains of a structure, the
 * query sequence of an MSA, the SMILES of a molecule list.
 *
 * Structures are parsed in the browser because no tool returns chain
 * sequences, and a PDB or mmCIF file is small next to what the viewer
 * already loads.
 */

/** Input structures and .a3m MSAs from list_shared_assets; Boltz outputs are left out. */
export function targetFiles(payload: unknown): { structures: Asset[]; msas: Asset[] } {
  const assets = parseSharedAssets(payload).filter((a) => a.url);
  return { structures: assets.filter((a) => a.kind === "structure"), msas: assets.filter((a) => a.ext === "a3m") };
}

export async function fetchText(path: string): Promise<string> {
  const url = assetUrl(path);
  if (!url) throw new Error(`${path} is not on the shared volume.`);
  const response = await fetch(url);
  if (!response.ok) throw new Error(response.status === 404 ? `${path} was not found.` : `Could not read ${path} (HTTP ${response.status}).`);
  return response.text();
}

// ---------------------------------------------------------------------------
// protein chains
// ---------------------------------------------------------------------------

const AMINO: Record<string, string> = {
  ALA: "A", ARG: "R", ASN: "N", ASP: "D", CYS: "C", GLN: "Q", GLU: "E", GLY: "G", HIS: "H", ILE: "I",
  LEU: "L", LYS: "K", MET: "M", PHE: "F", PRO: "P", SER: "S", THR: "T", TRP: "W", TYR: "Y", VAL: "V",
};

/** Modified residues common in crystal structures, as the residue they modify; the TTP takes the 20 standard ones only. */
const MODIFIED: Record<string, string> = { MSE: "M", SEP: "S", TPO: "T", PTR: "Y", CSO: "C", CME: "C", CAS: "C", CSD: "C", OCS: "C", HYP: "P", MLY: "K", KCX: "K", LLP: "K", M3L: "K" };

/** The one-letter code of a standard or common modified amino acid, else null. */
export const letter = (code: string) => AMINO[code.toUpperCase()] ?? MODIFIED[code.toUpperCase()] ?? null;

/** One or more chains with the same protein sequence. */
export interface ProteinChain {
  /** Chain ids, e.g. ["A", "B"] for a homodimer. */
  ids: string[];
  sequence: string;
  /** The deposited construct (SEQRES, entity_poly) or only the residues with coordinates. */
  source: "construct" | "atoms";
}

type RawChain = { id: string; sequence: string; source: ProteinChain["source"] };

function append(map: Map<string, string[]>, key: string, ...items: string[]) {
  const list = map.get(key);
  if (list) list.push(...items);
  else map.set(key, items);
}

const MIN_RESIDUES = 10;

/** Residue names as a protein sequence, or null when they are mostly not amino acids (DNA, RNA, ligands). */
function proteinFrom(residues: string[]): string | null {
  const letters = residues.map(letter).filter((l): l is string => l !== null);
  return letters.length >= MIN_RESIDUES && letters.length * 2 >= residues.length ? letters.join("") : null;
}

/** One-letter code text (mmCIF pdbx_seq_one_letter_code_can) as the 20 standard residues. */
const fromOneLetter = (text: string) => text.toUpperCase().replace(/[^ACDEFGHIKLMNPQRSTVWY]/g, "");

function group(chains: RawChain[]): ProteinChain[] {
  const out: ProteinChain[] = [];
  for (const chain of chains) {
    const same = out.find((c) => c.sequence === chain.sequence && c.source === chain.source);
    if (same) same.ids.push(chain.id);
    else out.push({ ids: [chain.id], sequence: chain.sequence, source: chain.source });
  }
  return out;
}

/** Protein chains of a PDB file: SEQRES when present, else the ATOM records of the first model. */
export function pdbChains(text: string): ProteinChain[] {
  const seqres = new Map<string, string[]>();
  const atoms = new Map<string, string[]>();
  const last = new Map<string, string>();
  for (const line of text.split(/\r?\n/)) {
    const record = line.slice(0, 6);
    if (record === "SEQRES") {
      const id = line[11]?.trim() || "_";
      append(seqres, id, ...line.slice(19).trim().split(/\s+/).filter(Boolean));
    } else if (record === "ATOM  " || record === "HETATM") {
      const resname = line.slice(17, 20).trim();
      if (record === "HETATM" && !MODIFIED[resname]) continue;
      const id = line[21]?.trim() || "_";
      const key = line.slice(22, 27);
      if (last.get(id) === key) continue;
      last.set(id, key);
      append(atoms, id, resname);
    } else if (record === "ENDMDL") break;
  }
  const chains: RawChain[] = [];
  for (const id of new Set([...seqres.keys(), ...atoms.keys()])) {
    const construct = proteinFrom(seqres.get(id) ?? []);
    const observed = construct ? null : proteinFrom(atoms.get(id) ?? []);
    if (construct) chains.push({ id, sequence: construct, source: "construct" });
    else if (observed) chains.push({ id, sequence: observed, source: "atoms" });
  }
  return group(chains);
}

type Token = { value: string; quoted: boolean };

const CIF_TOKEN = /'(?:[^']|'(?=\S))*'(?=\s|$)|"(?:[^"]|"(?=\S))*"(?=\s|$)|\S+/g;

function* cifTokens(text: string): Generator<Token> {
  const lines = text.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.startsWith(";")) {
      const parts = [line.slice(1)];
      while (++i < lines.length && !lines[i].startsWith(";")) parts.push(lines[i]);
      yield { value: parts.join("\n"), quoted: true };
      continue;
    }
    for (const match of line.matchAll(CIF_TOKEN)) {
      const token = match[0];
      if (token.startsWith("#")) break;
      if (token[0] === "'" || token[0] === '"') yield { value: token.slice(1, -1), quoted: true };
      else yield { value: token, quoted: false };
    }
  }
}

type Row = Record<string, string | null>;

/** The rows of some mmCIF categories ("entity_poly", ...), keyed by field name in lower case. */
export function readCif(text: string, categories: readonly string[]): Map<string, Row[]> {
  const wanted = new Set(categories.map((c) => c.toLowerCase()));
  const out = new Map<string, Row[]>();
  const split = (tag: string) => {
    const dot = tag.indexOf(".");
    return dot < 0 ? [tag.slice(1).toLowerCase(), ""] : [tag.slice(1, dot).toLowerCase(), tag.slice(dot + 1).toLowerCase()];
  };
  let mode: "none" | "tags" | "values" = "none";
  let tags: string[] = [];
  let values: (string | null)[] = [];
  let pending: string | null = null;
  const flush = () => {
    if (tags.length) {
      const [category] = split(tags[0]);
      if (wanted.has(category)) {
        const fields = tags.map((t) => split(t)[1]);
        const rows = out.get(category) ?? [];
        for (let i = 0; i + fields.length <= values.length; i += fields.length) rows.push(Object.fromEntries(fields.map((f, j) => [f, values[i + j]])));
        out.set(category, rows);
      }
    }
    mode = "none";
    tags = [];
    values = [];
  };
  let keep = false;
  for (const token of cifTokens(text)) {
    const value = !token.quoted && (token.value === "." || token.value === "?") ? null : token.value;
    if (pending) {
      const [category, field] = split(pending);
      if (wanted.has(category)) {
        const rows = out.get(category) ?? [{}];
        rows[0][field] = value;
        out.set(category, rows);
      }
      pending = null;
      continue;
    }
    const bare = !token.quoted;
    if (bare && token.value.toLowerCase() === "loop_") {
      flush();
      mode = "tags";
      continue;
    }
    if (bare && token.value.startsWith("_")) {
      if (mode === "tags") {
        tags.push(token.value);
        keep = wanted.has(split(tags[0])[0]);
        continue;
      }
      flush();
      pending = token.value;
      continue;
    }
    if (bare && /^data_/i.test(token.value)) {
      flush();
      continue;
    }
    if (mode === "tags" || mode === "values") {
      mode = "values";
      if (keep) values.push(value);
      else values.length = 0;
    }
  }
  flush();
  return out;
}

/** Protein chains of an mmCIF file: entity_poly, else entity_poly_seq, else atom_site of the first model. */
export function cifChains(text: string): ProteinChain[] {
  const tables = readCif(text, ["entity_poly", "entity_poly_seq", "struct_asym"]);
  const asyms = tables.get("struct_asym") ?? [];
  const chainsOf = (entity: string, strands: string | null | undefined) =>
    strands ? strands.split(",").map((s) => s.trim()).filter(Boolean) : asyms.filter((a) => a.entity_id === entity && a.id).map((a) => a.id as string);
  const polySeq = new Map<string, string[]>();
  const seenNum = new Set<string>();
  for (const row of tables.get("entity_poly_seq") ?? []) {
    const entity = row.entity_id ?? "";
    const key = `${entity}:${row.num ?? ""}`;
    if (!row.mon_id || seenNum.has(key)) continue;
    seenNum.add(key);
    append(polySeq, entity, row.mon_id);
  }
  const chains: RawChain[] = [];
  const polys = tables.get("entity_poly") ?? [];
  if (polys.length) {
    for (const poly of polys) {
      const entity = poly.entity_id ?? "";
      if (poly.type && !/polypeptide/i.test(poly.type)) continue;
      const can = poly.pdbx_seq_one_letter_code_can ? fromOneLetter(poly.pdbx_seq_one_letter_code_can) : "";
      const sequence = can.length >= MIN_RESIDUES ? can : proteinFrom(polySeq.get(entity) ?? []);
      if (sequence) for (const id of chainsOf(entity, poly.pdbx_strand_id)) chains.push({ id, sequence, source: "construct" });
    }
  } else {
    for (const [entity, residues] of polySeq) {
      const sequence = proteinFrom(residues);
      if (sequence) for (const id of chainsOf(entity, null)) chains.push({ id, sequence, source: "construct" });
    }
  }
  if (chains.length) return group(chains);

  const atoms = readCif(text, ["atom_site"]).get("atom_site") ?? [];
  const model = atoms[0]?.pdbx_pdb_model_num ?? null;
  const residues = new Map<string, string[]>();
  const last = new Map<string, string>();
  for (const atom of atoms) {
    if (model !== null && atom.pdbx_pdb_model_num !== model) break;
    const resname = atom.label_comp_id ?? atom.auth_comp_id ?? "";
    if (atom.group_pdb === "HETATM" && !MODIFIED[resname]) continue;
    const id = atom.auth_asym_id ?? atom.label_asym_id ?? "_";
    const key = `${atom.auth_seq_id ?? atom.label_seq_id ?? ""}${atom.pdbx_pdb_ins_code ?? ""}`;
    if (last.get(id) === key) continue;
    last.set(id, key);
    append(residues, id, resname);
  }
  for (const [id, list] of residues) {
    const sequence = proteinFrom(list);
    if (sequence) chains.push({ id, sequence, source: "atoms" });
  }
  return group(chains);
}

export const isCif = (path: string, text: string) => /\.(cif|mmcif)$/i.test(path) || (!/\.(pdb|ent)$/i.test(path) && /^\s*data_/.test(text));

export const structureChains = (path: string, text: string) => (isCif(path, text) ? cifChains(text) : pdbChains(text));

export async function readStructureChains(path: string): Promise<ProteinChain[]> {
  return structureChains(path, await fetchText(path));
}

/** "5UF0" for 5uf0.cif or 5UF0_protein.pdb; null when the name is not a PDB id. */
export function pdbIdFromPath(path: string): string | null {
  const name = path.replace(/\\/g, "/").split("/").pop() ?? "";
  return /^([0-9][a-z0-9]{3})(?=[._-])/i.exec(name)?.[1].toUpperCase() ?? null;
}

// ---------------------------------------------------------------------------
// MSAs
// ---------------------------------------------------------------------------

/** The query (first record) of an a3m, read as the Boltz service reads it: gaps dropped, upper case. */
export function a3mQuery(text: string): string | null {
  const residues: string[] = [];
  let seen = false;
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) continue;
    if (line.startsWith(">")) {
      if (seen) break;
      seen = true;
      continue;
    }
    if (seen) residues.push(line);
  }
  const sequence = residues.join("").replace(/-/g, "").toUpperCase();
  return sequence || null;
}

const MAX_QUERY_BYTES = 1 << 20;

/** The query sequence of a shared .a3m, reading only as far as its second record. */
export async function readA3mQuery(path: string): Promise<string | null> {
  const url = assetUrl(path);
  if (!url) throw new Error(`${path} is not on the shared volume.`);
  const response = await fetch(url);
  if (!response.ok) throw new Error(response.status === 404 ? `${path} was not found.` : `Could not read ${path} (HTTP ${response.status}).`);
  const reader = response.body?.getReader();
  if (!reader) return a3mQuery(await response.text());
  const decoder = new TextDecoder();
  let text = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    text += decoder.decode(value, { stream: true });
    const first = text.indexOf(">");
    if ((first >= 0 && text.indexOf("\n>", first + 1) >= 0) || text.length > MAX_QUERY_BYTES) {
      void reader.cancel().catch(() => undefined);
      break;
    }
  }
  return a3mQuery(text);
}

const aa = (n: number) => `${n.toLocaleString("en")} aa`;

/** Why an MSA does not go with the sequence, or null when it does (or either is missing). */
export function msaMismatch(query: string | null, sequence: string): string | null {
  if (!query || !sequence || query === sequence) return null;
  if (query.length !== sequence.length) return `The MSA is for a ${aa(query.length)} sequence, not this ${aa(sequence.length)} one: Boltz would fold the MSA's sequence.`;
  return "The MSA's sequence differs from this one: Boltz would fold the MSA's sequence.";
}

// ---------------------------------------------------------------------------
// seed molecules
// ---------------------------------------------------------------------------

/** The SMILES in a shared .smi, .csv or .sdf file, and how many records could not be read. */
export async function readSeedFile(path: string): Promise<{ smiles: string[]; failed: number }> {
  const name = path.replace(/\\/g, "/").split("/").pop() ?? path;
  const kind = fileKind(name);
  if (kind !== "smi" && kind !== "csv" && kind !== "sdf") throw new Error("Pick a .smi, .csv or .sdf file.");
  return moleculesFromFile(new File([await fetchText(path)], name));
}
