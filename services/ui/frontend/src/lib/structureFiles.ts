// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { assetUrl } from "./assets";
import { oneLetter, type FileResidue } from "./numbering";

/**
 * The files that go with a structure: the file itself, and for a Boltz
 * prediction its ligand, confidence and affinity files, its input with the
 * MSA and template it names; a FASTA of the protein and the ligand's SMILES
 * are written in the browser when there is no file for them.
 */

export interface RelatedFile {
  label: string;
  /** "PDB", "SDF", "A3M"… */
  type: string;
  url: string;
  /** Download name. */
  name: string;
}

/** A Boltz output: predictions/out_<id>/boltz_results_batch_<id>/predictions/<mol>/<mol>_model_<k>.pdb. */
const BOLTZ = /^(.*\/)?predictions\/out_([0-9a-f]+)\/boltz_results_batch_\2\/predictions\/(mol_\d+)\/\3_model_(\d+)\.(?:pdb|cif)$/;

export interface BoltzOutput {
  /** Files next to the model, worth a look when they exist. */
  candidates: RelatedFile[];
  /** The YAML Boltz was given, which names the MSA and templates. */
  input: RelatedFile;
}

const fileOf = (label: string, path: string): RelatedFile | null => {
  const url = assetUrl(path);
  const name = path.split("/").pop() ?? path;
  const type = (name.split(".").pop() ?? "").toUpperCase();
  return url ? { label, type, url, name } : null;
};

export function boltzOutput(path: string): BoltzOutput | null {
  const clean = path.replace(/\\/g, "/").replace(/^\/?shared\//, "");
  const match = BOLTZ.exec(clean);
  if (!match) return null;
  const [, root = "", id, mol, model] = match;
  const dir = clean.slice(0, clean.lastIndexOf("/"));
  const candidates = [
    fileOf("Ligand", `${dir}/${mol}_model_${model}_lig.sdf`),
    fileOf("Confidence", `${dir}/confidence_${mol}_model_${model}.json`),
    fileOf("Affinity", `${dir}/affinity_${mol}.json`),
  ].filter((file): file is RelatedFile => file !== null);
  const input = fileOf("Boltz input", `${root}batches/batch_${id}/${mol}.yaml`);
  return input ? { candidates, input } : null;
}

export interface BoltzInput {
  proteins: { id: string; sequence: string }[];
  msas: string[];
  templates: string[];
}

/** The parts of a Boltz input YAML the files list needs; a line-by-line read, not a YAML parser. */
export function parseBoltzInput(text: string): BoltzInput {
  const out: BoltzInput = { proteins: [], msas: [], templates: [] };
  let entity: string | null = null;
  let id: string | null = null;
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim().replace(/^-\s+/, "");
    const key = /^(\w+):\s*(.*)$/.exec(line);
    if (!key) continue;
    const [, name, value] = key;
    const unquoted = value.replace(/^["']|["']$/g, "");
    if (!unquoted && ["protein", "ligand", "dna", "rna"].includes(name)) {
      entity = name;
      id = null;
    } else if (name === "id") id = unquoted.replace(/^\[|\]$/g, "").split(",")[0].trim();
    else if (name === "sequence" && entity === "protein" && unquoted) out.proteins.push({ id: id ?? String.fromCharCode(65 + out.proteins.length), sequence: unquoted });
    else if (name === "msa" && unquoted && unquoted !== "empty") out.msas.push(unquoted);
    else if ((name === "cif" || name === "pdb") && unquoted) out.templates.push(unquoted);
  }
  return out;
}

const dataUrl = (text: string) => `data:text/plain;charset=utf-8,${encodeURIComponent(text)}`;

export function fastaFile(stem: string, proteins: { id: string; sequence: string }[]): RelatedFile | null {
  if (!proteins.length) return null;
  const text = proteins.map((p) => `>${stem}|${p.id}\n${p.sequence.replace(/(.{80})/g, "$1\n").trim()}\n`).join("");
  return { label: "Sequence", type: "FASTA", url: dataUrl(text), name: `${stem}.fasta` };
}

/** The protein chains of a loaded structure, from its residues, for a FASTA. */
export function chainSequences(residues: FileResidue[]): { id: string; sequence: string }[] {
  const chains = new Map<string, string[]>();
  for (const residue of residues) {
    const letter = oneLetter(residue.resname);
    if (!letter) continue;
    const id = residue.chain ?? "A";
    const letters = chains.get(id);
    if (letters) letters.push(letter);
    else chains.set(id, [letter]);
  }
  return [...chains].filter(([, letters]) => letters.length >= 10).map(([id, letters]) => ({ id, sequence: letters.join("") }));
}

export function smilesFile(stem: string, smiles: string): RelatedFile {
  return { label: "Ligand", type: "SMILES", url: dataUrl(`${smiles}\t${stem}\n`), name: `${stem}.smi` };
}

/** The MSAs and templates a Boltz input names. */
export function inputFiles(input: BoltzInput): RelatedFile[] {
  return [
    ...input.msas.map((path) => fileOf("MSA", path)),
    ...input.templates.map((path) => fileOf("Template", path)),
  ].filter((file, i, all): file is RelatedFile => file !== null && all.findIndex((other) => other?.url === file.url) === i);
}
