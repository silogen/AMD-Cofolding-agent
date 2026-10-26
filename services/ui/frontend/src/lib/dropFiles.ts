// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { RDKitModule } from "@rdkit/rdkit";
import { loadRDKit, withMol } from "./rdkit";
import { looksLikeSmiles } from "./smiles";

/**
 * Files dropped on the chat or picked with the paperclip.
 *
 * Molecule lists (.smi, .csv, .sdf, .mol) are read in the browser into
 * SMILES, which is what the agent's tools take. Structures (.pdb, .cif) are
 * uploaded instead: the tools read them from the shared volume.
 */
export type FileKind = "smi" | "csv" | "sdf" | "structure";

export const ACCEPTED_FILES = ".smi,.csv,.sdf,.mol,.pdb,.cif,.mmcif,.ent";

export function fileKind(name: string): FileKind | null {
  const suffix = /\.([a-z0-9]+)$/i.exec(name)?.[1]?.toLowerCase();
  if (suffix === "smi" || suffix === "smiles") return "smi";
  if (suffix === "csv") return "csv";
  if (suffix === "sdf" || suffix === "mol") return "sdf";
  if (suffix === "pdb" || suffix === "cif" || suffix === "mmcif" || suffix === "ent") return "structure";
  return null;
}

/** A .smi file: one molecule per line, the SMILES first, a name after it. */
export function parseSmi(text: string): string[] {
  const out: string[] = [];
  for (const line of text.split(/\r?\n/)) {
    const token = line.trim().split(/[\s,]+/)[0];
    if (token && !token.startsWith("#") && looksLikeSmiles(token)) out.push(token);
  }
  return out;
}

/** One CSV line, with quoted fields. */
function csvFields(line: string, separator: string): string[] {
  const fields: string[] = [];
  let field = "";
  let quoted = false;
  for (let index = 0; index < line.length; index++) {
    const char = line[index];
    if (quoted) {
      if (char === '"' && line[index + 1] === '"') {
        field += '"';
        index++;
      } else if (char === '"') quoted = false;
      else field += char;
    } else if (char === '"') quoted = true;
    else if (char === separator) {
      fields.push(field.trim());
      field = "";
    } else field += char;
  }
  fields.push(field.trim());
  return fields;
}

const SMILES_HEADERS = ["smiles", "canonical_smiles", "smi", "input_smiles", "mol"];

/** A CSV with a SMILES column: the one named so, else the first whose values look like SMILES. */
export function parseCsv(text: string): string[] {
  const lines = text.split(/\r?\n/).filter((line) => line.trim());
  if (!lines.length) return [];
  const separator = lines[0].includes("\t") ? "\t" : lines[0].includes(";") && !lines[0].includes(",") ? ";" : ",";
  const rows = lines.map((line) => csvFields(line, separator));
  const header = rows[0].map((cell) => cell.toLowerCase());
  let column = header.findIndex((cell) => SMILES_HEADERS.includes(cell));
  let body = rows.slice(1);
  if (column < 0) {
    const sample = rows[1] ?? rows[0];
    column = sample.findIndex((cell) => looksLikeSmiles(cell));
    if (column < 0) return [];
    if (looksLikeSmiles(rows[0][column] ?? "")) body = rows;
  }
  return body.map((row) => row[column] ?? "").filter((cell) => cell.length > 0);
}

/** The molblocks of an SD file (a .mol file is one). */
export function splitSdf(text: string): string[] {
  const blocks: string[] = [];
  let lines: string[] = [];
  const close = () => {
    const end = lines.findIndex((line) => line.startsWith("M  END"));
    if (end >= 0) blocks.push(lines.slice(0, end + 1).join("\n"));
    lines = [];
  };
  for (const line of text.split(/\r?\n/)) {
    if (line.startsWith("$$$$")) close();
    else lines.push(line);
  }
  close();
  return blocks;
}

export function sdfToSmiles(rdkit: RDKitModule, text: string): { smiles: string[]; failed: number } {
  const smiles: string[] = [];
  let failed = 0;
  for (const block of splitSdf(text)) {
    const found = withMol(rdkit, block, (mol) => mol.get_smiles());
    if (found) smiles.push(found);
    else failed++;
  }
  return { smiles, failed };
}

/** The SMILES in a molecule file, and how many records could not be read. */
export async function moleculesFromFile(file: File): Promise<{ smiles: string[]; failed: number }> {
  const text = await file.text();
  const kind = fileKind(file.name);
  if (kind === "smi") return { smiles: parseSmi(text), failed: 0 };
  if (kind === "csv") return { smiles: parseCsv(text), failed: 0 };
  if (kind === "sdf") return sdfToSmiles(await loadRDKit(), text);
  throw new Error(`${file.name} is not a molecule file`);
}
