// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { isPdbId } from "./rcsb";
import { readCif } from "./sharedFiles";
import { fetchFromRcsb, uploadFile, type UploadedFile } from "./uploads";

/**
 * What part of a looked-up target Boltz cofolds: the full UniProt sequence
 * or a residue range of it (a structure gives its own construct). Boltz cost
 * grows steeply with length, so a kinase domain folds in a fraction of the
 * time of the full-length protein.
 */

export type Region = "full" | "range";

/** The residues *from*..*to* (UniProt numbering, inclusive) of *reference*, or why they cannot be cut. */
export function sliceRegion(reference: string, from: number, to: number): { sequence: string } | { error: string } {
  if (!Number.isInteger(from) || !Number.isInteger(to)) return { error: "Residue numbers are whole numbers." };
  if (from < 1 || to > reference.length) return { error: `Residues run from 1 to ${reference.length}.` };
  if (from > to) return { error: "The first residue comes after the last." };
  if (to - from + 1 < 10) return { error: "At least 10 residues." };
  return { sequence: reference.slice(from - 1, to) };
}

/** Where a chain of a PDB entry sits in its UniProt entry. */
export interface UniprotSpan {
  accession: string | null;
  start: number;
  end: number;
}

const int = (value: string | null | undefined) => {
  const n = value == null ? NaN : Number.parseInt(value, 10);
  return Number.isFinite(n) ? n : null;
};

/** UniProt ranges of an mmCIF's chains (by author chain id), from struct_ref_seq; empty without the records. */
export function uniprotSpans(text: string): Map<string, UniprotSpan> {
  const tables = readCif(text, ["struct_ref", "struct_ref_seq"]);
  const refs = tables.get("struct_ref") ?? [];
  const uniprotRefs = new Set(refs.filter((r) => !r.db_name || /^(UNP|UNIPROT)$/i.test(r.db_name)).map((r) => r.id));
  const out = new Map<string, UniprotSpan>();
  for (const row of tables.get("struct_ref_seq") ?? []) {
    if (refs.length && !uniprotRefs.has(row.ref_id ?? null)) continue;
    const chain = row.pdbx_strand_id;
    const start = int(row.db_align_beg);
    const end = int(row.db_align_end);
    if (!chain || start === null || end === null) continue;
    const known = out.get(chain);
    out.set(chain, {
      accession: known?.accession ?? row.pdbx_db_accession ?? null,
      start: Math.min(known?.start ?? start, start),
      end: Math.max(known?.end ?? end, end),
    });
  }
  return out;
}

export const rcsbCifUrl = (id: string) => `https://files.rcsb.org/download/${id.trim().toUpperCase()}.cif`;

/** An uploaded file's path relative to the shared volume, as TTP paths are. */
export const sharedPath = (file: UploadedFile) => decodeURIComponent(file.url.replace(/^\/api\/artifacts\//, ""));

/** A structure file as text and where it sits on the shared volume, for the server to read it too. */
export interface SharedStructure {
  path: string;
  text: string;
}

/**
 * A PDB entry's mmCIF in the conversation's uploads: fetched by the gateway,
 * else by the browser from RCSB and uploaded.
 */
export async function sharedRcsbCif(id: string, session: string): Promise<SharedStructure> {
  const code = id.trim().toUpperCase();
  if (!isPdbId(code)) throw new Error(`“${id}” is not a PDB id.`);
  let file: UploadedFile;
  try {
    file = await fetchFromRcsb(session, code);
  } catch (err) {
    const direct = await fetch(rcsbCifUrl(code)).catch(() => null);
    if (direct?.status === 404) throw new Error(`${code} is not in the PDB.`, { cause: err });
    if (!direct?.ok) throw err;
    const text = await direct.text();
    file = await uploadFile(session, new File([text], `${code}.cif`, { type: "chemical/x-mmcif" }));
    return { path: sharedPath(file), text };
  }
  const response = await fetch(file.url);
  if (!response.ok) throw new Error(`Could not read ${code} (HTTP ${response.status}).`);
  return { path: sharedPath(file), text: await response.text() };
}

/** A structure from the user's disk, uploaded to the conversation. */
export async function uploadStructure(file: File, session: string): Promise<SharedStructure> {
  const uploaded = await uploadFile(session, file);
  return { path: sharedPath(uploaded), text: await file.text() };
}
