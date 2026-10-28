// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { isPdbId } from "./rcsb";
import { readCif } from "./sharedFiles";
import { fetchFromRcsb } from "./uploads";

/**
 * What part of the target Boltz cofolds: the full UniProt sequence, a
 * residue range of it, or the construct of a deposited structure. Boltz cost
 * grows steeply with length, so a kinase domain folds in a fraction of the
 * time of the full-length protein.
 */

export type Region = "full" | "range" | "pdb";

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

/** A PDB entry's mmCIF, from RCSB, else through the gateway (which keeps a copy in the conversation's uploads). */
export async function fetchRcsbCif(id: string, session: string): Promise<string> {
  const code = id.trim().toUpperCase();
  if (!isPdbId(code)) throw new Error(`“${id}” is not a PDB id.`);
  const direct = await fetch(rcsbCifUrl(code)).catch(() => null);
  if (direct?.ok) return direct.text();
  if (direct?.status === 404) throw new Error(`${code} is not in the PDB.`);
  const file = await fetchFromRcsb(session, code);
  const response = await fetch(file.url);
  if (!response.ok) throw new Error(`Could not fetch ${code} (HTTP ${response.status}).`);
  return response.text();
}
