// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { Residue } from "./molstar";
import { oneLetter, resolveResidue, type FileResidue, type Numbering } from "./numbering";

/** A residue as the viewer addresses it, in the file's numbering. */
export interface Located {
  chain: string | null;
  resnum: number;
  insCode: string | null;
}

/** Same residue type: equal codes, or variants of one amino acid (HIS and HIE, MET and MSE). */
const sameType = (a: string, b: string) => {
  const [x, y] = [a.toUpperCase(), b.toUpperCase()];
  if (x === y) return true;
  const [p, q] = [oneLetter(x), oneLetter(y)];
  return p !== null && p === q;
};

/**
 * Where a named residue is in a loaded structure. *residues* are its polymer
 * residues; a residue outside them (a ligand, a hetero group) is left to the
 * viewer to find. Null when it is certainly not there: no residue at that
 * number, or one of another type, as when a link names a residue of another
 * protein or in another numbering.
 *
 * A chain the file does not have is dropped, and a trailing letter that is
 * not an insertion code in the file is read as the chain ("ASN140A").
 */
export function locateResidue(residues: FileResidue[], numbering: Numbering | null, wanted: Residue): Located[] | null {
  const chains = new Set(residues.map((r) => r.chain));
  const insCodes = new Set(residues.flatMap((r) => (r.insCode ? [r.insCode] : [])));
  let chain = wanted.chain && (!chains.size || chains.has(wanted.chain)) ? wanted.chain : null;
  let insCode = wanted.insCode ?? null;
  if (insCode && !insCodes.has(insCode)) {
    if (!chain && chains.has(insCode)) chain = insCode;
    insCode = null;
  }
  const resnum = resolveResidue(numbering, { ...wanted, chain });
  if (!residues.length) return [{ chain, resnum, insCode }];
  const find = (inChain: string | null) => {
    const here = residues.filter((r) => r.resnum === resnum && (!inChain || r.chain === inChain) && (r.insCode ?? null) === insCode);
    return wanted.resname ? here.filter((r) => sameType(r.resname, wanted.resname as string)) : here;
  };
  const named = find(chain);
  const found = named.length || !chain ? named : find(null);
  if (found.length) return found.map((r) => ({ chain: r.chain, resnum: r.resnum, insCode: r.insCode ?? null }));
  // Not a residue of the polymer: a ligand or hetero group the viewer may still find.
  const atNumber = residues.some((r) => r.resnum === resnum && (r.insCode ?? null) === insCode);
  return atNumber || (wanted.resname && oneLetter(wanted.resname)) ? null : [{ chain, resnum, insCode }];
}
