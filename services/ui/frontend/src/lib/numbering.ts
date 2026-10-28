// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useSyncExternalStore } from "react";

/**
 * Residue numbers in a structure file and in the UniProt sequence of its
 * protein. Boltz numbers residues from 1 along the sequence it was given,
 * deposited structures mostly follow UniProt but skip missing loops and
 * carry mutations; the UI speaks UniProt so numbers match the literature.
 */

/** One polymer residue of the structure file, in file order. */
export interface FileResidue {
  chain: string | null;
  resnum: number;
  /** Three-letter code, e.g. "ASN". */
  resname: string;
}

export interface Numbering {
  /** "chain:resnum" in the file to the UniProt number. */
  toUniprot: Map<string, number>;
  /** "chain:uniprot" to the file number. */
  toFile: Map<string, number>;
  /** "chain:resnum" in the file to its residue name. */
  names: Map<string, string>;
  /** True when every mapped residue keeps its number. */
  same: boolean;
}

const ONE_LETTER: Record<string, string> = {
  ALA: "A", ARG: "R", ASN: "N", ASP: "D", CYS: "C", GLN: "Q", GLU: "E", GLY: "G", HIS: "H", ILE: "I",
  LEU: "L", LYS: "K", MET: "M", PHE: "F", PRO: "P", SER: "S", THR: "T", TRP: "W", TYR: "Y", VAL: "V",
  HID: "H", HIE: "H", HIP: "H", HSD: "H", HSE: "H", HSP: "H", CYX: "C", MSE: "M", SEC: "U", PYL: "O",
};

export const oneLetter = (resname: string) => ONE_LETTER[resname.toUpperCase()] ?? null;

const key = (chain: string | null, resnum: number) => `${chain ?? ""}:${resnum}`;

/** Shortest exact match that anchors a chain to the sequence. */
const SEED = 6;
/** Below this many residues in exact matches, a chain is not mapped. */
const MIN_MATCHED = 12;
/** And below this share of the chain's residues. */
const MIN_COVERAGE = 0.6;
/** Unmatched residues past the last match that still follow its diagonal (point mutations at an end). */
const END_SLACK = 3;

interface Match {
  s: number;
  u: number;
  len: number;
}

/** Maximal exact matches of at least SEED letters between *s* and *u*. */
function exactMatches(s: string, u: string): Match[] {
  const index = new Map<string, number[]>();
  for (let j = 0; j + SEED <= u.length; j++) {
    const kmer = u.slice(j, j + SEED);
    const list = index.get(kmer);
    if (list) list.push(j);
    else index.set(kmer, [j]);
  }
  const out: Match[] = [];
  for (let i = 0; i + SEED <= s.length; i++) {
    const hits = index.get(s.slice(i, i + SEED));
    // Low-complexity repeats (poly-Q, His tags) anchor nothing.
    if (!hits || hits.length > 20) continue;
    for (const j of hits) {
      if (i > 0 && j > 0 && s[i - 1] === u[j - 1] && s[i - 1] !== "X") continue;
      let len = SEED;
      while (i + len < s.length && j + len < u.length && s[i + len] === u[j + len] && s[i + len] !== "X") len++;
      out.push({ s: i, u: j, len });
    }
  }
  return out;
}

/**
 * The longest matches that agree in order, sorted along the chain. A match
 * that runs into a longer one is trimmed: at an indel the two flanking
 * matches often share a letter or two.
 */
function chain(matches: Match[]): Match[] {
  const kept: Match[] = [];
  for (const original of [...matches].sort((a, b) => b.len - a.len)) {
    const m = { ...original };
    for (const k of kept) {
      if (k.s <= m.s) {
        const cut = Math.max(0, k.s + k.len - m.s, k.u + k.len - m.u);
        m.s += cut;
        m.u += cut;
        m.len -= cut;
      } else {
        m.len -= Math.max(0, m.s + m.len - k.s, m.u + m.len - k.u);
      }
    }
    const fits = kept.every((k) => (m.s + m.len <= k.s && m.u + m.len <= k.u) || (k.s + k.len <= m.s && k.u + k.len <= m.u));
    if (m.len >= SEED && fits) kept.push(m);
  }
  return kept.sort((a, b) => a.s - b.s);
}

/**
 * Position in *uniprot* (0-based) of each residue of one chain, or null for
 * residues with no confident counterpart; null overall when the chain is not
 * this protein.
 */
export function alignChain(sequence: string, uniprot: string): (number | null)[] | null {
  const n = sequence.length;
  if (!n || !uniprot) return null;
  const out: (number | null)[] = new Array<number | null>(n).fill(null);
  const exact = uniprot.indexOf(sequence);
  if (exact >= 0 && !sequence.includes("X")) return out.map((_, i) => exact + i);

  const anchors = chain(exactMatches(sequence, uniprot));
  const matched = anchors.reduce((sum, m) => sum + m.len, 0);
  if (matched < MIN_MATCHED || matched < MIN_COVERAGE * n) return null;

  for (const m of anchors) for (let i = 0; i < m.len; i++) out[m.s + i] = m.u + i;
  // Between two anchors on the same diagonal the gap is a run of mutations.
  for (let a = 0; a + 1 < anchors.length; a++) {
    const [left, right] = [anchors[a], anchors[a + 1]];
    if (right.u - right.s !== left.u - left.s) continue;
    for (let i = left.s + left.len; i < right.s; i++) out[i] = left.u - left.s + i;
  }
  const first = anchors[0];
  for (let i = Math.max(0, first.s - END_SLACK); i < first.s; i++) {
    const u = first.u - first.s + i;
    if (u >= 0) out[i] = u;
  }
  const last = anchors[anchors.length - 1];
  for (let i = last.s + last.len; i < Math.min(n, last.s + last.len + END_SLACK); i++) {
    const u = last.u - last.s + i;
    if (u < uniprot.length) out[i] = u;
  }
  return out;
}

/**
 * Map the file's residues to UniProt numbers, chain by chain. Null when no
 * chain is confidently the protein of *uniprot*.
 */
export function buildNumbering(residues: FileResidue[], uniprot: string | null | undefined): Numbering | null {
  const target = (uniprot ?? "").toUpperCase().replace(/[^A-Z]/g, "");
  if (!target) return null;
  const byChain = new Map<string, FileResidue[]>();
  for (const residue of residues) {
    if (!oneLetter(residue.resname)) continue;
    const list = byChain.get(residue.chain ?? "") ?? [];
    list.push(residue);
    byChain.set(residue.chain ?? "", list);
  }
  const numbering: Numbering = { toUniprot: new Map(), toFile: new Map(), names: new Map(), same: true };
  for (const list of byChain.values()) {
    const positions = alignChain(list.map((r) => oneLetter(r.resname) ?? "X").join(""), target);
    if (!positions) continue;
    list.forEach((residue, i) => {
      const position = positions[i];
      numbering.names.set(key(residue.chain, residue.resnum), residue.resname.toUpperCase());
      if (position === null) return;
      numbering.toUniprot.set(key(residue.chain, residue.resnum), position + 1);
      numbering.toFile.set(key(residue.chain, position + 1), residue.resnum);
      if (position + 1 !== residue.resnum) numbering.same = false;
    });
  }
  return numbering.toUniprot.size ? numbering : null;
}

/** Look a residue up by chain, or in any chain when the chain is not given. */
function lookup(map: Map<string, number>, chain: string | null | undefined, resnum: number): number | null {
  if (chain) return map.get(key(chain, resnum)) ?? null;
  const direct = map.get(key(null, resnum));
  if (direct !== undefined) return direct;
  for (const [k, value] of map) if (k.endsWith(`:${resnum}`)) return value;
  return null;
}

export function toUniprot(numbering: Numbering | null, chain: string | null | undefined, resnum: number): number | null {
  return numbering ? lookup(numbering.toUniprot, chain, resnum) : null;
}

/** The residue's number to show: UniProt when mapped, else the file's. */
export function displayNumber(numbering: Numbering | null, chain: string | null | undefined, resnum: number): number {
  return toUniprot(numbering, chain, resnum) ?? resnum;
}

/**
 * "LEU94", with the file number when it differs: { label: "LEU94", file: 51 }.
 * A name ending in a digit is kept apart from the number: "CY0 797", not "CY0797".
 */
export function residueLabel(
  numbering: Numbering | null,
  resname: string,
  chain: string | null | undefined,
  resnum: number,
): { label: string; file: number | null } {
  const shown = displayNumber(numbering, chain, resnum);
  const label = /\d$/.test(resname) ? `${resname} ${shown}` : `${resname}${shown}`;
  return { label, file: shown === resnum ? null : resnum };
}

/**
 * The file number of a residue someone named: as written when the file has
 * that residue under that name, else read as a UniProt number. Without a
 * name, the UniProt reading wins, as that is the numbering on screen.
 */
export function resolveResidue(
  numbering: Numbering | null,
  residue: { chain?: string | null; resnum: number; resname?: string | null },
): number {
  if (!numbering) return residue.resnum;
  const name = residue.resname?.toUpperCase();
  const fileName = lookupName(numbering, residue.chain, residue.resnum);
  const fromUniprot = lookup(numbering.toFile, residue.chain, residue.resnum);
  if (name) {
    if (fileName === name) return residue.resnum;
    if (fromUniprot !== null && lookupName(numbering, residue.chain, fromUniprot) === name) return fromUniprot;
    return residue.resnum;
  }
  return fromUniprot ?? residue.resnum;
}

function lookupName(numbering: Numbering, chain: string | null | undefined, resnum: number): string | null {
  if (chain) return numbering.names.get(key(chain, resnum)) ?? null;
  const direct = numbering.names.get(key(null, resnum));
  if (direct) return direct;
  for (const [k, value] of numbering.names) if (k.endsWith(`:${resnum}`)) return value;
  return null;
}

// ---------------------------------------------------------------------------
// The numberings of structures the viewer has loaded, by file path, so the
// contact cards in the conversation can speak the same numbers.

const known = new Map<string, Numbering | null>();
const listeners = new Set<() => void>();
let version = 0;

const normalise = (path: string) => path.replace(/\\/g, "/").replace(/^\/?shared\//, "").replace(/^\//, "");

export function rememberNumbering(path: string, numbering: Numbering | null): void {
  const k = normalise(path);
  if (known.get(k) === numbering) return;
  known.set(k, numbering);
  version++;
  listeners.forEach((listener) => listener());
}

export function numberingFor(path: string | null | undefined): Numbering | null {
  if (!path) return null;
  const k = normalise(path);
  const hit = known.get(k);
  if (hit !== undefined) return hit;
  for (const [stored, value] of known) if (stored.endsWith(`/${k}`) || k.endsWith(`/${stored}`)) return value;
  return null;
}

/** numberingFor, re-rendering when the viewer learns a new numbering. */
export function useNumbering(path: string | null | undefined): Numbering | null {
  useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => version,
  );
  return numberingFor(path);
}
