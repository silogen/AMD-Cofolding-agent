// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * Spotting SMILES in free text, without RDKit.
 *
 * This only decides what looks like a SMILES; whether it is a valid
 * molecule is RDKit's call. Gene names (BRD4), compound codes (JQ1), CO2 and
 * English words must not turn into molecules, so a token has to be made of
 * SMILES atoms only and show some structure: a ring bond, a branch, a
 * bracket atom, a bond symbol, or at least two carbons.
 */

const ORGANIC = /^(Cl|Br|[BCNOPSFI]|[bcnops])/;
const RING = /^%[0-9]{2}|^[0-9]/;
const BOND = /^[-=#$:/\\.]/;
const LEADING = /^["'`“”‘’(]+/;
const TRAILING = /["'`“”‘’,;:!?.]+$/;

/** Whether *token* is written like a SMILES: atoms, bonds, rings and branches only, balanced. */
export function looksLikeSmiles(token: string): boolean {
  if (token.length < 3 || token.length > 500) return false;
  let depth = 0;
  let atoms = 0;
  let carbons = 0;
  let structured = false;
  const rings = new Map<string, number>();
  let index = 0;
  while (index < token.length) {
    const rest = token.slice(index);
    const char = token[index];
    if (char === "[") {
      const end = token.indexOf("]", index);
      if (end < 0 || end === index + 1) return false;
      atoms += 1;
      structured = true;
      index = end + 1;
      continue;
    }
    const atom = ORGANIC.exec(rest);
    if (atom) {
      atoms += 1;
      if (atom[0] === "C") carbons += 1;
      index += atom[0].length;
      continue;
    }
    if (atoms === 0) return false;
    if (char === "(" || char === ")") {
      depth += char === "(" ? 1 : -1;
      if (depth < 0) return false;
      structured = true;
      index += 1;
      continue;
    }
    if (char === "*") {
      atoms += 1;
      index += 1;
      continue;
    }
    const ring = RING.exec(rest);
    if (ring) {
      rings.set(ring[0], (rings.get(ring[0]) ?? 0) + 1);
      structured = true;
      index += ring[0].length;
      continue;
    }
    const bond = BOND.exec(rest);
    if (!bond) return false;
    structured = true;
    index += 1;
  }
  // A digit that closes no ring in a short token is a formula (CO2), not a typo.
  if (atoms < 4 && [...rings.values()].some((uses) => uses % 2 === 1)) return false;
  return depth === 0 && atoms >= 2 && (structured || carbons >= 2);
}

/** The SMILES-looking tokens of *text*, in order, each once. */
export function findSmiles(text: string): string[] {
  const out: string[] = [];
  for (const raw of text.split(/[\s,;]+/)) {
    let token = raw.replace(LEADING, "").replace(TRAILING, "");
    // "(CCO)" lost its opening bracket above; drop the closing one too.
    while (token.endsWith(")") && count(token, ")") > count(token, "(")) token = token.slice(0, -1).replace(TRAILING, "");
    if (looksLikeSmiles(token) && !out.includes(token)) out.push(token);
  }
  return out;
}

function count(text: string, char: string): number {
  return text.split(char).length - 1;
}

/** *text* with the first stand-alone occurrence of *from* replaced by *to*. */
export function replaceToken(text: string, from: string, to: string): string {
  const escaped = from.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const pattern = new RegExp(`(^|[\\s,;"'(\`“‘])${escaped}(?=$|[\\s,;"'.)\`!?:”’])`);
  return text.replace(pattern, (_, lead: string) => lead + to);
}
