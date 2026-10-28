// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { InlineCode, Link, Parent, Root, RootContent, Text } from "mdast";
import type { Panel } from "../types/api";
import type { Residue } from "./molstar";

/**
 * Residues named in the agent's answers ("the ASN140 hydrogen bond",
 * "Trp81", "CY0 797") become links that show the residue in the 3D viewer,
 * on the structure the answer was talking about there.
 */

const CODES =
  "Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val|Hid|Hie|Hip|Mse";

// Title case or upper case only: "Asn140" and "ASN140", not "asn140".
const CODE_PATTERN = CODES.split("|")
  .map((code) => `${code}|${code.toUpperCase()}`)
  .join("|");

// "ASN140", "Asn 140A" (insertion code), "Trp81.A", "Trp81/A", "LEU83 (chain B)".
const AMINO = `\\b(${CODE_PATTERN})[ -]?(\\d{1,4})([A-Z])?\\b(?:[.:/]([A-Z])\\b|\\s\\(chain ([A-Z0-9])\\))?`;
// A modified residue or adduct whose code ends in a digit, always written apart from its number: "CY0 797".
// A number followed by a decimal or Å is a distance after an atom name ("C48 4.4 Å").
const HETERO = `\\b((?=[A-Z0-9]{0,2}\\d)[A-Z][A-Z0-9]{2}) (\\d{1,4})(?![.,]?\\d)(?!\\s?Å)\\b`;
const IN_TEXT = new RegExp(`${AMINO}|${HETERO}`, "g");

export type Segment = { text: string } | { text: string; residue: Residue };

export function splitResidues(text: string): Segment[] {
  const out: Segment[] = [];
  let last = 0;
  for (const match of text.matchAll(IN_TEXT)) {
    const start = match.index ?? 0;
    if (start > last) out.push({ text: text.slice(last, start) });
    const residue: Residue = match[1]
      ? { chain: match[4] ?? match[5] ?? null, resnum: Number(match[2]), resname: match[1].toUpperCase() }
      : { chain: null, resnum: Number(match[7]), resname: match[6] };
    if (match[3]) residue.insCode = match[3];
    out.push({ text: match[0], residue });
    last = start + match[0].length;
  }
  if (last < text.length) out.push({ text: text.slice(last) });
  return out;
}

// ---------------------------------------------------------------------------
// which structure

/** "3MXF", "4ZC9_A", "mol_3" or "mol_0_model_0.pdb" in prose. */
const MENTION = /\b(mol_\d+)\b|([\w.-]+\.(?:pdb|cif|mmcif))\b|\b([1-9][A-Za-z0-9]{3})(?:_[A-Za-z0-9])?\b/g;

const normal = (path: string) => path.replace(/\\/g, "/").replace(/^\/?shared\//, "");
const baseName = (path: string) => normal(path).split("/").pop() ?? path;

/** Structures named in *text*, in order: PDB ids upper-cased, Boltz molecules and file names as written. */
export function structureMentions(text: string): string[] {
  const out: string[] = [];
  for (const match of text.matchAll(MENTION)) {
    if (match[1] || match[2]) out.push(match[1] ?? match[2]);
    else if (/[A-Za-z]/.test(match[3].slice(1))) out.push(match[3].toUpperCase());
  }
  return out;
}

const BOLTZ_MODEL = /\/(mol_\d+)\/\1_model_(\d+)(\.\w+)$/;

/**
 * The first of *paths* that *ref* names: the path itself, its file name, its
 * Boltz molecule or its PDB id. Batch indices repeat across runs, so a Boltz
 * molecule ("mol_3") said *near* a prediction is that prediction's sibling.
 */
export function matchStructure(ref: string, paths: readonly string[], near?: string | null): string | null {
  const clean = normal(ref);
  if (/^mol_\d+$/.test(clean) && near && BOLTZ_MODEL.test(normal(near))) {
    const sibling = normal(near).replace(BOLTZ_MODEL, (_, _mol, model: string, ext: string) => `/${clean}/${clean}_model_${model}${ext}`);
    return paths.find((path) => normal(path) === sibling) ?? null;
  }
  const sameTail = (a: string, b: string) => a === b || a.endsWith(`/${b}`) || b.endsWith(`/${a}`);
  const found =
    (clean.includes("/") ? paths.find((path) => sameTail(normal(path), clean)) : undefined) ??
    (/\.(pdb|cif|mmcif)$/i.test(clean) ? paths.find((path) => baseName(path) === baseName(clean)) : undefined) ??
    (/^mol_\d+$/.test(clean) ? paths.find((path) => path.includes(`/${clean}/`) || baseName(path).startsWith(`${clean}_`)) : undefined) ??
    (/^[1-9][A-Z0-9]{3}$/i.test(clean) ? paths.find((path) => new RegExp(`^${clean}(?=[._-]|$)`, "i").test(baseName(path))) : undefined);
  return found ?? null;
}

const isStructurePath = (ref: string) => /\.(pdb|cif|mmcif)$/i.test(ref) && ref.includes("/");

/** The structure files a turn worked on, in order: those its tools were given and those they returned. */
export function turnStructures(steps: { name: string; arguments: Record<string, unknown> }[], panels: Panel[]): string[] {
  const out: string[] = [];
  const add = (value: unknown) => {
    if (typeof value === "string" && isStructurePath(value) && !out.includes(normal(value))) out.push(normal(value));
  };
  for (const step of steps) {
    if (step.name === "list_shared_assets") continue;
    for (const value of Object.values(step.arguments ?? {})) {
      if (Array.isArray(value)) value.forEach(add);
      else add(value);
    }
  }
  for (const panel of panels) {
    if (panel.kind === "structure") add(panel.path);
    else if (panel.kind === "interactions") add(panel.structurePath);
  }
  return out;
}

interface Target {
  structure: string | null;
  /** The turn's prediction, for a Boltz molecule that is its sibling. */
  near?: string;
}

/**
 * The structure a residue after the mentions *named* belongs to: the turn's
 * structure the last mention names, else the mention itself (resolved later
 * among the conversation's structures), else the turn's only structure.
 */
function structureFor(named: string | null, paths: readonly string[]): Target {
  if (named) {
    const path = matchStructure(named, paths);
    if (path) return { structure: path };
    if (!/^mol_\d+$/.test(named)) return { structure: named };
    const near = [...paths].reverse().find((p) => BOLTZ_MODEL.test(p));
    if (near) return { structure: named, near };
  }
  const distinct = [...new Set(paths)];
  return { structure: distinct.length === 1 ? distinct[0] : null };
}

// ---------------------------------------------------------------------------
// hrefs

const HREF_PREFIX = "#residue?";

/** "#residue?n=ASN&r=140&c=A&s=…"; the name tells the viewer which numbering the answer used. */
export function residueHref(residue: Residue, structure?: string | null, near?: string | null): string {
  const params = new URLSearchParams();
  if (residue.resname) params.set("n", residue.resname);
  params.set("r", String(residue.resnum));
  if (residue.insCode) params.set("i", residue.insCode);
  if (residue.chain) params.set("c", residue.chain);
  if (structure) params.set("s", structure);
  if (structure && near) params.set("w", near);
  return `${HREF_PREFIX}${params.toString()}`;
}

export function residueFromHref(href: string | undefined): Residue | null {
  if (!href?.startsWith(HREF_PREFIX)) return null;
  const params = new URLSearchParams(href.slice(HREF_PREFIX.length));
  const resnum = Number(params.get("r"));
  if (!Number.isInteger(resnum)) return null;
  const residue: Residue = { resnum, chain: params.get("c") || null };
  const resname = params.get("n");
  if (resname) residue.resname = resname;
  const insCode = params.get("i");
  if (insCode) residue.insCode = insCode;
  return residue;
}

/**
 * The structure a residue link points at: a file path, or a name ("3MXF") to
 * look up, with the prediction it was said near; null for the one on screen.
 */
export function structureFromHref(href: string | undefined): { path: string | null; ref: string | null; near: string | null } {
  const params = href?.startsWith(HREF_PREFIX) ? new URLSearchParams(href.slice(HREF_PREFIX.length)) : null;
  const value = params?.get("s");
  if (!value) return { path: null, ref: null, near: null };
  return isStructurePath(value) ? { path: value, ref: null, near: null } : { path: null, ref: value, near: params?.get("w") || null };
}

// ---------------------------------------------------------------------------
// the remark plugin

export interface ResidueLinkOptions {
  /** The structures the turn worked on, so residues go to the one the answer is discussing. */
  structures?: readonly string[];
}

/** Remark plugin: residue names in text become residue links. Code and existing links are left alone. */
export function remarkResidueLinks(options: ResidueLinkOptions = {}) {
  const paths = options.structures ?? [];
  return (tree: Root) => {
    let named: string | null = null;
    const note = (text: string) => {
      const mentions = structureMentions(text);
      if (mentions.length) named = mentions[mentions.length - 1];
    };
    const walk = (node: Parent) => {
      const next: RootContent[] = [];
      let changed = false;
      for (const child of node.children as RootContent[]) {
        if (child.type === "inlineCode") {
          note((child as InlineCode).value);
        } else if (child.type === "text") {
          const segments = splitResidues((child as Text).value);
          if (segments.length > 1 || (segments[0] && "residue" in segments[0])) {
            changed = true;
            for (const segment of segments) {
              if ("residue" in segment) {
                const target = structureFor(named, paths);
                const link: Link = {
                  type: "link",
                  url: residueHref(segment.residue, target.structure, target.near),
                  title: "Show in the 3D viewer",
                  children: [{ type: "text", value: segment.text }],
                };
                next.push(link);
              } else {
                note(segment.text);
                next.push({ type: "text", value: segment.text });
              }
            }
            continue;
          }
          note((child as Text).value);
        } else if ("children" in child && child.type !== "link" && child.type !== "linkReference") {
          walk(child as Parent);
        }
        next.push(child);
      }
      if (changed) node.children = next as Parent["children"];
    };
    walk(tree);
  };
}
