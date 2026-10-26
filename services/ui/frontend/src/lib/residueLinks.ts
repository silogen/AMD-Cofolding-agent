// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { Link, Parent, Root, RootContent, Text } from "mdast";
import type { Residue } from "./molstar";

/**
 * Residues named in the agent's answers ("the ASN140 hydrogen bond",
 * "Trp81") become links that show the residue in the 3D viewer.
 */

const CODES =
  "Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val|Hid|Hie|Hip|Mse";

// Title case or upper case only: "Asn140" and "ASN140", not "asn140".
const CODE_PATTERN = CODES.split("|")
  .map((code) => `${code}|${code.toUpperCase()}`)
  .join("|");

const IN_TEXT = new RegExp(`\\b(${CODE_PATTERN})[ -]?(\\d{1,4})(?:[.:/]([A-Z]))?\\b`, "g");

export type Segment = { text: string } | { text: string; residue: Residue };

export function splitResidues(text: string): Segment[] {
  const out: Segment[] = [];
  let last = 0;
  for (const match of text.matchAll(IN_TEXT)) {
    const start = match.index ?? 0;
    if (start > last) out.push({ text: text.slice(last, start) });
    out.push({ text: match[0], residue: { chain: match[3] ?? null, resnum: Number(match[2]) } });
    last = start + match[0].length;
  }
  if (last < text.length) out.push({ text: text.slice(last) });
  return out;
}

const HREF_PREFIX = "#residue/";

export function residueHref(residue: Residue): string {
  return `${HREF_PREFIX}${residue.resnum}${residue.chain ? `/${residue.chain}` : ""}`;
}

export function residueFromHref(href: string | undefined): Residue | null {
  if (!href?.startsWith(HREF_PREFIX)) return null;
  const [resnum, chain] = href.slice(HREF_PREFIX.length).split("/");
  const value = Number(resnum);
  return Number.isFinite(value) ? { resnum: value, chain: chain || null } : null;
}

/** Remark plugin: residue names in text become residue links. Code and existing links are left alone. */
export function remarkResidueLinks() {
  return (tree: Root) => {
    const walk = (node: Parent) => {
      const next: RootContent[] = [];
      let changed = false;
      for (const child of node.children as RootContent[]) {
        if (child.type === "text") {
          const segments = splitResidues((child as Text).value);
          if (segments.length > 1 || (segments[0] && "residue" in segments[0])) {
            changed = true;
            for (const segment of segments) {
              if ("residue" in segment) {
                const link: Link = {
                  type: "link",
                  url: residueHref(segment.residue),
                  title: "Show in the 3D viewer",
                  children: [{ type: "text", value: segment.text }],
                };
                next.push(link);
              } else {
                next.push({ type: "text", value: segment.text });
              }
            }
            continue;
          }
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
