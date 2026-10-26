// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { BackgroundName, ColorName, Residue, ViewStyleName } from "./molstar";

/**
 * The viewer's command line: short plain-English instructions such as
 * "focus ASN140", "pocket", "surface", "color by chain" or "show hbonds".
 * They run in the browser, instantly, without asking the agent.
 */
export type ViewerAction =
  | { kind: "help" }
  | { kind: "fit" }
  | { kind: "ligand" }
  | { kind: "pocket"; radius: number }
  | { kind: "residues"; residues: Residue[]; mode: "focus" | "highlight" }
  | { kind: "contacts"; type: string | null }
  | { kind: "style"; style: ViewStyleName }
  | { kind: "color"; color: ColorName }
  | { kind: "background"; background: BackgroundName }
  | { kind: "spin"; on: boolean | null }
  | { kind: "clear" }
  | { kind: "controls"; on: boolean }
  | { kind: "screenshot" }
  | { kind: "download" }
  | { kind: "panel"; tab: "contacts" | "quality" | "files" }
  | { kind: "structure"; step: 1 | -1 };

export const VIEWER_EXAMPLES = [
  "fit",
  "focus ligand",
  "pocket",
  "focus ASN140",
  "show hbonds",
  "surface",
  "color by chain",
  "spin",
  "background light",
  "screenshot",
];

export const VIEWER_HELP =
  "fit · focus ligand · pocket [6 Å] · focus ASN140 LEU92 · highlight 140 · show hbonds / contacts · " +
  "cartoon / surface / atoms / illustrative · color by chain / element / hydrophobicity / confidence · " +
  "background dark / light · spin / stop · clear · screenshot · download · contacts / quality / files · next / previous";

const RESIDUE_CODES =
  "ALA|ARG|ASN|ASP|CYS|GLN|GLU|GLY|HIS|HID|HIE|HIP|ILE|LEU|LYS|MET|PHE|PRO|SER|THR|TRP|TYR|VAL|SEC|PYL|MSE";

/** "ASN140", "Asn 140", "140", "A:140", "140.A", "ASN140:A". */
const RESIDUE_TOKEN = new RegExp(
  `(?:([A-Za-z0-9]):)?(?:(${RESIDUE_CODES})\\s?)?(-?\\d{1,4})(?:[.:]([A-Za-z0-9]))?\\b`,
  "gi",
);

export function parseResidues(text: string): Residue[] {
  const out: Residue[] = [];
  for (const match of text.matchAll(RESIDUE_TOKEN)) {
    const chain = (match[1] ?? match[4])?.toUpperCase() ?? null;
    out.push({ chain, resnum: Number(match[3]) });
  }
  return out;
}

const CONTACT_WORDS: [RegExp, string | null][] = [
  [/^(h[- ]?bonds?|hydrogen bonds?)$/, "hbond"],
  [/^(hydrophobic|hydrophobics|hydrophobic contacts)$/, "hydrophobic"],
  [/^(ionic|salt bridges?)$/, "ionic"],
  [/^(pi[- ]?stacking|stacking|pi stacks?)$/, "pi_stacking"],
  [/^(halogen bonds?|halogen)$/, "halogen_bond"],
  [/^(pi[- ]?cation|cation[- ]?pi)$/, "pi_cation"],
  [/^(metal|metals|metal coordination)$/, "metal"],
  [/^(water bridges?|waters)$/, "water_bridge"],
  [/^(contacts|interactions|all contacts)$/, null],
];

const STYLE_WORDS: Record<string, ViewStyleName> = {
  cartoon: "cartoon",
  ribbon: "cartoon",
  default: "cartoon",
  surface: "surface",
  "molecular surface": "surface",
  atoms: "sticks",
  sticks: "sticks",
  "ball and stick": "sticks",
  "all atoms": "sticks",
  illustrative: "illustrative",
  cartoonish: "illustrative",
};

const COLOR_WORDS: Record<string, ColorName> = {
  chain: "chain",
  chains: "chain",
  element: "element",
  elements: "element",
  atom: "element",
  residue: "residue",
  "residue type": "residue",
  hydrophobicity: "hydrophobicity",
  hydrophobic: "hydrophobicity",
  confidence: "confidence",
  plddt: "confidence",
  "b-factor": "confidence",
  bfactor: "confidence",
  secondary: "secondary",
  "secondary structure": "secondary",
};

function normalise(text: string): string {
  return text
    .toLowerCase()
    .replace(/[’'`]/g, "")
    .replace(/[.!?]+$/g, "")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^(please |can you |could you )/, "")
    .replace(/^(the )/, "");
}

/** Parse one command. Null when it is not a viewer command. */
export function parseViewerCommand(input: string): ViewerAction | null {
  const t = normalise(input);
  if (!t) return null;

  if (/^(help|\?|commands)$/.test(t)) return { kind: "help" };
  if (/^(fit|reset|reset view|home|zoom out|center|centre|show all)$/.test(t)) return { kind: "fit" };
  if (/^((focus|zoom|center|centre|go)( on| to)? )?(the )?(ligand|compound|drug|molecule)$/.test(t)) return { kind: "ligand" };

  const pocket = /^((focus|zoom|show)( on| to)? )?(the )?(pocket|binding site|binding pocket|active site)(?: within)? ?(\d+(?:\.\d+)?)? ?(a|å|angstroms?)?$/.exec(t);
  if (pocket) return { kind: "pocket", radius: pocket[6] ? Math.min(12, Number(pocket[6])) : 5 };

  const contacts = /^(?:show|display|highlight)? ?(?:the )?(.+)$/.exec(t);
  if (contacts) {
    for (const [pattern, type] of CONTACT_WORDS) if (pattern.test(contacts[1])) return { kind: "contacts", type };
  }

  const style = /^(?:show |display |style |use |as |switch to )?(?:as )?(.+?)(?: view| style| representation)?$/.exec(t);
  if (style && STYLE_WORDS[style[1]]) return { kind: "style", style: STYLE_WORDS[style[1]] };

  const color = /^colou?r(?: by| as)? (.+)$/.exec(t);
  if (color && COLOR_WORDS[color[1]]) return { kind: "color", color: COLOR_WORDS[color[1]] };

  const background = /^(?:background|bg)(?: to)? (dark|black|light|white)$/.exec(t);
  if (background) return { kind: "background", background: background[1] as BackgroundName };

  if (/^(spin|rotate|turn|start spinning)$/.test(t)) return { kind: "spin", on: null };
  if (/^(stop|stop spinning|stop rotating|still)$/.test(t)) return { kind: "spin", on: false };
  if (/^(clear|deselect|clear selection|unselect)$/.test(t)) return { kind: "clear" };
  if (/^(show|open|display) (mol\* |molstar )?(controls?|panel|tools)$/.test(t)) return { kind: "controls", on: true };
  if (/^(hide|close) (mol\* |molstar )?(controls?|panel|tools)$/.test(t)) return { kind: "controls", on: false };
  if (/^(screenshot|snapshot|image|save image|picture|png)$/.test(t)) return { kind: "screenshot" };
  if (/^(download|save|download (pdb|cif|structure|file))$/.test(t)) return { kind: "download" };
  if (/^((show|open) )?(details|contacts list|contact list|quality|confidence details|files|file)$/.test(t)) {
    const tab = /quality|confidence/.test(t) ? "quality" : /file/.test(t) ? "files" : "contacts";
    return { kind: "panel", tab };
  }
  if (/^(next|forward)( structure| pose| model| complex)?$/.test(t)) return { kind: "structure", step: 1 };
  if (/^(previous|prev|back)( structure| pose| model| complex)?$/.test(t)) return { kind: "structure", step: -1 };

  const residues = /^(focus|zoom|show|go to|select|highlight|center|centre)(?: on| to)?(?: residues?)? (.+)$/.exec(t);
  if (residues) {
    const found = parseResidues(residues[2]);
    if (found.length) return { kind: "residues", residues: found, mode: residues[1] === "highlight" ? "highlight" : "focus" };
  }
  // A bare residue list: "ASN140 LEU92".
  if (/^[a-z0-9:.,\s]+$/.test(t) && /\d/.test(t)) {
    const found = parseResidues(t);
    if (found.length) return { kind: "residues", residues: found, mode: "focus" };
  }
  return null;
}
