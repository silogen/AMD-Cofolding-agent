// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { Contact } from "../types/api";
import type { ContactLine } from "./molstar";

/** Interaction types of the profiler, with the colour each one has everywhere in the UI. */
export const CONTACT_TYPES: Record<string, { label: string; color: string }> = {
  hbond: { label: "Hydrogen bond", color: "#00c2de" },
  hydrophobic: { label: "Hydrophobic", color: "#c1a968" },
  pi_stacking: { label: "π-stacking", color: "#b57bff" },
  pi_cation: { label: "π-cation", color: "#ff6fb5" },
  cation_pi: { label: "Cation-π", color: "#ff6fb5" },
  ionic: { label: "Salt bridge", color: "#f26522" },
  halogen_bond: { label: "Halogen bond", color: "#3ccf91" },
  chalcogen_bond: { label: "Chalcogen bond", color: "#7dd3fc" },
  metal: { label: "Metal", color: "#f5c542" },
  water_bridge: { label: "Water bridge", color: "#5b8def" },
  ch_pi: { label: "CH-π", color: "#a3a3a3" },
  chalcogen_pi: { label: "Chalcogen-π", color: "#93c5fd" },
  amide_pi: { label: "Amide-π", color: "#d8b4fe" },
  vdw_contact: { label: "van der Waals", color: "#6b7280" },
  covalent: { label: "Covalent", color: "#ed1c24" },
};

export const contactLabel = (type: string) => CONTACT_TYPES[type]?.label ?? type.replace(/_/g, " ");
/** Light-theme overrides: the dark-theme gold and yellow wash out on white. */
const LIGHT_COLORS: Record<string, string> = {
  hydrophobic: "#a38a3e",
  metal: "#d6a300",
};

const currentTheme = (): "dark" | "light" =>
  typeof document !== "undefined" && !document.documentElement.classList.contains("dark") ? "light" : "dark";

/** The colour of an interaction type; *theme* defaults to the one on the document. */
export function contactColor(type: string, theme: "dark" | "light" = currentTheme()): string {
  if (theme === "light" && LIGHT_COLORS[type]) return LIGHT_COLORS[type];
  return CONTACT_TYPES[type]?.color ?? "#888888";
}

/** Types most worth reading first: specific interactions before generic contacts. */
const ORDER = ["covalent", "ionic", "hbond", "halogen_bond", "pi_stacking", "pi_cation", "cation_pi", "metal", "water_bridge"];
const rank = (type: string) => {
  const index = ORDER.indexOf(type);
  return index < 0 ? ORDER.length : index;
};

export function sortedTypes(counts: Record<string, number>): string[] {
  return Object.keys(counts).sort((a, b) => rank(a) - rank(b) || counts[b] - counts[a]);
}

/** The contact types a chip click leaves shown: none selected means all. */
export function toggleType(selected: string[], type: string): string[] {
  return selected.includes(type) ? selected.filter((t) => t !== type) : [...selected, type];
}

export const matchesTypes = (contact: Contact, selected: string[]) => selected.length === 0 || selected.includes(contact.type);

/** Up to this many dashes carry their distance; more would be unreadable. */
const LABELLED = 10;
/** Contacts any close packing makes, as opposed to directed interactions. */
const GENERIC = new Set(["hydrophobic", "vdw_contact"]);

/**
 * One dashed line per contact, the most specific type first when two share
 * the same pair of atoms (a CH-π contact is also a hydrophobic one).
 */
export function contactLines(contacts: Contact[], theme: "dark" | "light"): ContactLine[] {
  const seen = new Set<string>();
  const kept: Contact[] = [];
  const generic = (type: string) => (GENERIC.has(type) ? 1 : 0);
  for (const c of [...contacts].sort((a, b) => generic(a.type) - generic(b.type) || rank(a.type) - rank(b.type))) {
    const pair = `${c.chain}:${c.resnum}:${c.receptorAtom}|${c.ligandAtom ?? c.ligandAtomIndex}`;
    if (c.resnum === null || seen.has(pair)) continue;
    seen.add(pair);
    kept.push(c);
  }
  return kept.map((c) => {
    const strong = rank(c.type) < ORDER.length;
    const labelled = strong || kept.length <= LABELLED;
    return {
      chain: c.chain,
      resnum: c.resnum as number,
      receptorAtom: c.receptorAtom,
      ligandAtom: c.ligandAtom,
      ligandAtomIndex: c.ligandAtomIndex,
      color: contactColor(c.type, theme),
      strong,
      label: labelled && c.distance != null ? `${c.distance.toFixed(1)} Å` : null,
    };
  });
}

/** Distinct residues of a set of contacts, in the order they first appear. */
export function residuesOf(contacts: Contact[]): { chain: string | null; resnum: number }[] {
  const seen = new Set<string>();
  const out: { chain: string | null; resnum: number }[] = [];
  for (const contact of contacts) {
    if (contact.resnum === null) continue;
    const key = `${contact.chain}:${contact.resnum}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ chain: contact.chain, resnum: contact.resnum });
  }
  return out;
}
