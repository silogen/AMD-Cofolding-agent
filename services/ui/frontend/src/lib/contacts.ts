// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { Contact } from "../types/api";

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
export const contactColor = (type: string) => CONTACT_TYPES[type]?.color ?? "#888888";

/** Types most worth reading first: specific interactions before generic contacts. */
const ORDER = ["covalent", "ionic", "hbond", "halogen_bond", "pi_stacking", "pi_cation", "cation_pi", "metal", "water_bridge"];

export function sortedTypes(counts: Record<string, number>): string[] {
  const rank = (type: string) => {
    const index = ORDER.indexOf(type);
    return index < 0 ? ORDER.length : index;
  };
  return Object.keys(counts).sort((a, b) => rank(a) - rank(b) || counts[b] - counts[a]);
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
