// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { oneLetter } from "../../lib/numbering";

/**
 * What a residue clicked in the viewer is: the ligand (a non-polymer group
 * such as MI1), a residue of the chain that carries a covalently bound
 * ligand (CY0, a Cys797 fused with its inhibitor), a known modified amino
 * acid (PTR), a standard residue, or a water or ion.
 */
export type PickKind = "ligand" | "adduct" | "modified" | "residue" | "solvent";

/** Modified amino acids that belong to the protein rather than carry a ligand. */
const MODIFIED_RESIDUES = new Set([
  "PTR", "SEP", "TPO", "MSE", "HYP", "MLY", "M3L", "MLZ", "ALY", "KCX", "CSO", "CSD", "CSS", "CME", "OCS", "CAS",
  "SCH", "SCY", "SMC", "CSX", "SNC", "NEP", "HIC", "CGU", "PCA", "FME", "TYS", "CXM", "ABA", "NLE", "AIB", "ORN",
  "DAL", "MEN", "SAC", "AGM", "MHO", "OAS", "TYI", "IYR", "HTR", "TRO", "BMT",
]);

const SOLVENT = new Set([
  "HOH", "WAT", "DOD", "H2O", "SOL", "NA", "K", "CL", "BR", "IOD", "MG", "CA", "ZN", "MN", "FE", "FE2", "CU", "CO", "NI", "CD", "SO4", "PO4",
]);

/** Names Boltz and common pipelines give a docked or cofolded ligand. */
const GENERIC_LIGANDS = new Set(["LIG", "LIG1", "UNL", "UNK", "MOL", "INH", "DRG"]);

/**
 * *inPolymer* says whether the structure lists the residue in a polymer
 * chain; null when that is not known yet.
 */
export function pickKind(resname: string, inPolymer: boolean | null): PickKind {
  const name = resname.toUpperCase();
  if (oneLetter(name)) return "residue";
  if (GENERIC_LIGANDS.has(name)) return inPolymer ? "residue" : "ligand";
  if (SOLVENT.has(name)) return "solvent";
  if (MODIFIED_RESIDUES.has(name)) return "modified";
  if (inPolymer === null) return "residue";
  return inPolymer ? "adduct" : "ligand";
}

/** "2J5E (uploads/s-.../2J5E.cif)": the name and where the agent finds the file. */
export function structureRef(title: string, path: string | null | undefined): string {
  if (!path || path === title) return title;
  return `${title} (${path})`;
}

export interface PickedResidue {
  kind: PickKind;
  resname: string;
  /** As shown, e.g. "LEU94" or "CY0 797". */
  label: string;
  chain: string | null;
  /** The residue's number in the file when the label shows another (UniProt) number. */
  file: number | null;
}

/** How the picked residue is named in the button and the question. */
export function pickTitle(pick: PickedResidue): string {
  return pick.kind === "ligand" && GENERIC_LIGANDS.has(pick.resname.toUpperCase()) ? "the ligand" : pick.label;
}

/** The question about a picked residue, naming the file so the agent need not search for it. */
export function pickQuestion(title: string, path: string | null | undefined, pick: PickedResidue): string {
  const where = structureRef(title, path);
  const details = [pick.chain ? `chain ${pick.chain}` : null, pick.file != null ? `residue ${pick.file} in the file` : null].filter(Boolean);
  const named = `${pick.label}${details.length ? ` (${details.join(", ")})` : ""}`;
  switch (pick.kind) {
    case "ligand":
      return pickTitle(pick) === "the ligand"
        ? `In ${where}, how is the ligand held in the pocket? Name its key contacts from the interaction profile.`
        : `In ${where}, how is the ligand ${named} held in the pocket? Profile that ligand and name its key contacts.`;
    case "adduct":
      return (
        `In ${where}, ${named} is a non-standard residue of the protein chain, likely an amino acid with a covalently bound ligand. ` +
        `Which residue does it modify, what is attached to it, and how is that ligand held in the pocket? Use the interaction profile.`
      );
    case "modified":
      return `In ${where}, what role does the modified residue ${named} play near the ligand? Use the interaction profile.`;
    default:
      return `In ${where}, what role does ${named} play in binding the ligand? Use the interaction profile.`;
  }
}
