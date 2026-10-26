// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * Grouping of structure files into poses of one molecule, and what kind of
 * structure a file is, both from its path alone.
 */

/** Suffixes that number the samples of one prediction, e.g. mol_0_model_1.pdb. */
const SAMPLE = /[_-](?:model|sample|pose|rank|seed)[_-]?(\d+)$/i;

const clean = (path: string) => path.replace(/\\/g, "/");

/** The molecule a structure is a sample of (directory plus stem without the sample number), and its sample number. */
export function poseKey(path: string): { key: string; sample: number | null } {
  const p = clean(path);
  const slash = p.lastIndexOf("/");
  const dir = p.slice(0, slash + 1);
  const stem = p.slice(slash + 1).replace(/\.[a-z0-9]+$/i, "");
  const match = SAMPLE.exec(stem);
  if (!match) return { key: dir + stem, sample: null };
  return { key: dir + stem.slice(0, match.index), sample: Number(match[1]) };
}

/**
 * The paths to step through from *path*: the other samples of the same
 * molecule and run, in sample order, or every path when it has no siblings.
 */
export function poseSiblings(paths: string[], path: string): { paths: string[]; grouped: boolean } {
  const { key } = poseKey(path);
  const siblings = paths
    .map((p) => ({ p, ...poseKey(p) }))
    .filter((entry) => entry.key === key && entry.sample !== null)
    .sort((a, b) => (a.sample ?? 0) - (b.sample ?? 0))
    .map((entry) => entry.p);
  if (siblings.length > 1 && siblings.includes(path)) return { paths: siblings, grouped: true };
  return { paths, grouped: false };
}

/** The neighbour *step* places away in *paths*, wrapping around. */
export function stepPose(paths: string[], path: string, step: number): string | null {
  if (paths.length === 0) return null;
  const index = Math.max(0, paths.indexOf(path));
  return paths[(index + step + paths.length * Math.abs(step)) % paths.length];
}

export type StructureKind = "predicted" | "experimental" | "prepared";

export const STRUCTURE_KINDS: Record<StructureKind, { label: string; color: string; hint: string }> = {
  predicted: { label: "Predicted", color: "var(--color-molecule)", hint: "Cofolding output (Boltz)" },
  experimental: { label: "Experimental", color: "var(--color-structure)", hint: "Deposited structure (RCSB PDB)" },
  prepared: { label: "Prepared", color: "var(--color-table)", hint: "Output of structure preparation" },
};

/** What kind of structure a file is, inferred from its path; null when the path does not say. */
export function structureKind(path: string, predictedMetrics = false, tool?: string): StructureKind | null {
  if (tool === "rcsb") return "experimental";
  const p = clean(path).toLowerCase();
  const name = p.slice(p.lastIndexOf("/") + 1);
  if (/(^|\/)(prep|prepared|structure_prep|prepped)[^/]*\/|_(prep|prepared|fixed|protonated|clean(ed)?)\.(pdb|cif|mmcif)$/.test(p)) return "prepared";
  if (/(^|\/)(predictions|boltz_results[^/]*|cofold[^/]*)\//.test(p) || predictedMetrics) return "predicted";
  if (/(^|\/)(rcsb|pdb|pdb_files)\//.test(p) || /^(pdb)?[0-9][a-z0-9]{3}\.(pdb|cif|mmcif|ent)(\.gz)?$/.test(name)) return "experimental";
  return null;
}

/** A structure in a one-line list: "JQ1 · 7aa793db · pose 2". */
export function structureOption(title: string, path: string, run: string | null): string {
  const { sample } = poseKey(path);
  return [title, run, sample !== null && sample > 0 ? `pose ${sample + 1}` : null].filter(Boolean).join(" · ");
}

/** PoseBusters checks in plain words, by the names the tool reports. */
const POSEBUSTERS: Record<string, string> = {
  mol_pred_loaded: "Ligand file reads",
  mol_true_loaded: "Reference ligand reads",
  mol_cond_loaded: "Protein file reads",
  sanitization: "Ligand chemistry is valid",
  inchi_convertible: "Ligand converts to InChI",
  all_atoms_connected: "All ligand atoms are connected",
  bond_lengths: "Bond lengths are realistic",
  bond_angles: "Bond angles are realistic",
  internal_steric_clash: "No clash within the ligand",
  aromatic_ring_flatness: "Aromatic rings are flat",
  non_aromatic_ring_non_flatness: "Saturated rings are puckered",
  double_bond_flatness: "Double bonds are flat",
  internal_energy: "Ligand strain is reasonable",
  protein_ligand_maximum_distance: "Ligand sits at the protein",
  minimum_distance_to_protein: "No clash with the protein",
  minimum_distance_to_organic_cofactors: "No clash with organic cofactors",
  minimum_distance_to_inorganic_cofactors: "No clash with ions or metals",
  minimum_distance_to_waters: "No clash with waters",
  volume_overlap_with_protein: "No overlap with the protein",
  volume_overlap_with_organic_cofactors: "No overlap with organic cofactors",
  volume_overlap_with_inorganic_cofactors: "No overlap with ions or metals",
  volume_overlap_with_waters: "No overlap with waters",
};

export function posebustersCheck(name: string): string {
  const key = name.trim().toLowerCase().replace(/[\s-]+/g, "_");
  if (POSEBUSTERS[key]) return POSEBUSTERS[key];
  const words = name.replace(/[_-]+/g, " ").trim();
  return words.charAt(0).toUpperCase() + words.slice(1);
}
