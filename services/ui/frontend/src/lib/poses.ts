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
export function structureKind(path: string, predictedMetrics = false): StructureKind | null {
  const p = clean(path).toLowerCase();
  const name = p.slice(p.lastIndexOf("/") + 1);
  if (/(^|\/)(prep|prepared|structure_prep|prepped)[^/]*\/|_(prep|prepared|fixed|protonated|clean(ed)?)\.(pdb|cif|mmcif)$/.test(p)) return "prepared";
  if (/(^|\/)(predictions|boltz_results[^/]*|cofold[^/]*)\//.test(p) || predictedMetrics) return "predicted";
  if (/(^|\/)(rcsb|pdb|pdb_files)\//.test(p) || /^(pdb)?[0-9][a-z0-9]{3}\.(pdb|cif|mmcif|ent)(\.gz)?$/.test(name)) return "experimental";
  return null;
}
