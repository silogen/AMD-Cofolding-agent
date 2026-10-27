// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * Superposition of two structures of the same protein: residues paired by
 * sequence, the best rigid fit on their CA atoms, and the RMSD of the two
 * ligands in that frame.
 */

export type Vec3 = [number, number, number];

/** The CA atom of one polymer residue. */
export interface CaAtom {
  resname: string;
  pos: Vec3;
}

/** One ligand molecule: its heavy atoms and the bonds between them (indices into atoms). */
export interface Ligand {
  atoms: { name: string; element: string; pos: Vec3 }[];
  bonds: [number, number][];
}

/** A rigid transform: rotation (row-major 3×3) then translation. */
export interface Transform {
  rotation: number[];
  translation: Vec3;
}

export interface Superposition {
  /** Moves the second structure onto the first. */
  transform: Transform;
  /** Over the core pairs kept by the fit. */
  caRmsd: number;
  core: number;
  aligned: number;
  /** First and last residue of the second structure that aligned, as indices into its CA list. */
  span: [number, number];
}

const ONE_LETTER: Record<string, string> = {
  ALA: "A", ARG: "R", ASN: "N", ASP: "D", CYS: "C", GLN: "Q", GLU: "E", GLY: "G", HIS: "H", ILE: "I",
  LEU: "L", LYS: "K", MET: "M", PHE: "F", PRO: "P", SER: "S", THR: "T", TRP: "W", TYR: "Y", VAL: "V",
  MSE: "M", SEC: "U", PYL: "O", HID: "H", HIE: "H", HIP: "H", CYX: "C",
};

export const oneLetter = (resname: string) => ONE_LETTER[resname.toUpperCase()] ?? "X";

/**
 * Local alignment (Smith-Waterman, linear gaps): the pairs [i, j] of
 * residues of *a* and *b* that align. Unknown residues match nothing.
 */
export function alignSequences(a: string, b: string, scores = { match: 4, mismatch: -1, gap: -4 }): [number, number][] {
  const n = a.length;
  const m = b.length;
  if (!n || !m) return [];
  const width = m + 1;
  const H = new Int32Array((n + 1) * width);
  // 0 stop, 1 diagonal, 2 up (gap in b), 3 left (gap in a)
  const from = new Uint8Array((n + 1) * width);
  let best = 0;
  let bestAt = 0;
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const same = a[i - 1] === b[j - 1] && a[i - 1] !== "X";
      const diag = H[(i - 1) * width + j - 1] + (same ? scores.match : scores.mismatch);
      const up = H[(i - 1) * width + j] + scores.gap;
      const left = H[i * width + j - 1] + scores.gap;
      let value = 0;
      let dir = 0;
      if (diag > value) [value, dir] = [diag, 1];
      if (up > value) [value, dir] = [up, 2];
      if (left > value) [value, dir] = [left, 3];
      H[i * width + j] = value;
      from[i * width + j] = dir;
      if (value > best) [best, bestAt] = [value, i * width + j];
    }
  }
  const pairs: [number, number][] = [];
  let i = Math.floor(bestAt / width);
  let j = bestAt % width;
  while (i > 0 && j > 0 && from[i * width + j] !== 0) {
    const dir = from[i * width + j];
    if (dir === 1) {
      pairs.push([i - 1, j - 1]);
      i--;
      j--;
    } else if (dir === 2) i--;
    else j--;
  }
  return pairs.reverse();
}

const centroid = (points: Vec3[]): Vec3 => {
  const c: Vec3 = [0, 0, 0];
  for (const p of points) for (let k = 0; k < 3; k++) c[k] += p[k] / points.length;
  return c;
};

export function apply(t: Transform, p: Vec3): Vec3 {
  const r = t.rotation;
  return [
    r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + t.translation[0],
    r[3] * p[0] + r[4] * p[1] + r[5] * p[2] + t.translation[1],
    r[6] * p[0] + r[7] * p[1] + r[8] * p[2] + t.translation[2],
  ];
}

const distance2 = (a: Vec3, b: Vec3) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2;

export function rmsd(a: Vec3[], b: Vec3[]): number {
  if (!a.length || a.length !== b.length) return Number.NaN;
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += distance2(a[i], b[i]);
  return Math.sqrt(sum / a.length);
}

/** Eigenvector of the largest eigenvalue of a symmetric 4×4 matrix, by Jacobi rotations. */
function topEigenvector(matrix: number[][]): number[] {
  const a = matrix.map((row) => [...row]);
  const v = [0, 1, 2, 3].map((i) => [0, 1, 2, 3].map((j): number => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 50; sweep++) {
    let off = 0;
    for (let p = 0; p < 4; p++) for (let q = p + 1; q < 4; q++) off += a[p][q] ** 2;
    if (off < 1e-20) break;
    for (let p = 0; p < 4; p++) {
      for (let q = p + 1; q < 4; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < 4; k++) {
          const akp = a[k][p];
          const akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < 4; k++) {
          const apk = a[p][k];
          const aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < 4; k++) {
          const vkp = v[k][p];
          const vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  let top = 0;
  for (let k = 1; k < 4; k++) if (a[k][k] > a[top][top]) top = k;
  return v.map((row) => row[top]);
}

/** The rotation and translation that best move *mobile* onto *target* (Horn's quaternion form of Kabsch). */
export function fit(mobile: Vec3[], target: Vec3[]): Transform {
  const cm = centroid(mobile);
  const ct = centroid(target);
  const S = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  for (let i = 0; i < mobile.length; i++) {
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) S[r * 3 + c] += (mobile[i][r] - cm[r]) * (target[i][c] - ct[c]);
  }
  const [xx, xy, xz, yx, yy, yz, zx, zy, zz] = S;
  const [w, x, y, z] = topEigenvector([
    [xx + yy + zz, yz - zy, zx - xz, xy - yx],
    [yz - zy, xx - yy - zz, xy + yx, zx + xz],
    [zx - xz, xy + yx, -xx + yy - zz, yz + zy],
    [xy - yx, zx + xz, yz + zy, -xx - yy + zz],
  ]);
  const rotation = [
    w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z,
  ];
  const moved = apply({ rotation, translation: [0, 0, 0] }, cm);
  return { rotation, translation: [ct[0] - moved[0], ct[1] - moved[1], ct[2] - moved[2]] };
}

/**
 * Superpose *b* on *a* by their CA atoms: pair residues by sequence, fit,
 * then refit a few times without pairs further than *cutoff* Å apart, so a
 * moved loop or a disordered tail does not pull the core. Null when fewer
 * than *minPairs* residues align.
 */
export function superposeCa(a: CaAtom[], b: CaAtom[], { cutoff = 2, cycles = 5, minPairs = 10 } = {}): Superposition | null {
  const seqA = a.map((atom) => oneLetter(atom.resname)).join("");
  const seqB = b.map((atom) => oneLetter(atom.resname)).join("");
  const pairs = alignSequences(seqA, seqB);
  if (pairs.length < Math.min(minPairs, a.length, b.length) || pairs.length < 3) return null;
  let core = pairs;
  let transform = fit(
    core.map(([, j]) => b[j].pos),
    core.map(([i]) => a[i].pos),
  );
  for (let cycle = 0; cycle < cycles; cycle++) {
    const kept = pairs.filter(([i, j]) => distance2(apply(transform, b[j].pos), a[i].pos) <= cutoff * cutoff);
    if (kept.length === core.length || kept.length < Math.max(3, pairs.length / 4)) break;
    core = kept;
    transform = fit(
      core.map(([, j]) => b[j].pos),
      core.map(([i]) => a[i].pos),
    );
  }
  const caRmsd = rmsd(
    core.map(([, j]) => apply(transform, b[j].pos)),
    core.map(([i]) => a[i].pos),
  );
  return { transform, caRmsd, core: core.length, aligned: pairs.length, span: [pairs[0][1], pairs[pairs.length - 1][1]] };
}

/** Column-major 4×4, as Mol* takes it. */
export function toMat4(t: Transform): number[] {
  const r = t.rotation;
  return [r[0], r[3], r[6], 0, r[1], r[4], r[7], 0, r[2], r[5], r[8], 0, t.translation[0], t.translation[1], t.translation[2], 1];
}

/** Minimum-cost assignment of rows to columns of a square cost matrix (Hungarian algorithm). */
export function assignment(cost: number[][]): number[] {
  const n = cost.length;
  const u = new Array<number>(n + 1).fill(0);
  const v = new Array<number>(n + 1).fill(0);
  const p = new Array<number>(n + 1).fill(0);
  const way = new Array<number>(n + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Array<number>(n + 1).fill(Infinity);
    const used = new Array<boolean>(n + 1).fill(false);
    do {
      used[j0] = true;
      const i0 = p[j0];
      let delta = Infinity;
      let j1 = 0;
      for (let j = 1; j <= n; j++) {
        if (used[j]) continue;
        const cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) [minv[j], way[j]] = [cur, j0];
        if (minv[j] < delta) [delta, j1] = [minv[j], j];
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else minv[j] -= delta;
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }
  const rowToCol = new Array<number>(n).fill(-1);
  for (let j = 1; j <= n; j++) if (p[j]) rowToCol[p[j] - 1] = j - 1;
  return rowToCol;
}

const elementOf = (atom: { element: string }) => atom.element.toUpperCase();
const composition = (ligand: Ligand) => ligand.atoms.map(elementOf).sort().join(",");

/**
 * Symmetry-corrected RMSD: the smallest over every mapping of *a*'s atoms
 * onto *b*'s that keeps elements and bonds (so a flipped ring or a rotated
 * methyl group counts as the same pose), by branch and bound. Null when no
 * such mapping exists or the search gives up after *budget* steps.
 */
function graphRmsd(a: Ligand, b: Ligand, budget: number): number | null {
  const n = a.atoms.length;
  const adjacency = (ligand: Ligand) => {
    const list = ligand.atoms.map(() => new Set<number>());
    for (const [i, j] of ligand.bonds) {
      list[i].add(j);
      list[j].add(i);
    }
    return list;
  };
  const adjA = adjacency(a);
  const adjB = adjacency(b);
  const signature = (adj: Set<number>[], ligand: Ligand, i: number) => `${elementOf(ligand.atoms[i])}${adj[i].size}`;
  const sigA = a.atoms.map((_, i) => signature(adjA, a, i));
  const sigB = b.atoms.map((_, i) => signature(adjB, b, i));
  if ([...sigA].sort().join() !== [...sigB].sort().join()) return null;

  // Visit a's atoms breadth-first from the rarest kind, so every atom after the first has a placed neighbour.
  const counts = new Map<string, number>();
  for (const s of sigA) counts.set(s, (counts.get(s) ?? 0) + 1);
  const order: number[] = [];
  const seen = new Set<number>();
  const starts = a.atoms.map((_, i) => i).sort((x, y) => counts.get(sigA[x])! - counts.get(sigA[y])!);
  for (const start of starts) {
    if (seen.has(start)) continue;
    const queue = [start];
    seen.add(start);
    while (queue.length) {
      const i = queue.shift()!;
      order.push(i);
      for (const j of adjA[i]) {
        if (!seen.has(j)) {
          seen.add(j);
          queue.push(j);
        }
      }
    }
  }

  const map = new Array<number>(n).fill(-1);
  const used = new Array<boolean>(n).fill(false);
  let best = Infinity;
  let steps = 0;
  const search = (depth: number, sum: number) => {
    if (sum >= best || steps > budget) return;
    if (depth === n) {
      best = sum;
      return;
    }
    const i = order[depth];
    for (let j = 0; j < n; j++) {
      if (used[j] || sigB[j] !== sigA[i]) continue;
      let consistent = true;
      for (let d = 0; d < depth && consistent; d++) {
        const k = order[d];
        if (adjA[i].has(k) !== adjB[j].has(map[k])) consistent = false;
      }
      if (!consistent) continue;
      steps++;
      map[i] = j;
      used[j] = true;
      search(depth + 1, sum + distance2(a.atoms[i].pos, b.atoms[j].pos));
      used[j] = false;
      map[i] = -1;
    }
  };
  search(0, 0);
  return Number.isFinite(best) && steps <= budget ? Math.sqrt(best / n) : null;
}

/** The closest pairing that keeps elements, ignoring bonds: a lower bound, for when the bonds disagree. */
function elementRmsd(a: Ligand, b: Ligand): number {
  let sum = 0;
  for (const element of new Set(a.atoms.map(elementOf))) {
    const xs = a.atoms.filter((atom) => elementOf(atom) === element);
    const ys = b.atoms.filter((atom) => elementOf(atom) === element);
    const cost = xs.map((x) => ys.map((y) => distance2(x.pos, y.pos)));
    assignment(cost).forEach((j, i) => (sum += cost[i][j]));
  }
  return Math.sqrt(sum / a.atoms.length);
}

/**
 * RMSD of two copies of the same ligand, in place: symmetry-corrected over
 * the bond graph, else by atom name when the names agree, else the closest
 * element-preserving pairing. Null when they are not the same molecule.
 */
export function ligandRmsd(a: Ligand, b: Ligand, budget = 200_000): number | null {
  if (!a.atoms.length || a.atoms.length !== b.atoms.length || composition(a) !== composition(b)) return null;
  if (a.bonds.length && b.bonds.length) {
    const value = graphRmsd(a, b, budget);
    if (value !== null) return value;
  }
  const names = new Map(b.atoms.map((atom) => [atom.name, atom]));
  const partners = a.atoms.map((atom) => names.get(atom.name));
  if (names.size === b.atoms.length && partners.every((atom, i) => atom && elementOf(atom) === elementOf(a.atoms[i]))) {
    return rmsd(
      a.atoms.map((atom) => atom.pos),
      partners.map((atom) => atom!.pos),
    );
  }
  return elementRmsd(a, b);
}

export function moveLigand(ligand: Ligand, transform: Transform): Ligand {
  return { ...ligand, atoms: ligand.atoms.map((atom) => ({ ...atom, pos: apply(transform, atom.pos) })) };
}

/**
 * Which of *b*'s ligands (already in *a*'s frame) answers *a*'s first one:
 * the closest copy of the same molecule, with its RMSD; failing that, the
 * one whose centre is nearest, with no RMSD. Null when either has none.
 */
export function matchLigand(a: Ligand[], b: Ligand[]): { index: number; rmsd: number | null } | null {
  const target = a[0];
  if (!target || !b.length) return null;
  let best: { index: number; rmsd: number | null } | null = null;
  b.forEach((ligand, index) => {
    const value = ligandRmsd(target, ligand);
    if (value !== null && (best?.rmsd == null || value < best.rmsd)) best = { index, rmsd: value };
  });
  if (best) return best;
  const centre = centroid(target.atoms.map((atom) => atom.pos));
  const distances = b.map((ligand) => distance2(centroid(ligand.atoms.map((atom) => atom.pos)), centre));
  return { index: distances.indexOf(Math.min(...distances)), rmsd: null };
}
