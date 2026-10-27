// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import {
  alignSequences,
  apply,
  assignment,
  fit,
  ligandRmsd,
  matchLigand,
  moveLigand,
  rmsd,
  superposeCa,
  toMat4,
  type CaAtom,
  type Ligand,
  type Transform,
  type Vec3,
} from "./superpose";

const rotationZ = (angle: number) => [Math.cos(angle), -Math.sin(angle), 0, Math.sin(angle), Math.cos(angle), 0, 0, 0, 1];
const rotationX = (angle: number) => [1, 0, 0, 0, Math.cos(angle), -Math.sin(angle), 0, Math.sin(angle), Math.cos(angle)];
const multiply = (a: number[], b: number[]) =>
  [0, 1, 2].flatMap((r) => [0, 1, 2].map((c) => a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c]));
const moved: Transform = { rotation: multiply(rotationZ(0.7), rotationX(-1.1)), translation: [12, -4, 30] };

/** A helix-like trace, so the points are not coplanar. */
const trace = (n: number): Vec3[] => Array.from({ length: n }, (_, i) => [2.3 * Math.cos(i * 1.7), 2.3 * Math.sin(i * 1.7), 1.5 * i]);
const SEQUENCE = "MSNPKKPGRVTNQLQYLHKVVMKALWKHQFAWPFRQPVDAVKLNLPDYYKIIKTPMDMGTIKKRLE";

describe("fit", () => {
  it("recovers a rigid motion", () => {
    const target = trace(20);
    const mobile = target.map((p) => apply(moved, p));
    const t = fit(mobile, target);
    expect(rmsd(mobile.map((p) => apply(t, p)), target)).toBeLessThan(1e-6);
  });

  it("never returns a reflection", () => {
    const target = trace(12);
    const mirrored = target.map(([x, y, z]): Vec3 => [-x, y, z]);
    const r = fit(mirrored, target).rotation;
    const det = r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6]) + r[2] * (r[3] * r[7] - r[4] * r[6]);
    expect(det).toBeCloseTo(1, 6);
  });
});

describe("alignSequences", () => {
  it("pairs a construct with the same residues in a longer chain", () => {
    const pairs = alignSequences(SEQUENCE.slice(10, 50), `AAAAGGGG${SEQUENCE}WWWW`);
    expect(pairs).toHaveLength(40);
    expect(pairs[0]).toEqual([0, 18]);
    expect(pairs.every(([i, j]) => j - i === 18)).toBe(true);
  });

  it("finds nothing in unrelated or unknown sequences", () => {
    expect(alignSequences("XXXX", "XXXX")).toEqual([]);
    expect(alignSequences("", "ACD")).toEqual([]);
  });
});

describe("superposeCa", () => {
  const atoms = (positions: Vec3[], sequence = SEQUENCE): CaAtom[] => positions.map((pos, i) => ({ resname: THREE[sequence[i]], pos }));
  const THREE: Record<string, string> = {
    A: "ALA", R: "ARG", N: "ASN", D: "ASP", C: "CYS", Q: "GLN", E: "GLU", G: "GLY", H: "HIS", I: "ILE",
    L: "LEU", K: "LYS", M: "MET", F: "PHE", P: "PRO", S: "SER", T: "THR", W: "TRP", Y: "TYR", V: "VAL",
  };

  it("superposes a renumbered, moved copy and ignores a moved loop", () => {
    const n = SEQUENCE.length;
    const a = atoms(trace(n));
    const b = atoms(
      trace(n).map((p, i) => apply(moved, i >= 30 && i < 36 ? ([p[0] + 6, p[1], p[2]] as Vec3) : p)),
    ).slice(5);
    const result = superposeCa(a, b);
    expect(result).not.toBeNull();
    expect(result!.aligned).toBe(n - 5);
    expect(result!.span).toEqual([0, n - 6]);
    expect(result!.core).toBe(n - 5 - 6);
    expect(result!.caRmsd).toBeLessThan(1e-6);
    expect(rmsd([apply(result!.transform, b[0].pos)], [a[5].pos])).toBeLessThan(1e-6);
  });

  it("gives up on unrelated proteins", () => {
    expect(superposeCa(atoms(trace(20), "W".repeat(20)), atoms(trace(20), "G".repeat(20)))).toBeNull();
  });
});

describe("toMat4", () => {
  it("is column-major with the translation last", () => {
    const m = toMat4({ rotation: [1, 2, 3, 4, 5, 6, 7, 8, 9], translation: [10, 11, 12] });
    expect(m).toEqual([1, 4, 7, 0, 2, 5, 8, 0, 3, 6, 9, 0, 10, 11, 12, 1]);
  });
});

describe("assignment", () => {
  it("finds the cheapest pairing", () => {
    expect(
      assignment([
        [4, 1, 3],
        [2, 0, 5],
        [3, 2, 2],
      ]),
    ).toEqual([1, 0, 2]);
  });
});

/** Toluene-like: a ring of six carbons, a methyl on C1 and a chlorine para to it. */
function ring(offset: Vec3 = [0, 0, 0], flip = false): Ligand {
  const ringAtoms = Array.from({ length: 6 }, (_, i): Vec3 => {
    const angle = (i * Math.PI) / 3;
    return [1.4 * Math.cos(angle), (flip ? -1 : 1) * 1.4 * Math.sin(angle), i % 2 ? 0.3 : 0];
  });
  const atoms = [
    ...ringAtoms.map((pos, i) => ({ name: `C${i + 1}`, element: "C", pos })),
    { name: "C7", element: "C", pos: [2.9, 0, 0] as Vec3 },
    { name: "CL1", element: "Cl", pos: [-3.1, 0, 0] as Vec3 },
  ].map((atom) => ({ ...atom, pos: [atom.pos[0] + offset[0], atom.pos[1] + offset[1], atom.pos[2] + offset[2]] as Vec3 }));
  const bonds: [number, number][] = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 6], [3, 7]];
  return { atoms, bonds };
}

describe("ligandRmsd", () => {
  it("is zero for the same pose, whatever the atom names and order", () => {
    const a = ring();
    const order = [7, 3, 1, 6, 0, 5, 2, 4];
    const b: Ligand = {
      atoms: order.map((i, k) => ({ ...a.atoms[i], name: `X${k}` })),
      bonds: a.bonds.map(([i, j]) => [order.indexOf(i), order.indexOf(j)]),
    };
    expect(ligandRmsd(a, b)).toBeCloseTo(0, 6);
  });

  it("treats a flipped ring as the same pose", () => {
    expect(ligandRmsd(ring(), ring([0, 0, 0], true))).toBeCloseTo(0, 6);
  });

  it("measures a shift", () => {
    expect(ligandRmsd(ring(), ring([1, 0, 0]))).toBeCloseTo(1, 6);
  });

  it("is null for different molecules", () => {
    const other = ring();
    other.atoms[7] = { ...other.atoms[7], element: "Br" };
    expect(ligandRmsd(ring(), other)).toBeNull();
  });

  it("falls back to names when there are no bonds", () => {
    const a = { ...ring(), bonds: [] };
    expect(ligandRmsd(a, { ...ring([0, 2, 0]), bonds: [] })).toBeCloseTo(2, 6);
  });
});

describe("matchLigand", () => {
  const glycerol: Ligand = {
    atoms: [
      { name: "C1", element: "C", pos: [0, 0, 0] },
      { name: "O1", element: "O", pos: [1.4, 0, 0] },
    ],
    bonds: [[0, 1]],
  };

  it("takes the closest copy of the ligand, after the superposition", () => {
    const back: Transform = fit(
      ring().atoms.map((atom) => apply(moved, atom.pos)),
      ring().atoms.map((atom) => atom.pos),
    );
    const far = moveLigand(ring([8, 0, 0]), moved);
    const near = moveLigand(ring([0.5, 0, 0]), moved);
    const match = matchLigand([ring()], [glycerol, far, near].map((ligand) => moveLigand(ligand, back)));
    expect(match?.index).toBe(2);
    expect(match?.rmsd).toBeCloseTo(0.5, 6);
  });

  it("falls back to the nearest other molecule, without an RMSD", () => {
    const other = ring([1, 0, 0]);
    other.atoms[7] = { ...other.atoms[7], element: "Br" };
    expect(matchLigand([ring()], [{ ...glycerol, atoms: glycerol.atoms.map((atom) => ({ ...atom, pos: [20, 0, 0] as Vec3 })) }, other])).toEqual({ index: 1, rmsd: null });
  });

  it("is null without a ligand on either side", () => {
    expect(matchLigand([], [ring()])).toBeNull();
    expect(matchLigand([ring()], [])).toBeNull();
  });
});
