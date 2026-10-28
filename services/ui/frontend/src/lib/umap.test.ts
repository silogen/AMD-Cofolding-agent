// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { fuzzyGraph, initialLayout, knn, packBytes, popcount, random, tanimotoDistance, umap } from "./umap";

const WORDS = 2;

/** n fingerprints in two families that share most bits within a family. */
function families(n: number): Uint32Array {
  const fps = new Uint32Array(n * WORDS);
  const rand = random(7);
  for (let i = 0; i < n; i++) {
    const base = i % 2 ? 0xffff0000 : 0x0000ffff;
    fps[i * WORDS] = (base ^ (1 << Math.floor(rand() * 32))) >>> 0;
    fps[i * WORDS + 1] = (i % 2 ? 0xff00ff00 : 0x00ff00ff) >>> 0;
  }
  return fps;
}

describe("bits", () => {
  it("counts bits", () => {
    expect(popcount(0)).toBe(0);
    expect(popcount(0xffffffff)).toBe(32);
    expect(popcount(0b1011)).toBe(3);
  });
  it("packs RDKit's bytes little-endian", () => {
    expect(Array.from(packBytes(Uint8Array.from([1, 0, 0, 0, 0, 0, 0, 0x80]), 2))).toEqual([1, 0x80000000]);
  });
  it("measures Tanimoto distance", () => {
    const fps = Uint32Array.from([0b1100, 0b1010, 0b1100, 0]);
    expect(tanimotoDistance(fps, 1, 0, 1)).toBeCloseTo(1 - 1 / 3);
    expect(tanimotoDistance(fps, 1, 0, 2)).toBe(0);
    expect(tanimotoDistance(fps, 1, 3, 3)).toBe(0);
  });
});

describe("knn and graph", () => {
  it("finds neighbours in the same family, nearest first", () => {
    const n = 20;
    const { indices, distances } = knn(families(n), n, WORDS, 4);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < 4; j++) expect(indices[i * 4 + j] % 2).toBe(i % 2);
      for (let j = 1; j < 4; j++) expect(distances[i * 4 + j]).toBeGreaterThanOrEqual(distances[i * 4 + j - 1]);
    }
  });
  it("builds a symmetric fuzzy graph with weights in (0, 1]", () => {
    const n = 12;
    const { indices, distances } = knn(families(n), n, WORDS, 3);
    const g = fuzzyGraph(indices, distances, n, 3);
    const edges = new Map<string, number>();
    for (let e = 0; e < g.head.length; e++) edges.set(`${g.head[e]}-${g.tail[e]}`, g.weight[e]);
    for (const [key, w] of edges) {
      const [i, j] = key.split("-");
      expect(edges.get(`${j}-${i}`)).toBeCloseTo(w);
      expect(w).toBeGreaterThan(0);
      expect(w).toBeLessThanOrEqual(1);
    }
  });
});

describe("umap", () => {
  it("separates two families", () => {
    const n = 40;
    const coords = umap({ fps: families(n), n, words: WORDS, neighbours: 5, epochs: 200 });
    const centre = (parity: number) => {
      let x = 0;
      let y = 0;
      for (let i = parity; i < n; i += 2) {
        x += coords[2 * i];
        y += coords[2 * i + 1];
      }
      return [x / (n / 2), y / (n / 2)];
    };
    const [a, b] = [centre(0), centre(1)];
    const between = Math.hypot(a[0] - b[0], a[1] - b[1]);
    let within = 0;
    for (let i = 0; i < n; i++) {
      const c = i % 2 ? b : a;
      within += Math.hypot(coords[2 * i] - c[0], coords[2 * i + 1] - c[1]);
    }
    expect(between).toBeGreaterThan((within / n) * 2);
    expect(coords.every(Number.isFinite)).toBe(true);
  });
  it("is deterministic for a seed and handles tiny inputs", () => {
    const n = 10;
    const fps = families(n);
    expect(Array.from(umap({ fps, n, words: WORDS, seed: 3 }))).toEqual(Array.from(umap({ fps, n, words: WORDS, seed: 3 })));
    expect(umap({ fps: new Uint32Array(0), n: 0, words: WORDS })).toHaveLength(0);
    expect(umap({ fps: fps.slice(0, 4), n: 2, words: WORDS })).toHaveLength(4);
  });
  it("starts from a previous layout, new points beside their neighbours", () => {
    const n = 6;
    const indices = Int32Array.from([1, 2, 0, 2, 0, 1, 4, 5, 3, 5, 3, 4, 0, 1, 2, 3, 4, 5].slice(0, n * 2));
    const init = Float32Array.from([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, NaN, NaN]);
    // Point 5's neighbours (indices 10, 11 → 3 and 4) are placed.
    const { coords, warm } = initialLayout(n, indices, 2, init, random(1));
    expect(warm).toBe(true);
    expect(coords[0]).toBe(0);
    expect(coords[10]).toBeCloseTo(3.5, 0);
    const cold = initialLayout(n, indices, 2, null, random(1));
    expect(cold.warm).toBe(false);
  });
});
