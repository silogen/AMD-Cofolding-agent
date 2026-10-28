// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * A small UMAP (McInnes et al. 2018) for a few thousand fingerprints, on typed
 * arrays: exact k nearest neighbours by Tanimoto (Jaccard) distance on
 * bit-packed fingerprints, the fuzzy simplicial set, and the SGD layout of
 * umap-learn with negative sampling. Runs in a worker (workers/umap.worker.ts).
 *
 * A previous layout can seed the next one (`init`, NaN for new points): new
 * points start next to their nearest placed neighbours and the layout is only
 * refined, so a map that grows with a run keeps its shape.
 */

export interface UmapInput {
  /** n × words packed bits. */
  fps: Uint32Array;
  n: number;
  words: number;
  neighbours?: number;
  /** n × 2, NaN where unknown. */
  init?: Float32Array | null;
  epochs?: number;
  seed?: number;
}

/** a and b of the low-dimensional kernel 1 / (1 + a d^2b) for min_dist 0.1, spread 1. */
const A = 1.5769434603113077;
const B = 0.8950608779109733;
const NEGATIVE_RATE = 5;

export function popcount(v: number): number {
  v -= (v >>> 1) & 0x55555555;
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
  return (((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

/** Packs RDKit's fingerprint bytes into 32-bit words. */
export function packBytes(bytes: Uint8Array, words: number, out: Uint32Array = new Uint32Array(words), offset = 0): Uint32Array {
  for (let w = 0; w < words; w++) {
    const b = w * 4;
    out[offset + w] = ((bytes[b] ?? 0) | ((bytes[b + 1] ?? 0) << 8) | ((bytes[b + 2] ?? 0) << 16) | ((bytes[b + 3] ?? 0) << 24)) >>> 0;
  }
  return out;
}

export function tanimotoDistance(fps: Uint32Array, words: number, i: number, j: number, counts?: Uint16Array): number {
  let inter = 0;
  let union = 0;
  const a = i * words;
  const b = j * words;
  for (let w = 0; w < words; w++) {
    inter += popcount(fps[a + w] & fps[b + w]);
    if (!counts) union += popcount(fps[a + w] | fps[b + w]);
  }
  if (counts) union = counts[i] + counts[j] - inter;
  return union === 0 ? 0 : 1 - inter / union;
}

/** Exact k nearest neighbours of every point (itself excluded), nearest first. */
export function knn(fps: Uint32Array, n: number, words: number, k: number): { indices: Int32Array; distances: Float32Array } {
  k = Math.max(1, Math.min(k, n - 1));
  const indices = new Int32Array(n * k).fill(-1);
  const distances = new Float32Array(n * k).fill(Infinity);
  const counts = new Uint16Array(n);
  for (let i = 0; i < n; i++) {
    let c = 0;
    for (let w = 0; w < words; w++) c += popcount(fps[i * words + w]);
    counts[i] = c;
  }
  const insert = (i: number, j: number, d: number) => {
    const base = i * k;
    if (d >= distances[base + k - 1]) return;
    let p = k - 1;
    while (p > 0 && distances[base + p - 1] > d) {
      distances[base + p] = distances[base + p - 1];
      indices[base + p] = indices[base + p - 1];
      p--;
    }
    distances[base + p] = d;
    indices[base + p] = j;
  };
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = tanimotoDistance(fps, words, i, j, counts);
      insert(i, j, d);
      insert(j, i, d);
    }
  }
  return { indices, distances };
}

/**
 * The fuzzy simplicial set: per point the distance to its nearest neighbour
 * (rho) and a bandwidth (sigma) so memberships sum to log2(k + 1), made
 * symmetric with the probabilistic t-conorm a + b - ab. As edge lists.
 */
export function fuzzyGraph(indices: Int32Array, distances: Float32Array, n: number, k: number): { head: Int32Array; tail: Int32Array; weight: Float32Array } {
  const target = Math.log2(k + 1);
  let meanAll = 0;
  for (let i = 0; i < distances.length; i++) meanAll += Number.isFinite(distances[i]) ? distances[i] : 0;
  meanAll /= distances.length || 1;
  const directed = new Map<number, number>();
  for (let i = 0; i < n; i++) {
    const base = i * k;
    let rho = 0;
    let sumD = 0;
    for (let j = 0; j < k; j++) {
      const d = distances[base + j];
      sumD += d;
      if (rho === 0 && d > 0) rho = d;
    }
    let lo = 0;
    let hi = Infinity;
    let sigma = 1;
    for (let iter = 0; iter < 64; iter++) {
      let psum = 0;
      for (let j = 0; j < k; j++) {
        const d = distances[base + j] - rho;
        psum += d > 0 ? Math.exp(-d / sigma) : 1;
      }
      if (Math.abs(psum - target) < 1e-5) break;
      if (psum > target) {
        hi = sigma;
        sigma = (lo + hi) / 2;
      } else {
        lo = sigma;
        sigma = hi === Infinity ? sigma * 2 : (lo + hi) / 2;
      }
    }
    const floor = 1e-3 * (rho > 0 ? sumD / k : meanAll);
    if (sigma < floor) sigma = floor;
    for (let j = 0; j < k; j++) {
      const other = indices[base + j];
      if (other < 0) continue;
      const d = distances[base + j] - rho;
      const w = d <= 0 || sigma === 0 ? 1 : Math.exp(-d / sigma);
      directed.set(i * n + other, w);
    }
  }
  const head: number[] = [];
  const tail: number[] = [];
  const weight: number[] = [];
  for (const [key, w] of directed) {
    const i = Math.floor(key / n);
    const j = key - i * n;
    const back = directed.get(j * n + i) ?? 0;
    // Each undirected edge is kept in both directions, as umap-learn's symmetric matrix does.
    const sym = w + back - w * back;
    head.push(i);
    tail.push(j);
    weight.push(sym);
    if (!back) {
      head.push(j);
      tail.push(i);
      weight.push(sym);
    }
  }
  return { head: Int32Array.from(head), tail: Int32Array.from(tail), weight: Float32Array.from(weight) };
}

export function random(seed: number): () => number {
  let s = seed >>> 0 || 1;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Starting positions: the previous layout where known, new points beside their placed neighbours, the rest at random. */
export function initialLayout(n: number, indices: Int32Array, k: number, init: Float32Array | null | undefined, rand: () => number): { coords: Float32Array; warm: boolean } {
  const coords = new Float32Array(n * 2);
  const placed = new Uint8Array(n);
  let known = 0;
  if (init && init.length === n * 2) {
    for (let i = 0; i < n; i++) {
      if (Number.isFinite(init[2 * i]) && Number.isFinite(init[2 * i + 1])) {
        coords[2 * i] = init[2 * i];
        coords[2 * i + 1] = init[2 * i + 1];
        placed[i] = 1;
        known++;
      }
    }
  }
  const warm = known >= Math.max(4, n * 0.3);
  if (!warm) {
    for (let i = 0; i < 2 * n; i++) coords[i] = rand() * 20 - 10;
    return { coords, warm };
  }
  let x0 = Infinity;
  let x1 = -Infinity;
  let y0 = Infinity;
  let y1 = -Infinity;
  for (let i = 0; i < n; i++) {
    if (!placed[i]) continue;
    x0 = Math.min(x0, coords[2 * i]);
    x1 = Math.max(x1, coords[2 * i]);
    y0 = Math.min(y0, coords[2 * i + 1]);
    y1 = Math.max(y1, coords[2 * i + 1]);
  }
  const jitter = Math.max(x1 - x0, y1 - y0, 1) * 0.01;
  for (let i = 0; i < n; i++) {
    if (placed[i]) continue;
    let sx = 0;
    let sy = 0;
    let c = 0;
    for (let j = 0; j < k; j++) {
      const other = indices[i * k + j];
      if (other >= 0 && placed[other]) {
        sx += coords[2 * other];
        sy += coords[2 * other + 1];
        c++;
      }
    }
    coords[2 * i] = c ? sx / c + (rand() - 0.5) * jitter : x0 + rand() * (x1 - x0);
    coords[2 * i + 1] = c ? sy / c + (rand() - 0.5) * jitter : y0 + rand() * (y1 - y0);
  }
  return { coords, warm };
}

const clip = (v: number) => (v > 4 ? 4 : v < -4 ? -4 : v);

/** umap-learn's optimize_layout_euclidean, in place on *coords*. */
export function optimizeLayout(coords: Float32Array, n: number, graph: { head: Int32Array; tail: Int32Array; weight: Float32Array }, epochs: number, rand: () => number, alpha0 = 1): void {
  let wMax = 0;
  for (const w of graph.weight) if (w > wMax) wMax = w;
  if (wMax <= 0) return;
  const kept: number[] = [];
  for (let e = 0; e < graph.weight.length; e++) if (graph.weight[e] >= wMax / epochs) kept.push(e);
  const m = kept.length;
  const head = new Int32Array(m);
  const tail = new Int32Array(m);
  const perSample = new Float32Array(m);
  for (let e = 0; e < m; e++) {
    head[e] = graph.head[kept[e]];
    tail[e] = graph.tail[kept[e]];
    perSample[e] = wMax / graph.weight[kept[e]];
  }
  const perNegative = perSample.map((v) => v / NEGATIVE_RATE);
  const nextSample = Float32Array.from(perSample);
  const nextNegative = Float32Array.from(perNegative);
  for (let epoch = 0; epoch < epochs; epoch++) {
    const alpha = alpha0 * (1 - epoch / epochs);
    for (let e = 0; e < m; e++) {
      if (nextSample[e] > epoch) continue;
      const i = head[e];
      const j = tail[e];
      let dx = coords[2 * i] - coords[2 * j];
      let dy = coords[2 * i + 1] - coords[2 * j + 1];
      let d2 = dx * dx + dy * dy;
      if (d2 > 0) {
        const coeff = (-2 * A * B * d2 ** (B - 1)) / (A * d2 ** B + 1);
        const gx = clip(coeff * dx) * alpha;
        const gy = clip(coeff * dy) * alpha;
        coords[2 * i] += gx;
        coords[2 * i + 1] += gy;
        coords[2 * j] -= gx;
        coords[2 * j + 1] -= gy;
      }
      nextSample[e] += perSample[e];
      const negatives = Math.floor((epoch - nextNegative[e]) / perNegative[e]);
      for (let p = 0; p < negatives; p++) {
        const other = Math.floor(rand() * n);
        if (other === i) continue;
        dx = coords[2 * i] - coords[2 * other];
        dy = coords[2 * i + 1] - coords[2 * other + 1];
        d2 = dx * dx + dy * dy;
        if (d2 <= 0) continue;
        const coeff = (2 * B) / ((0.001 + d2) * (A * d2 ** B + 1));
        coords[2 * i] += clip(coeff * dx) * alpha;
        coords[2 * i + 1] += clip(coeff * dy) * alpha;
      }
      nextNegative[e] += negatives * perNegative[e];
    }
  }
}

/** The 2D embedding, n × 2. */
export function umap({ fps, n, words, neighbours = 14, init, epochs, seed = 42 }: UmapInput): Float32Array {
  if (n === 0) return new Float32Array(0);
  if (n < 3) return Float32Array.from({ length: n * 2 }, (_, i) => (i % 2 ? 0 : i / 2));
  const k = Math.max(1, Math.min(neighbours, n - 1));
  const rand = random(seed);
  const { indices, distances } = knn(fps, n, words, k);
  const graph = fuzzyGraph(indices, distances, n, k);
  const { coords, warm } = initialLayout(n, indices, k, init, rand);
  optimizeLayout(coords, n, graph, epochs ?? (warm ? 120 : n > 2000 ? 250 : 350), rand, warm ? 0.4 : 1);
  return coords;
}
