// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import {
  autoScale,
  axisTicks,
  bandwidth,
  colourDomain,
  colourScale,
  describeEndpoint,
  formatValue,
  goalOf,
  heading,
  jitter,
  niceTicks,
  countMask,
  inPolygon,
  kde,
  moleculesOf,
  pointDensity,
  pointsInside,
  quartiles,
  sampleForMap,
  selectionMask,
  shouldRecompute,
  smooth,
  sortedIndices,
  spreads,
  stepStats,
  topByWeight,
  TOTAL,
  viridis,
  viridisCss,
} from "./runAnalysis";
import { parseRunResults } from "./runResults";

const CSV = [
  "Agent,Prior,Score,SMILES,SMILES_state,QED,QED (raw),TPSA,TPSA (raw),step",
  "1,1,0.5,CCO,1,0.6,0.6,0.4,40,1",
  "1,1,0.7,CCN,1,0.8,0.8,0.5,50,1",
  "1,1,0,C1,0,,,,,1",
  "1,1,0.9,CCO,1,0.9,0.9,0.7,70,2",
  "1,1,0,CCN,2,0,0,0,0,2",
  "1,1,0.3,c1ccccc1,1,0.2,0.2,nan,,2",
].join("\n");

const table = parseRunResults(CSV);

describe("moleculesOf", () => {
  it("keeps one row per SMILES, its best, without invalid or duplicate rows", () => {
    const m = moleculesOf(table);
    expect(m.n).toBe(3);
    expect(m.smiles).toEqual(["CCO", "CCN", "c1ccccc1"]);
    expect(Array.from(m.values.get(TOTAL)!)).toEqual([0.9, 0.7, 0.3]);
    expect(Array.from(m.step)).toEqual([2, 1, 2]);
    expect(m.lastStep).toBe(2);
    expect(m.endpoints.map((e) => `${e.kind}:${e.key}:${e.source}`)).toEqual(["total:Score:total", "component:QED:raw", "component:TPSA:raw"]);
    expect(Array.from(m.values.get("TPSA")!.slice(0, 2))).toEqual([70, 50]);
    expect(Array.from(m.scores.get("TPSA")!.slice(0, 2))).toEqual([0.7, 0.5]);
    expect(Number.isNaN(m.values.get("TPSA")![2])).toBe(true);
    expect(m.endpoints.map(heading)).toEqual(["Total", "QED", "TPSA (Å²)"]);
  });

  it("shows Boltz affinity as pIC50 and falls back to the score without a raw value", () => {
    const csv = ["Score,SMILES,SMILES_state,Boltz affinity,Boltz affinity (raw),Pose quality,step", "0.5,CCO,1,0.6,0.3,0.8,1", "0.4,CCN,1,0.5,-1.5,0.7,1"].join("\n");
    const m = moleculesOf(parseRunResults(csv));
    const affinity = m.endpoints.find((e) => e.key === "Boltz affinity")!;
    expect(heading(affinity)).toBe("Affinity (pIC50)");
    expect(Array.from(m.values.get("Boltz affinity")!)).toEqual([5.7, 7.5]);
    const pose = m.endpoints.find((e) => e.key === "Pose quality")!;
    expect(pose.source).toBe("score");
    expect(pose.unit).toBe("score");
    expect(Array.from(m.values.get("Pose quality")!)).toEqual([0.8, 0.7]);
  });

  it("copes with an empty file", () => {
    const m = moleculesOf(parseRunResults(""));
    expect(m.n).toBe(0);
    expect(m.lastStep).toBe(0);
  });
});

describe("stepStats", () => {
  it("has the mean, min and max per step over valid rows", () => {
    const m = moleculesOf(table);
    const stats = stepStats(table, m.endpoints);
    expect(stats.get(TOTAL)).toEqual([
      { step: 1, mean: 0.6, min: 0.5, max: 0.7 },
      { step: 2, mean: 0.6, min: 0.3, max: 0.9 },
    ]);
    expect(stats.get("TPSA")?.map((p) => p.mean)).toEqual([0.45, 0.7]);
  });
});

describe("smooth", () => {
  it("averages a centred window and skips NaN", () => {
    expect(smooth([1, 2, 3, 4, 5], 3)).toEqual([1.5, 2, 3, 4, 4.5]);
    expect(smooth([1, NaN, 3], 3)).toEqual([1, 2, 3]);
    expect(smooth([1, 2], 1)).toEqual([1, 2]);
  });
});

describe("topByWeight", () => {
  const components = ["QED", "Boltz affinity", "TPSA", "cLogP", "Pose quality"];
  it("takes the highest weights, ties in TTP order", () => {
    const weights = [
      { name: "qed", weight: 1 },
      { name: "Boltz affinity", weight: 3 },
      { name: "TPSA", weight: 1 },
      { name: "cLogP", weight: 2 },
      { name: "Pose quality", weight: 0 },
    ];
    expect(topByWeight(components, weights)).toEqual(["Boltz affinity", "cLogP", "QED"]);
  });
  it("leaves out zero weights and components the TTP does not name", () => {
    expect(topByWeight(components, [{ name: "TPSA", weight: 1 }, { name: "Pose quality", weight: 0 }])).toEqual(["TPSA"]);
  });
  it("breaks ties by how much a component varies, then by column order", () => {
    const equal = components.map((name) => ({ name, weight: name === "Pose quality" ? 0.6 : 1 }));
    const spread = new Map([
      ["QED", 0.2],
      ["Boltz affinity", 0.3],
      ["TPSA", 0.01],
      ["cLogP", 0.25],
      ["Pose quality", 0.9],
    ]);
    expect(topByWeight(components, equal, 3, spread)).toEqual(["Boltz affinity", "cLogP", "QED"]);
    expect(topByWeight(components, equal)).toEqual(["QED", "Boltz affinity", "TPSA"]);
    expect(spreads(new Map([["a", Float64Array.from([1, 3, NaN])]])).get("a")).toBeCloseTo(Math.SQRT2);
  });
  it("falls back to the first components without weights", () => {
    expect(topByWeight(components, [])).toEqual(["QED", "Boltz affinity", "TPSA"]);
    expect(topByWeight(components, [{ name: "unrelated", weight: 1 }])).toEqual(["QED", "Boltz affinity", "TPSA"]);
  });
});

describe("violin KDE", () => {
  it("peaks where the values are and is normalised to 1", () => {
    const values = [...Array(100).fill(0.2), ...Array(10).fill(0.8)];
    const grid = kde(values, 0, 1, 0.05, 11);
    expect(Math.max(...grid)).toBeCloseTo(1);
    expect(grid[2]).toBeGreaterThan(grid[8]);
    expect(grid[8]).toBeGreaterThan(grid[5]);
  });
  it("draws a subset on the full set's scale", () => {
    const all = [0.1, 0.2, 0.3, 0.4, 0.5];
    const h = bandwidth(all, 0.1, 0.5);
    const peak = Math.max(...kde(all, 0.1, 0.5, h, 20, 1));
    const sub = kde([0.1, 0.2], 0.1, 0.5, h, 20, peak);
    expect(Math.max(...sub)).toBeLessThan(1);
    expect(Math.max(...sub)).toBeGreaterThan(0);
  });
  it("handles constant and empty input", () => {
    expect(Array.from(kde([0.5, 0.5], 0.5, 0.5, 0.1, 4))).toEqual([1, 1, 1, 1]);
    expect(Array.from(kde([], 0, 1, 0.1, 3))).toEqual([0, 0, 0]);
    expect(bandwidth([1], 0, 1)).toBeGreaterThan(0);
  });
  it("finds quartiles", () => {
    expect(quartiles([1, 2, 3, 4, 5])).toEqual({ q1: 2, median: 3, q3: 4 });
    expect(quartiles([])).toBeNull();
  });
});

describe("selectionMask", () => {
  const m = moleculesOf(table);
  it("is null when nothing is selected", () => {
    expect(selectionMask(m, {}, null)).toBeNull();
  });
  it("combines endpoint ranges with AND and excludes missing values", () => {
    const mask = selectionMask(m, { [TOTAL]: [0.5, 1], QED: [0.85, 1] }, null)!;
    expect(Array.from(mask)).toEqual([1, 0, 0]);
    expect(Array.from(selectionMask(m, { TPSA: [40, 60] }, null)!)).toEqual([0, 1, 0]);
  });
  it("intersects with a region of the map", () => {
    const mask = selectionMask(m, { [TOTAL]: [0, 1] }, new Set(["CCN", "c1ccccc1"]))!;
    expect(Array.from(mask)).toEqual([0, 1, 1]);
    expect(countMask(mask)).toBe(2);
  });
  it("ignores ranges on endpoints the run does not have", () => {
    expect(selectionMask(m, { Gone: [0, 1] }, null)).toBeNull();
  });
  it("sorts the kept molecules by an endpoint, missing values last", () => {
    expect(sortedIndices(m, null, TOTAL)).toEqual([0, 1, 2]);
    expect(sortedIndices(m, null, "TPSA", false)).toEqual([1, 0, 2]);
    expect(sortedIndices(m, Uint8Array.from([0, 1, 1]), "step")).toEqual([2, 1]);
  });
});

describe("lasso and box", () => {
  const square = [0, 0, 10, 0, 10, 10, 0, 10];
  it("tells points inside a polygon", () => {
    expect(inPolygon(5, 5, square)).toBe(true);
    expect(inPolygon(15, 5, square)).toBe(false);
    // A concave L shape: the notch is outside.
    const l = [0, 0, 10, 0, 10, 4, 4, 4, 4, 10, 0, 10];
    expect(inPolygon(2, 8, l)).toBe(true);
    expect(inPolygon(8, 8, l)).toBe(false);
  });
  it("picks the points in a shape", () => {
    const xs = [1, 5, 12];
    const ys = [1, 5, 5];
    expect(pointsInside(xs, ys, { kind: "lasso", polygon: square })).toEqual([0, 1]);
    expect(pointsInside(xs, ys, { kind: "box", x0: 6, y0: 0, x1: 0, y1: 6 })).toEqual([0, 1]);
  });
});

describe("shouldRecompute", () => {
  const base = { busy: false, computedStep: 10, latestStep: 10, lastMs: 500, points: 500, computedPoints: 400 };
  it("computes a first map", () => {
    expect(shouldRecompute({ ...base, computedStep: null, computedPoints: null })).toBe(true);
  });
  it("waits for two more steps, and for the last one to finish", () => {
    expect(shouldRecompute({ ...base, latestStep: 11 })).toBe(false);
    expect(shouldRecompute({ ...base, latestStep: 12 })).toBe(true);
    expect(shouldRecompute({ ...base, latestStep: 12, busy: true })).toBe(false);
  });
  it("waits longer after a slow map", () => {
    expect(shouldRecompute({ ...base, latestStep: 12, lastMs: 20_000 })).toBe(false);
    expect(shouldRecompute({ ...base, latestStep: 20, lastMs: 20_000 })).toBe(true);
  });
  it("skips when nothing changed or there are too few molecules", () => {
    expect(shouldRecompute({ ...base, latestStep: 14, computedPoints: 500 })).toBe(false);
    expect(shouldRecompute({ ...base, computedStep: null, computedPoints: null, points: 3 })).toBe(false);
  });
});

describe("sampleForMap", () => {
  it("keeps everything under the cap, else the best and a stable spread", () => {
    const n = 50;
    const smiles = Array.from({ length: n }, (_, i) => `C${"C".repeat(i)}O`);
    const values = new Map([[TOTAL, Float64Array.from({ length: n }, (_, i) => i / n)]]);
    expect(sampleForMap({ n, smiles, values }, 100)).toHaveLength(n);
    const sample = sampleForMap({ n, smiles, values }, 20);
    expect(sample).toHaveLength(20);
    expect(sample).toContain(49);
    expect(sample).toContain(46);
    expect(sampleForMap({ n, smiles, values }, 20)).toEqual(sample);
  });
});

describe("colours", () => {
  it("maps 0..1 to viridis ends and clamps", () => {
    expect(viridis(0)).toEqual([68, 1, 84]);
    expect(viridis(1)).toEqual([253, 231, 37]);
    expect(viridis(2)).toEqual([253, 231, 37]);
    expect(viridis(NaN)).toEqual([68, 1, 84]);
    expect(viridisCss()).toContain("rgb(68,1,84)");
    expect(viridisCss(0.5)).not.toContain("rgb(68,1,84)");
  });
  it("trims outliers from the colour domain", () => {
    const values = [...Array.from({ length: 100 }, (_, i) => i / 100), 1000];
    expect(colourDomain(values)![1]).toBeLessThan(2);
    expect(colourDomain([NaN])).toBeNull();
    expect(colourDomain([3, 3])).toEqual([3, 4]);
  });
  it("gives dense clusters a higher density", () => {
    const xs = [...Array(30).fill(0), 10];
    const ys = [...Array(30).fill(0), 10];
    const d = pointDensity(xs, ys);
    expect(d[0]).toBe(1);
    expect(d[30]).toBeLessThan(0.2);
  });
});

describe("endpoint types", () => {
  const spread = (lo: number, hi: number, n = 50) => Float64Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1));
  it("gives each known raw value its label, unit, decimals and scale", () => {
    const cases: [string, Float64Array, string, number, string, string][] = [
      ["Boltz affinity", spread(4, 9), "Affinity (pIC50)", 1, "linear", "up"],
      ["QED", spread(0.2, 0.9), "QED", 2, "linear", "up"],
      ["Molecular weight", spread(200, 600), "MW (Da)", 0, "linear", "none"],
      ["cLogP", spread(-1, 6), "cLogP", 1, "linear", "none"],
      ["LogD", spread(-1, 5), "LogD", 1, "linear", "none"],
      ["TPSA", spread(20, 140), "TPSA (Å²)", 1, "linear", "none"],
      ["Synthetic accessibility", spread(1, 6), "SA", 1, "linear", "down"],
      ["Aqueous solubility", spread(-8, -2), "Solubility (log M)", 1, "linear", "up"],
      ["Caco-2 permeability", spread(-7, -4), "Permeability (log cm/s)", 2, "linear", "up"],
      ["Caco-2 permeability", Float64Array.from({ length: 50 }, (_, i) => 10 ** (-7 + (3 * i) / 49)), "Permeability (cm/s)", 2, "log", "up"],
      ["IC50 nM", spread(1, 10_000), "IC50 nM (nM)", 0, "log", "down"],
      ["Contacts", Float64Array.from([0, 0, 1, 1, 1, 2, 2, 3, 40, 60]), "Contacts", 0, "log1p", "up"],
    ];
    for (const [name, values, label, digits, scale, better] of cases) {
      const e = describeEndpoint(name, values);
      expect([heading(e), e.digits, e.scale, e.better], name).toEqual([label, digits, scale, better]);
    }
  });
  it("labels a score without a raw value as a score", () => {
    const e = describeEndpoint("Pose quality", null);
    expect([heading(e), e.source, e.digits]).toEqual(["Pose quality (score)", "score", 2]);
  });
  it("scales unknown values by how they spread", () => {
    expect(autoScale(Float64Array.from({ length: 50 }, (_, i) => 10 ** (-3 + (4 * i) / 49)))).toBe("log");
    expect(autoScale(spread(0.001, 10))).toBe("linear");
    expect(autoScale(spread(-1, 10))).toBe("linear");
    expect(autoScale([NaN, 1])).toBe("linear");
    expect(describeEndpoint("Something", spread(0.1, 0.9)).digits).toBe(3);
  });
});

describe("goalOf", () => {
  const affinity = describeEndpoint("Boltz affinity", Float64Array.of(5, 7));
  const mw = describeEndpoint("Molecular weight", Float64Array.of(300, 400));
  const sa = describeEndpoint("Synthetic accessibility", Float64Array.of(2, 3));
  it("reads the side from the TTP transform", () => {
    expect(goalOf(sa, { type: "reverse_sigmoid", low: 1, high: 4 })).toEqual({ direction: "down" });
    expect(goalOf(sa, { type: "sigmoid", low: 1, high: 4 })).toEqual({ direction: "up" });
    expect(goalOf(mw, { type: "double_sigmoid", low: 200, high: 500 })).toEqual({ direction: "target", target: [200, 500] });
    expect(goalOf(mw, { type: "step", low: 500, high: 200 })).toEqual({ direction: "target", target: [200, 500] });
  });
  it("flips with pIC50: lower log IC50 is a higher pIC50", () => {
    expect(goalOf(affinity, { type: "reverse_sigmoid", low: -3, high: 1 })).toEqual({ direction: "up" });
    expect(goalOf(affinity, { type: "double_sigmoid", low: -3, high: 1 })).toEqual({ direction: "target", target: [5, 9] });
  });
  it("falls back to the type's side, and scores are better high", () => {
    expect(goalOf(sa, undefined)).toEqual({ direction: "down" });
    expect(goalOf(mw, undefined)).toEqual({ direction: "none" });
    expect(goalOf(mw, null)).toEqual({ direction: "up" });
    expect(goalOf(describeEndpoint("Pose quality", null), { type: "reverse_sigmoid" })).toEqual({ direction: "up" });
  });
});

describe("colourScale", () => {
  const values = Float64Array.from({ length: 101 }, (_, i) => i);
  it("is bright at the better end", () => {
    const up = colourScale(values, "linear", { direction: "up" })!;
    const down = colourScale(values, "linear", { direction: "down" })!;
    expect(up.paint(1, false)).toEqual(viridis(1));
    expect(down.paint(0, false)).toEqual(viridis(1));
    expect(up.goodness(90)).toBeGreaterThan(up.goodness(10));
    expect(down.goodness(10)).toBeGreaterThan(down.goodness(90));
  });
  it("uses the 2nd and 98th percentiles, NaN aside", () => {
    const s = colourScale([...values, 10_000, NaN], "linear", { direction: "up" })!;
    expect(s.hi).toBeLessThan(101);
    expect(s.position(10_000)).toBe(1);
    expect(Number.isNaN(s.position(NaN))).toBe(true);
    expect(colourScale([NaN], "linear", { direction: "up" })).toBeNull();
  });
  it("diverges about a target window", () => {
    const s = colourScale(values, "linear", { direction: "target", target: [40, 60] })!;
    expect(s.position(50)).toBeCloseTo(0.5);
    expect(s.goodness(50)).toBeCloseTo(1);
    expect(s.goodness(50)).toBeGreaterThan(s.goodness(95));
    expect(s.paint(0, false)).not.toEqual(s.paint(1, false));
  });
  it("spreads a log scale by decades", () => {
    const s = colourScale([1, 10, 100, 1000, 10_000], "log", { direction: "down" })!;
    expect(s.position(100)).toBeCloseTo(0.5);
    expect(s.ticks).toEqual([1, 100, 10_000]);
    expect(Number.isNaN(s.position(-1))).toBe(true);
  });
});

describe("numbers", () => {
  it("formats with fixed decimals and never in e-notation", () => {
    expect(formatValue(0.71234, 2)).toBe("0.71");
    expect(formatValue(412.6, 0)).toBe("413");
    expect(formatValue(-4.12, 1)).toBe("−4.1");
    expect(formatValue(-0.001, 2)).toBe("0.00");
    expect(formatValue(0.00072, 2)).toBe("0.00");
    expect(formatValue(123_456, 1)).toBe("123,456");
    expect(formatValue(NaN)).toBe("—");
    expect(formatValue(7.2e-6, 2, "log")).toBe("7.2×10⁻⁶");
    expect(formatValue(1e-6, 2, "log")).toBe("10⁻⁶");
    expect(formatValue(0.0123, 2, "log")).toBe("0.012");
    expect(formatValue(2500, 0, "log")).toBe("2,500");
    for (const v of [1e-9, 3e-5, 7.2e-4, 1e9]) expect(formatValue(v, 2, "log")).not.toMatch(/e/);
  });
  it("picks round ticks", () => {
    expect(niceTicks(0, 1, 3)).toEqual([0, 0.5, 1]);
    expect(niceTicks(213, 587, 4)).toEqual([300, 400, 500]);
    expect(niceTicks(1, 1)).toEqual([1]);
    expect(axisTicks("log", -6, -3)).toEqual([1e-6, 1e-5, 1e-4, 1e-3]);
    expect(axisTicks("log1p", 0, 2)).toEqual([0, 2, 10, 50]);
  });
  it("jitters each SMILES the same way every time, within [−1, 1)", () => {
    expect(jitter("CCO")).toBe(jitter("CCO"));
    expect(jitter("CCO")).not.toBe(jitter("CCN"));
    const all = Array.from({ length: 500 }, (_, i) => jitter(`C${i}`));
    expect(Math.min(...all)).toBeGreaterThanOrEqual(-1);
    expect(Math.max(...all)).toBeLessThan(1);
    expect(Math.abs(all.reduce((a, b) => a + b, 0) / all.length)).toBeLessThan(0.1);
  });
});
