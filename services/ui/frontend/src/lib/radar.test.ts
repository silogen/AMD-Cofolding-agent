// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import type { PropertiesPanel } from "../types/api";
import { axisRule, describeRule, desirability, radarData, radarPoint } from "./radar";

describe("desirability", () => {
  it("rises for higher-is-better endpoints and is clamped", () => {
    const rule = axisRule("aqueous_solubility");
    expect(desirability(rule, -6)).toBe(0);
    expect(desirability(rule, -4.5)).toBeCloseTo(0.5);
    expect(desirability(rule, -3)).toBe(1);
    expect(desirability(rule, -1)).toBe(1);
    expect(desirability(rule, -9)).toBe(0);
  });

  it("falls for lower-is-better endpoints", () => {
    const rule = axisRule("efflux_mdck");
    expect(desirability(rule, 0)).toBe(1);
    expect(desirability(rule, 0.8)).toBeCloseTo(0.5);
    expect(desirability(rule, 2)).toBe(0);
  });

  it("is 1 inside a window and falls off to the edges", () => {
    const rule = axisRule("logp");
    expect(desirability(rule, 2)).toBe(1);
    expect(desirability(rule, 0)).toBeCloseTo(0.5);
    expect(desirability(rule, 4.5)).toBeCloseTo(0.5);
    expect(desirability(rule, 7)).toBe(0);
    expect(desirability(rule, Number.NaN)).toBe(0);
  });
});

describe("axisRule", () => {
  it("knows RetroDMTA endpoints by suffix", () => {
    expect(axisRule("FXA_pKi")).toMatchObject({ direction: "up", lo: 4, hi: 9 });
    expect(axisRule("MMP8_pIC50").direction).toBe("up");
    expect(axisRule("FXA_logD").direction).toBe("window");
    expect(axisRule("FXA_logS").direction).toBe("up");
  });

  it("scales unknown endpoints between the values seen", () => {
    const rule = axisRule("mystery", [2, 4, 6]);
    expect(rule).toMatchObject({ lo: 2, hi: 6, relative: true });
    expect(desirability(rule, 4)).toBeCloseTo(0.5);
    expect(axisRule("mystery", [3]).hi).toBe(4);
  });

  it("explains itself", () => {
    expect(describeRule(axisRule("efflux_caco2"))).toMatch(/^lower is better/);
    expect(describeRule(axisRule("logd"))).toMatch(/window/);
  });
});

const molecule = (label: string, values: Record<string, number>, ok = true) => ({
  smiles: "C",
  standardized: "C",
  label,
  ok,
  error: null,
  adPass: true,
  values: Object.fromEntries(Object.entries(values).map(([k, mean]) => [k, { mean }])),
});

const panel = (endpoints: string[], molecules: PropertiesPanel["molecules"]): PropertiesPanel => ({
  kind: "properties",
  tool: "predict_admet",
  service: "admet",
  target: null,
  endpoints,
  molecules,
});

describe("radarData", () => {
  it("needs three endpoints", () => {
    expect(radarData(panel(["logp", "logd"], [molecule("a", { logp: 2, logd: 2 })]))).toBeNull();
  });

  it("makes one series per molecule with predictions, up to the limit", () => {
    const data = radarData(
      panel(
        ["logp", "aqueous_solubility", "efflux_mdck", "unused"],
        [
          molecule("a", { logp: 2, aqueous_solubility: -3, efflux_mdck: 0 }),
          molecule("failed", {}, false),
          molecule("b", { logp: 7, aqueous_solubility: -6 }),
          molecule("c", { logp: 1 }),
        ],
      ),
      2,
    );
    expect(data?.axes.map((a) => a.endpoint)).toEqual(["logp", "aqueous_solubility", "efflux_mdck"]);
    expect(data?.series.map((s) => s.label)).toEqual(["a", "b"]);
    expect(data?.series[0].scores).toEqual([1, 1, 1]);
    expect(data?.series[1].scores).toEqual([0, 0, null]);
    expect(data?.series[1].raw).toEqual([7, -6, null]);
  });
});

describe("radarPoint", () => {
  it("starts at the top and goes clockwise", () => {
    const [x0, y0] = radarPoint(0, 4, 1, 50, 50, 10);
    expect(x0).toBeCloseTo(50);
    expect(y0).toBeCloseTo(40);
    const [x1, y1] = radarPoint(1, 4, 0.5, 50, 50, 10);
    expect(x1).toBeCloseTo(55);
    expect(y1).toBeCloseTo(50);
  });
});
