// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import type { PropertiesPanel } from "../types/api";
import { endpointLabel, endpointRange, endpointTitle, endpointUnit, formatValue, shortUnit } from "./properties";

const panel = (): PropertiesPanel => ({
  kind: "properties",
  tool: "predict_admet",
  service: "admet",
  target: null,
  endpoints: ["logp", "aqueous_solubility"],
  molecules: [
    { smiles: "CCO", standardized: null, label: null, ok: true, error: null, adPass: true, values: { logp: { mean: -0.31, std: 0.07, unit: "logP (dimensionless)" }, aqueous_solubility: { mean: -0.86, unit: "log10(mol/L)" } } },
    { smiles: "CCN", standardized: null, label: null, ok: true, error: null, adPass: null, values: { logp: { mean: -1.2 } } },
  ],
});

describe("properties", () => {
  it("names the ADMET endpoints and falls back to the key", () => {
    expect(endpointLabel("aqueous_solubility")).toBe("Aq. solubility");
    expect(endpointLabel("FXA_pKi")).toBe("FXA pKi");
  });

  it("takes an endpoint's unit from the first molecule reporting one", () => {
    expect(endpointUnit(panel(), "aqueous_solubility")).toBe("log10(mol/L)");
    expect(endpointUnit(panel(), "unknown")).toBeNull();
  });

  it("shortens the units the predictors report", () => {
    expect(shortUnit("log10(mol/L)")).toBe("log₁₀ M");
    expect(shortUnit("log10(cm/s)")).toBe("log₁₀ cm/s");
    expect(shortUnit("log10(efflux ratio)")).toBe("log₁₀ ER");
    expect(shortUnit("log10(IC50/µM)")).toBe("log₁₀ µM");
    expect(shortUnit("log10(ER)")).toBe("log₁₀ ER");
    expect(shortUnit("nM")).toBe("nM");
  });

  it("drops units that would repeat the label or crowd the header", () => {
    expect(shortUnit("logP (dimensionless)")).toBeNull();
    expect(shortUnit("logD (dimensionless)")).toBeNull();
    expect(shortUnit("log10(molecules per litre of solvent)")).toBeNull();
    expect(shortUnit(null)).toBeNull();
    expect(shortUnit("")).toBeNull();
  });

  it("keeps the full unit for the header's tooltip", () => {
    expect(endpointTitle("logp", "logP (dimensionless)")).toBe("logp — logP (dimensionless)");
    expect(endpointTitle("logp", null)).toBe("logp");
  });

  it("formats a value with its spread, digits by magnitude", () => {
    expect(formatValue({ mean: -0.31, std: 0.07 })).toBe("-0.31 ± 0.07");
    expect(formatValue({ mean: -0.31 })).toBe("-0.31");
    expect(formatValue({ mean: 345.6, std: 1.2 })).toBe("346 ± 1");
    expect(formatValue(undefined)).toBe("—");
  });

  it("ranges an endpoint over the molecules that have it", () => {
    expect(endpointRange(panel(), "logp")).toEqual([-1.2, -0.31]);
    expect(endpointRange(panel(), "aqueous_solubility")).toBeNull();
  });
});
