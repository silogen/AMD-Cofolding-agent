// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { PropertiesPanel, PropertyValue } from "../types/api";

/** Readable names for the ADMET service's endpoints. Others are shown as they come. */
const ENDPOINT_LABELS: Record<string, string> = {
  logp: "LogP",
  logd: "LogD",
  aqueous_solubility: "Aq. solubility",
  kinetic_solubility: "Kin. solubility",
  caco2_permeability: "Caco-2 Papp",
  efflux_mdck: "MDCK efflux",
  efflux_caco2: "Caco-2 efflux",
};

export function endpointLabel(endpoint: string): string {
  return ENDPOINT_LABELS[endpoint] ?? endpoint.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

/** The unit of an endpoint, from the first molecule that reports one. */
export function endpointUnit(panel: PropertiesPanel, endpoint: string): string | null {
  for (const molecule of panel.molecules) {
    const unit = molecule.values[endpoint]?.unit;
    if (unit) return unit;
  }
  return null;
}

/** Conventional short forms for the units the predictors report. */
const SHORT_UNITS: Record<string, string> = {
  "log10(mol/L)": "log₁₀ M",
  "log10(cm/s)": "log₁₀ cm/s",
  "log10(efflux ratio)": "log₁₀ ER",
  "log10(IC50/µM)": "log₁₀ µM",
};

/** Units past this length stay in the header's tooltip only, so the header keeps one line. */
const UNIT_MAX = 12;

/**
 * A compact unit for a table header, or null when it would say nothing the
 * endpoint's label does not. The full unit stays in the header's tooltip.
 */
export function shortUnit(unit: string | null | undefined): string | null {
  if (!unit) return null;
  if (SHORT_UNITS[unit]) return SHORT_UNITS[unit];
  const base = unit.replace(/\s*\(dimensionless\)$/i, "").trim();
  if (/^log\s?[pd]$/i.test(base)) return null;
  const log10 = /^log10\s?\((.+)\)$/i.exec(base);
  const short = log10 ? `log₁₀ ${log10[1]}` : base;
  return short && short.length <= UNIT_MAX ? short : null;
}

/** The endpoint's key and full unit, for a header tooltip. */
export function endpointTitle(endpoint: string, unit: string | null): string {
  return unit ? `${endpoint} — ${unit}` : endpoint;
}

export function formatValue(value: PropertyValue | undefined): string {
  if (!value) return "—";
  const digits = Math.abs(value.mean) >= 100 ? 0 : Math.abs(value.mean) >= 10 ? 1 : 2;
  const mean = value.mean.toFixed(digits);
  return value.std !== undefined && value.std > 0 ? `${mean} ± ${value.std.toFixed(digits)}` : mean;
}

/** [min, max] of an endpoint across the molecules, for the inline bars. */
export function endpointRange(panel: PropertiesPanel, endpoint: string): [number, number] | null {
  const values = panel.molecules.map((m) => m.values[endpoint]?.mean).filter((v): v is number => v !== undefined);
  if (values.length < 2) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  return max > min ? [min, max] : null;
}

export function serviceName(panel: PropertiesPanel): string {
  if (panel.service === "retrodmta" || panel.tool === "predict_retrodmta") return `RetroDMTA${panel.target ? ` · ${panel.target}` : ""}`;
  return "ADMET";
}
