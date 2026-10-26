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
