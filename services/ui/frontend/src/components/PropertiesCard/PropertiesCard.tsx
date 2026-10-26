// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useMemo, useState } from "react";
import type { PropertiesPanel } from "../../types/api";
import type { Theme } from "../../hooks/useTheme";
import { propertiesSheet } from "../../lib/csv";
import { endpointLabel, endpointRange, endpointUnit, formatValue, serviceName } from "../../lib/properties";
import { radarData } from "../../lib/radar";
import { ExportButtons } from "../DataTable/ExportButtons";
import { MolDepiction } from "../MolDepiction/MolDepiction";
import { RadarChart } from "./RadarChart";

const RADAR_MAX = 5;

interface Props {
  panel: PropertiesPanel;
  theme: Theme;
  /** Show every molecule and endpoint in the workbench. */
  onOpen?: () => void;
  onOpenMolecule?: (smiles: string) => void;
  /** Rows to show; all when unset. */
  limit?: number;
}

/**
 * Predicted properties, molecules by endpoints, each value with its
 * ensemble spread and a bar placing it among the other molecules.
 */
export function PropertiesCard({ panel, theme, onOpen, onOpenMolecule, limit }: Props) {
  const [view, setView] = useState<"table" | "radar">("table");
  const radar = useMemo(() => radarData(panel, RADAR_MAX), [panel]);
  const rows = limit ? panel.molecules.slice(0, limit) : panel.molecules;
  const hidden = panel.molecules.length - rows.length;
  const outside = panel.molecules.filter((m) => m.adPass === false).length;
  const showRadar = view === "radar" && radar !== null;

  return (
    <section className="bg-raised border-subtle shadow-card overflow-hidden rounded-xl border">
      <header className="border-subtle flex flex-wrap items-center gap-2 border-b px-3 py-2">
        <span aria-hidden="true" className="size-2 shrink-0 rounded-full" style={{ background: "var(--color-table)" }} />
        <span className="truncate text-xs font-semibold">{serviceName(panel)} predictions</span>
        <span className="text-faint ml-auto shrink-0 text-[11px]">
          {panel.molecules.length} molecule{panel.molecules.length === 1 ? "" : "s"} · {panel.endpoints.length} endpoints
        </span>
        {radar && (
          <span role="group" aria-label="View" className="bg-sunken flex shrink-0 rounded-md p-0.5 text-[10px] font-medium">
            {(["table", "radar"] as const).map((v) => (
              <button
                key={v}
                type="button"
                onClick={() => setView(v)}
                aria-pressed={view === v}
                className={`cursor-pointer rounded px-1.5 py-px capitalize touch:min-h-11 ${view === v ? "bg-raised text-body shadow-card" : "text-faint hover:text-body"}`}
              >
                {v}
              </button>
            ))}
          </span>
        )}
        <ExportButtons name={`${serviceName(panel)} predictions`} sheet={() => propertiesSheet(panel)} scope={`all ${panel.molecules.length} molecules`} />
      </header>
      {showRadar && <RadarChart axes={radar.axes} series={radar.series} total={panel.molecules.filter((m) => m.ok).length} />}
      <div className={`overflow-x-auto ${showRadar ? "hidden" : ""}`}>
        <table className="w-full text-[11px]">
          <thead>
            <tr className="text-muted border-subtle border-b text-left">
              <th className="px-2 py-1.5 font-medium">Molecule</th>
              {panel.endpoints.map((endpoint) => {
                const unit = endpointUnit(panel, endpoint);
                return (
                  <th key={endpoint} className="px-2 py-1.5 text-right font-medium whitespace-nowrap" title={endpoint}>
                    {endpointLabel(endpoint)}
                    {unit && <span className="text-faint block text-[9px] font-normal">{unit}</span>}
                  </th>
                );
              })}
              <th className="px-2 py-1.5 text-center font-medium" title="Inside the models' applicability domain">
                AD
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((molecule, index) => (
              <tr key={index} className="border-subtle border-b last:border-b-0">
                <td className="px-2 py-1">
                  {molecule.smiles ? (
                    <button
                      type="button"
                      onClick={() => molecule.smiles && onOpenMolecule?.(molecule.smiles)}
                      className="flex cursor-pointer items-center gap-2 text-left"
                      title={molecule.smiles}
                    >
                      <MolDepiction smiles={molecule.smiles} theme={theme} width={120} height={80} className="h-10 w-14 shrink-0" />
                      <span className="max-w-28 truncate font-mono text-[10px]">{molecule.label ?? molecule.smiles}</span>
                    </button>
                  ) : (
                    <span className="text-faint">—</span>
                  )}
                </td>
                {molecule.ok ? (
                  panel.endpoints.map((endpoint) => {
                    const value = molecule.values[endpoint];
                    const range = endpointRange(panel, endpoint);
                    const share = value && range ? (value.mean - range[0]) / (range[1] - range[0]) : null;
                    return (
                      <td key={endpoint} className="px-2 py-1 text-right font-mono whitespace-nowrap tabular-nums">
                        {formatValue(value)}
                        {share !== null && (
                          <span className="bg-inset mt-0.5 ml-auto block h-0.5 w-12 overflow-hidden rounded-full">
                            <span className="block h-full rounded-full" style={{ width: `${Math.max(6, share * 100)}%`, background: "var(--color-table)" }} />
                          </span>
                        )}
                      </td>
                    );
                  })
                ) : (
                  <td colSpan={panel.endpoints.length} className="px-2 py-1" style={{ color: "var(--color-accent)" }}>
                    {molecule.error ?? "Prediction failed"}
                  </td>
                )}
                <td className="px-2 py-1 text-center">
                  {molecule.adPass === null ? (
                    <span className="text-faint">—</span>
                  ) : (
                    <span
                      title={molecule.adPass ? "Inside the applicability domain" : "Outside the applicability domain: treat with caution"}
                      style={{ color: molecule.adPass ? "var(--color-ok)" : "var(--color-accent)" }}
                    >
                      {molecule.adPass ? "✓" : "!"}
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {(hidden > 0 || outside > 0 || onOpen) && (
        <footer className="border-subtle text-muted flex items-center gap-2 border-t px-3 py-1.5 text-[11px]">
          {outside > 0 && (
            <span style={{ color: "var(--color-accent)" }}>
              {outside} outside the applicability domain
            </span>
          )}
          {hidden > 0 && <span>{hidden} more</span>}
          {onOpen && (
            <button type="button" onClick={onOpen} className="hover:text-body ml-auto cursor-pointer font-medium">
              Open in workbench
            </button>
          )}
        </footer>
      )}
    </section>
  );
}
