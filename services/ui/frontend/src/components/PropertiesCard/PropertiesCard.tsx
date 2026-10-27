// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useMemo, useState } from "react";
import type { PropertiesPanel } from "../../types/api";
import type { Theme } from "../../hooks/useTheme";
import { propertiesSheet } from "../../lib/csv";
import { endpointLabel, endpointRange, endpointTitle, endpointUnit, formatValue, serviceName, shortUnit } from "../../lib/properties";
import { radarData } from "../../lib/radar";
import { ExportButtons } from "../DataTable/ExportButtons";
import { MolDepiction } from "../MolDepiction/MolDepiction";
import { SmilesText } from "../MolDepiction/SmilesText";
import { RadarChart } from "./RadarChart";

const RADAR_MAX = 5;
/** Endpoints shown before the "+N more" affordance; the exports always carry them all. */
const SHOWN_ENDPOINTS = 6;

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
  const [allEndpoints, setAllEndpoints] = useState(false);
  const radar = useMemo(() => radarData(panel, RADAR_MAX), [panel]);
  const rows = limit ? panel.molecules.slice(0, limit) : panel.molecules;
  const endpoints = allEndpoints ? panel.endpoints : panel.endpoints.slice(0, SHOWN_ENDPOINTS);
  const hiddenEndpoints = panel.endpoints.length - endpoints.length;
  const hidden = panel.molecules.length - rows.length;
  const outside = panel.molecules.filter((m) => m.adPass === false).length;
  const showRadar = view === "radar" && radar !== null;

  return (
    <section className="@container bg-raised border-subtle shadow-card overflow-hidden rounded-xl border">
      <header className="border-subtle flex flex-wrap items-center gap-2 border-b px-3 py-2">
        <span aria-hidden="true" className="size-2 shrink-0 rounded-full" style={{ background: "var(--color-table)" }} />
        <span className="truncate text-xs font-semibold">{serviceName(panel)} predictions</span>
        <span className="text-faint ml-auto shrink-0 text-[11px]">
          {panel.molecules.length} molecule{panel.molecules.length === 1 ? "" : "s"} · {panel.endpoints.length} endpoints
        </span>
        {(hiddenEndpoints > 0 || allEndpoints) && !showRadar && (
          <button
            type="button"
            onClick={() => setAllEndpoints(!allEndpoints)}
            aria-expanded={allEndpoints}
            title={allEndpoints ? `Show the first ${SHOWN_ENDPOINTS} endpoints only` : `Show the ${hiddenEndpoints} endpoint${hiddenEndpoints === 1 ? "" : "s"} not in the table`}
            className="text-faint hover:text-body hover:bg-inset shrink-0 cursor-pointer rounded px-1 text-[11px] font-medium touch:min-h-11"
          >
            {allEndpoints ? "fewer" : `+${hiddenEndpoints} more`}
          </button>
        )}
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
        <table className="w-full text-xs">
          <thead>
            <tr className="text-muted border-subtle border-b text-left">
              <th className="px-2 py-1.5 font-medium">Molecule</th>
              {endpoints.map((endpoint) => {
                const unit = endpointUnit(panel, endpoint);
                const short = shortUnit(unit);
                return (
                  <th key={endpoint} className="px-2 py-1.5 text-right font-medium whitespace-nowrap" title={endpointTitle(endpoint, unit)}>
                    {endpointLabel(endpoint)}
                    {short && <span className="text-faint ml-1 text-[9px] font-normal">{short}</span>}
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
              <tr key={index} className={`group border-subtle border-b last:border-b-0 ${molecule.ok ? "" : "bg-sunken text-muted"}`}>
                <td className="px-2 py-1">
                  {molecule.smiles ? (
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        disabled={!onOpenMolecule}
                        onClick={() => molecule.smiles && onOpenMolecule?.(molecule.smiles)}
                        title={onOpenMolecule ? "Open in the workbench" : undefined}
                        aria-label={`Open ${molecule.label ?? molecule.smiles} in the workbench`}
                        className="enabled:hover:bg-inset h-12 w-16 shrink-0 rounded enabled:cursor-pointer"
                      >
                        <MolDepiction
                          smiles={molecule.smiles}
                          theme={theme}
                          width={128}
                          height={96}
                          label={molecule.label ?? undefined}
                          className={`size-full ${molecule.ok ? "" : "opacity-60"}`}
                        />
                      </button>
                      <span className="flex min-w-0 flex-col">
                        <SmilesText smiles={molecule.smiles} label={molecule.label} className="max-w-28 text-[10px] @xl:max-w-44" />
                        {!molecule.ok && <FailureReason error={molecule.error} />}
                      </span>
                    </div>
                  ) : molecule.ok ? (
                    <span className="text-faint">—</span>
                  ) : (
                    <FailureReason error={molecule.error} />
                  )}
                </td>
                {molecule.ok ? (
                  endpoints.map((endpoint) => {
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
                  <td colSpan={endpoints.length} className="text-faint px-2 py-1">
                    <span className="sr-only">No predictions</span>
                    <span aria-hidden="true">—</span>
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
          {hidden > 0 && <span>{hidden} more molecules</span>}
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

function FailureReason({ error }: { error: string | null }) {
  const reason = error ?? "the prediction failed";
  return (
    <span className="mt-0.5 flex max-w-40 min-w-0 items-center gap-1 text-[10px] @xl:max-w-52" title={`Not predicted: ${reason}`}>
      <span aria-hidden="true" className="size-1.5 shrink-0 rounded-full" style={{ background: "var(--color-warn)" }} />
      <span className="truncate">Not predicted: {reason}</span>
    </span>
  );
}
