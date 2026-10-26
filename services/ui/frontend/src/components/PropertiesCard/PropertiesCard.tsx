// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { PropertiesPanel } from "../../types/api";
import type { Theme } from "../../hooks/useTheme";
import { endpointLabel, endpointRange, endpointUnit, formatValue, serviceName } from "../../lib/properties";
import { MolDepiction } from "../MolDepiction/MolDepiction";

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
  const rows = limit ? panel.molecules.slice(0, limit) : panel.molecules;
  const hidden = panel.molecules.length - rows.length;
  const outside = panel.molecules.filter((m) => m.adPass === false).length;

  return (
    <section className="bg-raised border-subtle shadow-card overflow-hidden rounded-xl border">
      <header className="border-subtle flex items-center gap-2 border-b px-3 py-2">
        <span aria-hidden="true" className="size-2 shrink-0 rounded-full" style={{ background: "var(--color-table)" }} />
        <span className="truncate text-xs font-semibold">{serviceName(panel)} predictions</span>
        <span className="text-faint ml-auto shrink-0 text-[11px]">
          {panel.molecules.length} molecule{panel.molecules.length === 1 ? "" : "s"} · {panel.endpoints.length} endpoints
        </span>
      </header>
      <div className="overflow-x-auto">
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
