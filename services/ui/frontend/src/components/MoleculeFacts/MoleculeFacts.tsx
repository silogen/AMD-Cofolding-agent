// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { ReactNode } from "react";
import type { StructureMetrics } from "../../types/api";
import { cofoldFacts, type Fact, type FactGroup } from "../../lib/moleculeFacts";

export interface MoleculeFactsProps {
  /** Total score, 0–1. */
  score?: number | null;
  /** The score's components as desirabilities (componentFacts). */
  components?: Fact[];
  /** Boltz and PoseBusters values of the predicted complex. */
  cofold?: StructureMetrics | null;
  /** Shows a 3D button next to the cofold values. */
  onOpenComplex?: () => void;
  /** QED, MW, cLogP... (propertyFacts, otherRawValues). */
  properties?: Fact[];
  /** ADMET and RetroDMTA predictions, one group per service (predictionGroups). */
  predictions?: FactGroup[];
}

/** What is known about a molecule, one line per kind of value. Renders nothing when nothing is known. */
export function MoleculeFacts({ score, components = [], cofold, onOpenComplex, properties = [], predictions = [] }: MoleculeFactsProps) {
  const scoreFacts: Fact[] = [...(typeof score === "number" ? [{ label: "Total", value: score.toFixed(2) }] : []), ...components];
  const cofolded = cofold ? cofoldFacts(cofold) : [];
  const rows: { title: string; facts: Fact[]; action?: ReactNode }[] = [];
  if (scoreFacts.length) rows.push({ title: "Score", facts: scoreFacts });
  if (cofolded.length || onOpenComplex) {
    rows.push({
      title: "Cofold",
      facts: cofolded,
      action: onOpenComplex && (
        <button
          type="button"
          onClick={onOpenComplex}
          title="Show the predicted complex in the viewer"
          className="hover:bg-inset -my-0.5 shrink-0 cursor-pointer rounded px-1.5 py-0.5 text-[10px] font-semibold"
          style={{ color: "var(--color-structure)" }}
        >
          3D
        </button>
      ),
    });
  }
  if (properties.length) rows.push({ title: "Properties", facts: properties });
  for (const group of predictions) rows.push(group);
  if (!rows.length) return null;

  return (
    <section className="bg-raised border-subtle divide-y rounded-xl border text-[11px]">
      {rows.map((row) => (
        <div key={row.title} className="flex items-start gap-3 px-3 py-1.5">
          <span className="text-faint w-16 shrink-0 truncate leading-5" title={row.title}>
            {row.title}
          </span>
          <dl className="flex min-w-0 flex-1 flex-wrap gap-x-3.5 leading-5">
            {row.facts.map((fact) => (
              <div key={fact.label} className="flex min-w-0 items-baseline gap-1" title={fact.title}>
                <dt className="text-muted max-w-36 truncate">{fact.label}</dt>
                <dd className="font-mono tabular-nums" style={fact.tone === "warn" ? { color: "var(--color-warn)" } : undefined}>
                  {fact.value ?? "—"}
                  {fact.unit && <span className="text-faint ml-0.5 font-sans">{fact.unit}</span>}
                </dd>
              </div>
            ))}
            {!row.facts.length && <span className="text-faint">No values</span>}
          </dl>
          {row.action}
        </div>
      ))}
    </section>
  );
}
