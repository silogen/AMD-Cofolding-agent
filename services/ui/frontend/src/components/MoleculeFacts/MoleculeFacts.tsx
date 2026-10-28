// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { StructureMetrics } from "../../types/api";
import { cofoldBars, predictionBars, propertyBars, scoreBars } from "../../lib/factBars";
import type { Fact, FactGroup } from "../../lib/moleculeFacts";
import { FactColumns } from "./FactBars";

export interface MoleculeFactsProps {
  /** Total score, 0–1. */
  score?: number | null;
  /** The score's components as desirabilities (componentFacts). */
  components?: Fact[];
  /** Boltz and PoseBusters values of the predicted complex. */
  cofold?: StructureMetrics | null;
  /** Shows a 3D button next to the cofold values. */
  onOpenComplex?: () => void;
  /** QED, MW, cLogP... (propertyFacts). */
  properties?: Fact[];
  /** Endpoint predictions, one group per source (predictionGroups, runPredictions). */
  predictions?: FactGroup[];
}

/**
 * What is known about a molecule, one value per row with a short bar: cofold,
 * predictions and score next to the properties when the enclosing
 * `@container` is wide, stacked when not. Renders nothing when nothing is known.
 */
export function MoleculeFacts({ score, components = [], cofold, onOpenComplex, properties = [], predictions = [] }: MoleculeFactsProps) {
  const scored = scoreBars(score, components);
  const cofolded = cofold ? cofoldBars(cofold) : [];
  const left = [
    ...(cofolded.length || onOpenComplex
      ? [
          {
            title: "Cofold",
            facts: cofolded,
            action: onOpenComplex && (
              <button
                type="button"
                onClick={onOpenComplex}
                title="Show the predicted complex in the viewer"
                className="hover:bg-inset -my-0.5 cursor-pointer rounded px-1.5 py-0.5 text-[10px] font-semibold"
                style={{ color: "var(--color-structure)" }}
              >
                3D
              </button>
            ),
          },
        ]
      : []),
    ...predictionBars(predictions),
    ...(scored.length ? [{ title: "Score", facts: scored }] : []),
  ];
  return <FactColumns left={left} right={properties.length ? [{ title: "Properties", facts: propertyBars(properties) }] : []} />;
}
