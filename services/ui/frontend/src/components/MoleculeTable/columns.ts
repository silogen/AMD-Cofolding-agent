// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { formatEndpoint, heading, TOTAL, type Endpoint, type Molecules } from "../../lib/runAnalysis";
import type { MoleculeCell, MoleculeColumn } from "./MoleculeTable";

/** An endpoint as a column: its unit under the label, the total in bold. */
export function endpointColumn(e: Endpoint, colour: string | undefined): MoleculeColumn {
  return {
    key: e.key,
    label: e.label,
    unit: e.source === "score" ? "score" : e.unit,
    title: `${heading(e)}${e.key !== e.label ? ` · ${e.key}` : ""}${e.source === "score" ? " · 0–1 score: no raw value in the CSV" : e.source === "raw" ? " · raw value; the bar is its 0–1 score" : ""}`,
    strong: e.key === TOTAL,
    colour,
  };
}

/** Molecule *i*'s value under each endpoint, with a bar of its score when the value is a raw one. */
export function endpointCells(endpoints: readonly Endpoint[], data: Pick<Molecules, "values" | "scores">, i: number): Record<string, MoleculeCell> {
  const cells: Record<string, MoleculeCell> = {};
  for (const e of endpoints) {
    cells[e.key] = { text: formatEndpoint(e, data.values.get(e.key)?.[i] ?? NaN), score: e.source === "raw" ? (data.scores.get(e.key)?.[i] ?? NaN) : null };
  }
  return cells;
}
