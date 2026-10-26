// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { MoleculePanel, StructurePanel, TablePanel, Turn, TurnBlock } from "../types/api";

type ToolStep = Extract<TurnBlock, { kind: "tool" }>;

export interface Sourced<T> {
  item: T;
  /** The tool that produced it. */
  tool: string;
  /** Epoch ms of the step that produced it. */
  at: number;
}

export interface Artifacts {
  structures: Sourced<StructurePanel>[];
  molecules: Sourced<MoleculePanel>[];
  tables: Sourced<TablePanel>[];
  steps: ToolStep[];
}

/**
 * Everything the conversation produced, newest first and without duplicates:
 * a structure by path, a molecule by SMILES. Tables are kept per step, since
 * two calls of the same tool are two different results.
 */
export function collectArtifacts(turns: Turn[]): Artifacts {
  const structures = new Map<string, Sourced<StructurePanel>>();
  const molecules = new Map<string, Sourced<MoleculePanel>>();
  const tables: Sourced<TablePanel>[] = [];
  const steps: ToolStep[] = [];

  for (const turn of turns) {
    if (turn.role !== "agent") continue;
    for (const block of turn.blocks) {
      if (block.kind !== "tool") continue;
      steps.push(block);
      for (const panel of block.result?.panels ?? []) {
        const sourced = { tool: block.name, at: block.at };
        if (panel.kind === "structure") {
          structures.delete(panel.path);
          structures.set(panel.path, { item: panel, ...sourced });
        } else if (panel.kind === "molecule") {
          molecules.delete(panel.smiles);
          molecules.set(panel.smiles, { item: panel, ...sourced });
        } else {
          tables.push({ item: panel, ...sourced });
        }
      }
    }
  }

  return {
    structures: [...structures.values()].reverse(),
    molecules: [...molecules.values()].reverse(),
    tables: tables.reverse(),
    steps: steps.reverse(),
  };
}
