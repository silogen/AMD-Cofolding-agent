// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type {
  InteractionsPanel,
  MoleculePanel,
  Panel,
  PropertiesPanel,
  StructurePanel,
  TablePanel,
  Turn,
  TurnBlock,
} from "../types/api";

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
  properties: Sourced<PropertiesPanel>[];
  interactions: Sourced<InteractionsPanel>[];
  steps: ToolStep[];
}

/**
 * Whether a panel of a tool's result is a result. Older sessions stored a
 * contact profile's fragments (an ester, a benzene) as molecules, and a
 * listing of the shared volume names every complex ever predicted; those
 * stay in the Assets tab.
 */
export function isResultPanel(tool: string, panel: Panel): boolean {
  if (panel.kind === "molecule" && tool === "profile_interactions") return false;
  if (panel.kind === "structure" && tool === "list_shared_assets") return false;
  return true;
}

/** One key per distinct result, so a result shown twice is shown once. */
export function panelKey(panel: Panel): string {
  switch (panel.kind) {
    case "structure":
      return `s:${panel.path}`;
    case "molecule":
      return `m:${panel.smiles}`;
    case "interactions":
      return `i:${panel.structurePath ?? ""}:${panel.smiles ?? ""}:${panel.totalContacts}`;
    case "properties":
      return `p:${panel.tool}:${panel.molecules.map((m) => m.smiles).join(".")}`;
    case "table":
      return `t:${panel.tool}:${JSON.stringify(panel.columns)}:${panel.rows.length}`;
  }
}

/**
 * Everything the conversation produced, newest first and without duplicates:
 * a structure by path, a molecule by SMILES, a contact profile by the
 * structure it was measured on. Tables and predictions are kept per step,
 * since two calls of the same tool are two different results.
 */
export function collectArtifacts(turns: Turn[]): Artifacts {
  const structures = new Map<string, Sourced<StructurePanel>>();
  const molecules = new Map<string, Sourced<MoleculePanel>>();
  const interactions = new Map<string, Sourced<InteractionsPanel>>();
  const tables: Sourced<TablePanel>[] = [];
  const properties: Sourced<PropertiesPanel>[] = [];
  const steps: ToolStep[] = [];

  const latest = <T>(map: Map<string, Sourced<T>>, key: string, value: Sourced<T>) => {
    map.delete(key);
    map.set(key, value);
  };

  for (const turn of turns) {
    if (turn.role !== "agent") continue;
    for (const block of turn.blocks) {
      if (block.kind !== "tool") continue;
      steps.push(block);
      for (const panel of block.result?.panels ?? []) {
        if (!isResultPanel(block.name, panel)) continue;
        const sourced = { tool: block.name, at: block.at };
        switch (panel.kind) {
          case "structure":
            latest(structures, panel.path, { item: panel, ...sourced });
            break;
          case "molecule":
            latest(molecules, panel.smiles, { item: panel, ...sourced });
            break;
          case "interactions":
            latest(interactions, panelKey(panel), { item: panel, ...sourced });
            break;
          case "properties":
            properties.push({ item: panel, ...sourced });
            break;
          case "table":
            tables.push({ item: panel, ...sourced });
            break;
        }
      }
    }
  }

  // A profile of a complex predicted in another conversation still makes
  // that complex part of this one, so its contacts can be shown on it.
  for (const { item, tool, at } of interactions.values()) {
    const path = item.structurePath;
    if (!path || structures.has(path)) continue;
    const name = path.split("/").pop() ?? path;
    const format = /\.(cif|mmcif)$/i.test(name) ? "cif" : "pdb";
    const structure: StructurePanel = { kind: "structure", tool, name, path, url: `/api/artifacts/${path}`, format };
    if (item.smiles) structure.smiles = item.smiles;
    structures.set(path, { item: structure, tool, at });
  }

  return {
    structures: [...structures.values()].reverse(),
    molecules: [...molecules.values()].reverse(),
    tables: tables.reverse(),
    properties: properties.reverse(),
    interactions: [...interactions.values()].reverse(),
    steps: steps.reverse(),
  };
}

/** The run a job file belongs to, from its path: "out_6b5e3f1a686f" or "job_d0cbeb98" give the id's first 8 characters. */
export function runOf(path: string): string | null {
  const match = /(?:^|\/)(?:out|job|batch)_([0-9a-f]{6,})/i.exec(path.replace(/\\/g, "/"));
  return match ? match[1].slice(0, 8) : null;
}

/** What to call a structure: its ligand's name, or its file name without the extension. */
export function structureTitle(panel: StructurePanel): string {
  return panel.label || panel.name.replace(/\.(cif|mmcif|pdb)$/i, "");
}

/**
 * Two paths to the same job file. Tools report them with different
 * prefixes ("/shared/predictions/out_x/…" or "predictions/out_x/…"), so one
 * must end with the other. Every run names its files mol_0_model_0.pdb, so
 * the shared end must reach the run directory.
 */
export function sameFile(a: string, b: string): boolean {
  if (a === b) return true;
  const parts = (path: string) => path.replace(/\\/g, "/").split("/").filter(Boolean);
  const [short, long] = [parts(a), parts(b)].sort((x, y) => x.length - y.length);
  const tail = long.slice(-short.length);
  return short.length >= 2 && short.every((part, i) => part === tail[i]) && (runOf(a) === runOf(b) || short.length === long.length);
}

/** The contact profile of a structure: measured on that file, or failing that on the same ligand in the same run. */
export function contactsFor(structure: StructurePanel, all: Sourced<InteractionsPanel>[]): InteractionsPanel | null {
  const byPath = all.find(({ item }) => item.structurePath && sameFile(item.structurePath, structure.path));
  if (byPath) return byPath.item;
  const run = runOf(structure.path);
  const bySmiles = structure.smiles
    ? all.find(({ item }) => item.smiles === structure.smiles && item.structurePath && runOf(item.structurePath) === run)
    : undefined;
  return bySmiles?.item ?? null;
}
