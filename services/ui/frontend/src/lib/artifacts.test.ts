// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";

import type { InteractionsPanel, Panel, StructurePanel, ToolResultEvent, Turn } from "../types/api";
import { collectArtifacts, contactsFor, isResultPanel, sameFile } from "./artifacts";

const profile: InteractionsPanel = {
  kind: "interactions",
  tool: "profile_interactions",
  name: "mol_0",
  structurePath: "predictions/out_7aa793db/mol_0_model_0.pdb",
  smiles: "CCO",
  error: null,
  counts: { hydrophobic: 1 },
  contacts: [],
  totalContacts: 1,
};

function turnWith(panels: Panel[], tool = "profile_interactions"): Turn {
  return {
    role: "agent",
    id: "t1",
    at: 1,
    done: true,
    blocks: [
      {
        kind: "tool",
        id: "c1",
        name: tool,
        arguments: {},
        at: 1,
        result: { panels } as unknown as ToolResultEvent,
      },
    ],
  };
}

describe("collectArtifacts", () => {
  it("makes a profiled complex a structure of the conversation", () => {
    const { structures } = collectArtifacts([turnWith([profile])]);
    expect(structures).toHaveLength(1);
    expect(structures[0].item).toMatchObject({
      path: profile.structurePath,
      url: `/api/artifacts/${profile.structurePath}`,
      format: "pdb",
      smiles: "CCO",
    });
  });

  it("keeps the predicted structure and its metrics when both exist", () => {
    const predicted: StructurePanel = {
      kind: "structure",
      tool: "score_molecules",
      name: "mol_0_model_0.pdb",
      path: profile.structurePath!,
      url: `/api/artifacts/${profile.structurePath}`,
      format: "pdb",
      metrics: { confidence: 0.98 },
    };
    const { structures } = collectArtifacts([turnWith([predicted, profile])]);
    expect(structures).toHaveLength(1);
    expect(structures[0].item.metrics).toEqual({ confidence: 0.98 });
  });
});

describe("isResultPanel", () => {
  it("keeps a listing of the shared volume out of the conversation's structures", () => {
    const listed: StructurePanel = { kind: "structure", tool: "list_shared_assets", name: "a.pdb", path: "predictions/out_1/a.pdb", url: "", format: "pdb" };
    expect(collectArtifacts([turnWith([listed], "list_shared_assets")]).structures).toHaveLength(0);
    expect(isResultPanel("score_molecules", listed)).toBe(true);
  });
});

describe("sameFile", () => {
  const run = (id: string) => `/shared/predictions/out_${id}/boltz_results_batch_${id}/predictions/mol_0/mol_0_model_0.pdb`;

  it("matches a file reported with and without the shared prefix", () => {
    expect(sameFile(run("7aa793db23f9"), "predictions/out_7aa793db23f9/boltz_results_batch_7aa793db23f9/predictions/mol_0/mol_0_model_0.pdb")).toBe(true);
  });

  it("tells apart runs whose files have the same name", () => {
    expect(sameFile(run("7aa793db23f9"), run("69943a24688e"))).toBe(false);
    expect(sameFile(run("7aa793db23f9"), "predictions/mol_0/mol_0_model_0.pdb")).toBe(false);
  });
});

describe("contactsFor", () => {
  const structure = (path: string): StructurePanel => ({ kind: "structure", tool: "score_molecules", name: "mol_0_model_0.pdb", path, url: "", format: "pdb", smiles: "CCO" });
  const sourced = (item: InteractionsPanel) => ({ item, tool: "profile_interactions", at: 1 });

  it("does not lend one run's contacts to the same ligand in another run", () => {
    const other = { ...profile, structurePath: "predictions/out_69943a24/mol_0_model_0.pdb" };
    expect(contactsFor(structure("predictions/out_7aa793db/mol_0_model_0.pdb"), [sourced(other)])).toBeNull();
    expect(contactsFor(structure("predictions/out_69943a24/mol_0_model_1.pdb"), [sourced(other)])).toBe(other);
  });
});
