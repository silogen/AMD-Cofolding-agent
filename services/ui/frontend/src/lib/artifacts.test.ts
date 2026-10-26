// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";

import type { InteractionsPanel, Panel, StructurePanel, ToolResultEvent, Turn } from "../types/api";
import { collectArtifacts } from "./artifacts";

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

function turnWith(panels: Panel[]): Turn {
  return {
    role: "agent",
    id: "t1",
    at: 1,
    done: true,
    blocks: [
      {
        kind: "tool",
        id: "c1",
        name: "profile_interactions",
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
