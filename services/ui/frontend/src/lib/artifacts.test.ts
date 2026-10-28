// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";

import type { InteractionsPanel, Panel, StructurePanel, ToolResultEvent, Turn } from "../types/api";
import { collectArtifacts, contactsFor, isResultPanel, mergeTurnPanels, panelKey, sameFile } from "./artifacts";

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

describe("mergeTurnPanels", () => {
  const smiles = "Cc1nc(F)c(C2CCC(O)NC2)cc1";
  const file = (run: string) => `predictions/out_${run}/boltz_results_batch_${run}/predictions/mol_0/mol_0_model_0.pdb`;
  const scored = (run: string, metrics: StructurePanel["metrics"]): StructurePanel => ({
    kind: "structure",
    tool: "score_molecules",
    name: "mol_0_model_0.pdb",
    path: file(run),
    url: `/api/artifacts/${file(run)}`,
    format: "pdb",
    smiles,
    label: "mol_0",
    metrics,
  });
  // The profiler names the file it read, so its result also yields a bare structure.
  const profiled = (run: string, totalContacts: number): Panel[] => [
    { kind: "structure", tool: "profile_interactions", name: "mol_0_model_0.pdb", path: file(run), url: `/api/artifacts/${file(run)}`, format: "pdb" },
    { ...profile, name: "mol_0", structurePath: file(run), smiles, counts: { hydrophobic: totalContacts }, totalContacts },
  ];
  // Scored, scored again with PoseBusters in another run, then each run profiled: what the user saw as two of each card.
  const turn: Panel[] = [
    scored("b0ad161c2f10", { confidence: 0.98, ligandIptm: 0.99, affinity: -1.85, binderProbability: 0.99 }),
    scored("d68a88ff09ab", { confidence: 0.98, ligandIptm: 0.99, posebustersPassed: 18, posebustersTotal: 22, affinity: -1.82 }),
    ...profiled("b0ad161c2f10", 35),
    ...profiled("d68a88ff09ab", 34),
  ];

  it("reproduces the duplicates: one key per file and per contact count gives two of each card", () => {
    const keys = new Set(turn.map(panelKey));
    expect([...keys].filter((key) => key.startsWith("s:"))).toHaveLength(2);
    expect([...keys].filter((key) => key.startsWith("i:"))).toHaveLength(2);
  });

  it("shows a ligand scored again as one structure with the newest run's values", () => {
    const structures = mergeTurnPanels(turn).filter((panel) => panel.kind === "structure");
    expect(structures).toHaveLength(1);
    expect(structures[0]).toMatchObject({ path: file("d68a88ff09ab"), smiles, metrics: { affinity: -1.82, posebustersPassed: 18 } });
  });

  it("shows one contact profile per complex, the newest", () => {
    const contacts = mergeTurnPanels(turn).filter((panel): panel is InteractionsPanel => panel.kind === "interactions");
    expect(contacts).toHaveLength(1);
    expect(contacts[0].totalContacts).toBe(34);
  });

  it("replaces a profile of the same file measured again", () => {
    const merged = mergeTurnPanels([...profiled("b0ad161c2f10", 35), ...profiled("b0ad161c2f10", 36)]);
    expect(merged.filter((panel) => panel.kind === "interactions")).toEqual([expect.objectContaining({ totalContacts: 36 })]);
  });

  it("keeps the measured contacts when profiling again fails", () => {
    const [, good] = profiled("b0ad161c2f10", 35);
    const failed = { ...good, error: "timeout", counts: {}, totalContacts: 0 };
    expect(mergeTurnPanels([good, failed])).toEqual([good]);
  });

  it("keeps the metrics of a file the profiler names again", () => {
    const first = scored("b0ad161c2f10", { confidence: 0.98 });
    const [mention] = profiled("b0ad161c2f10", 35);
    expect(mergeTurnPanels([first, mention])).toEqual([expect.objectContaining({ tool: "score_molecules", smiles, metrics: { confidence: 0.98 } })]);
  });

  it("keeps different ligands and different poses apart", () => {
    const other = { ...scored("d68a88ff09ab", { confidence: 0.9 }), smiles: "CCO" };
    const pose1 = { ...scored("b0ad161c2f10", { confidence: 0.7 }), path: file("b0ad161c2f10").replace("model_0", "model_1") };
    expect(mergeTurnPanels([scored("b0ad161c2f10", { confidence: 0.98 }), other, pose1]).filter((panel) => panel.kind === "structure")).toHaveLength(3);
  });

  it("leaves the workbench one profile of a complex profiled again", () => {
    const steps = [...profiled("b0ad161c2f10", 35), ...profiled("b0ad161c2f10", 36)];
    expect(collectArtifacts([turnWith(steps)]).interactions.map(({ item }) => item.totalContacts)).toEqual([36]);
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
