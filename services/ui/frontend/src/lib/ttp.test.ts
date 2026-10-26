// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import type { InteractionsPanel } from "../types/api";
import {
  applyPrompt,
  asTtpList,
  changes,
  evaluateTransform,
  extractTargetFacts,
  isDirty,
  latestTtp,
  loadDraft,
  makeDraft,
  patchObjective,
  removeObjective,
  retype,
  saveDraft,
  targetRange,
  toolEvents,
  transformCurve,
  ttpUpdates,
  weightShares,
  type ToolEvent,
  type Ttp,
} from "./ttp";

const SEQ = "MSAKAEQLLHAVEGNNKSVRQLLKA";

const ttp: Ttp = {
  id: "brd4-lead",
  name: "BRD4 lead optimisation",
  version: 2,
  target: { protein_sequence: SEQ, pdb_id: "3MXF" },
  pre_scoring: { components: [{ name: "QED", type: "native", component: "QED", weight: 1 }] },
  scoring: {
    aggregation: "geometric_mean",
    objectives: [
      {
        name: "MW",
        type: "native",
        component: "MolecularWeight",
        weight: 0.5,
        transform: { type: "double_sigmoid", low: 200, high: 500, coef_div: 500, coef_si: 20, coef_se: 20 },
      },
      {
        name: "Affinity",
        type: "external",
        service: "boltz",
        service_url: "http://boltz:8080/score",
        weight: 2,
        transform: { type: "reverse_sigmoid", low: -2, high: 1, k: 0.5 },
      },
    ],
  },
  constraints: [{ type: "custom_alerts", smarts: ["[#7]=[#7]"] }],
};

const memory = () => {
  const data = new Map<string, string>();
  return {
    getItem: (k: string) => data.get(k) ?? null,
    setItem: (k: string, v: string) => void data.set(k, v),
    removeItem: (k: string) => void data.delete(k),
    data,
  };
};

describe("transforms", () => {
  it("rewards the window of a double sigmoid", () => {
    const t = ttp.scoring.objectives[0].transform!;
    expect(evaluateTransform(t, 350)!).toBeGreaterThan(0.9);
    expect(evaluateTransform(t, 50)!).toBeLessThan(0.1);
    expect(evaluateTransform(t, 800)!).toBeLessThan(0.1);
  });

  it("rises for sigmoid and falls for reverse sigmoid", () => {
    const up = { type: "sigmoid", low: 0, high: 10, k: 0.5 };
    expect(evaluateTransform(up, 5)).toBeCloseTo(0.5);
    expect(evaluateTransform(up, 10)!).toBeGreaterThan(0.9);
    const down = { type: "reverse_sigmoid", low: 0, high: 10, k: 0.5 };
    expect(evaluateTransform(down, 10)!).toBeLessThan(0.1);
  });

  it("handles steps, decay and mappings", () => {
    expect(evaluateTransform({ type: "step", low: 1, high: 2 }, 1.5)).toBe(1);
    expect(evaluateTransform({ type: "right_step", low: 1 }, 0)).toBe(0);
    expect(evaluateTransform({ type: "left_step", low: 1 }, 0)).toBe(1);
    expect(evaluateTransform({ type: "exponential_decay", k: 1 }, 0)).toBe(1);
    expect(evaluateTransform({ type: "value_mapping" }, 1)).toBeNull();
    expect(transformCurve({ type: "value_mapping" })).toEqual([]);
    expect(transformCurve({ type: "sigmoid", low: 0, high: 1 }, 10)).toHaveLength(11);
  });

  it("keeps shared parameters when the type changes", () => {
    expect(retype({ type: "sigmoid", low: 3, high: 7, k: 0.2 }, "step")).toEqual({ type: "step", low: 3, high: 7 });
    expect(retype(null, "exponential_decay")).toEqual({ type: "exponential_decay", k: 1 });
  });

  it("describes the target window", () => {
    expect(targetRange(ttp.scoring.objectives[0].transform)).toBe("200 – 500");
    expect(targetRange({ type: "right_step", low: 0.5 })).toBe("≥ 0.5");
    expect(targetRange(null)).toBeNull();
  });
});

describe("editing", () => {
  it("patches and removes objectives without touching the original", () => {
    const edited = patchObjective(ttp, "scoring", 0, { weight: 1.5 });
    expect(edited.scoring.objectives[0].weight).toBe(1.5);
    expect(ttp.scoring.objectives[0].weight).toBe(0.5);
    expect(isDirty(ttp, edited)).toBe(true);
    expect(isDirty(ttp, patchObjective(ttp, "scoring", 0, { weight: 0.5 }))).toBe(false);

    const removed = removeObjective(edited, "pre_scoring", 0);
    expect(changes(ttp, removed)).toEqual({ changed: ["MW"], removed: ["QED"], aggregation: false });
  });

  it("sends whole changed sections to update_ttp", () => {
    const edited = patchObjective(ttp, "scoring", 1, { weight: 3 });
    const updates = ttpUpdates(ttp, edited);
    expect(Object.keys(updates)).toEqual(["scoring"]);
    expect((updates.scoring as Ttp["scoring"]).objectives).toHaveLength(2);
  });

  it("asks the agent to apply the edits", () => {
    const prompt = applyPrompt(ttp, patchObjective(ttp, "scoring", 1, { weight: 3 }));
    expect(prompt).toContain('TTP "brd4-lead"');
    expect(prompt).toContain("version 2");
    expect(prompt).toContain("changed: Affinity");
    expect(prompt).toContain('"weight": 3');
  });

  it("gives each weight its share of the total", () => {
    const shares = weightShares(ttp);
    expect(shares.get("Affinity")).toBeCloseTo(2 / 3.5);
  });

  it("keeps drafts per session", () => {
    const store = memory();
    const draft = { base: ttp, draft: patchObjective(ttp, "scoring", 0, { weight: 1 }), savedAt: 1 };
    saveDraft("s1", draft, store);
    expect(loadDraft("s1", store)?.draft.scoring.objectives[0].weight).toBe(1);
    expect(loadDraft("s2", store)).toBeNull();
    saveDraft("s1", null, store);
    expect(loadDraft("s1", store)).toBeNull();
    store.setItem("cofold.ttp-draft.s3", "{not json");
    expect(loadDraft("s3", store)).toBeNull();
    expect(makeDraft(ttp, ttp)).toBeNull();
    expect(makeDraft(ttp, patchObjective(ttp, "scoring", 0, { weight: 1 }))?.base).toBe(ttp);
  });
});

describe("events", () => {
  it("finds the session's latest TTP", () => {
    const events: ToolEvent[] = [
      { type: "tool_call", name: "create_ttp", arguments: { ttp: { ...ttp, version: 1 } } },
      { type: "tool_result", name: "create_ttp", result: JSON.stringify({ status: "created", id: ttp.id, ttp: { ...ttp, version: 1 } }) },
      { type: "tool_result", name: "update_ttp", result: JSON.stringify({ status: "updated", id: ttp.id, ttp }) },
    ];
    expect(latestTtp(events)).toEqual({ id: "brd4-lead", ttp: expect.objectContaining({ version: 2 }) });
    expect(latestTtp([...events, { type: "tool_call", name: "run_optimization", arguments: { ttp_id: "other" } }])).toEqual({ id: "other", ttp: null });
  });

  it("parses list_ttps", () => {
    expect(asTtpList(JSON.stringify([{ id: "a", name: "A", version: 3 }, { bad: 1 }]))).toEqual([{ id: "a", name: "A", version: 3 }]);
    expect(asTtpList("not json")).toEqual([]);
  });

  it("orders steps oldest first", () => {
    const events = toolEvents([
      { kind: "tool", id: "2", name: "b", arguments: {}, at: 2 },
      {
        kind: "tool",
        id: "1",
        name: "a",
        arguments: { x: 1 },
        at: 1,
        result: { type: "tool_result", seq: 1, ts: "", id: "1", name: "a", result: "{}", truncated: false, isError: false, panels: [], elapsedMs: 5 },
      },
    ]);
    expect(events.map((e) => `${e.type}:${e.name}`)).toEqual(["tool_call:a", "tool_result:a", "tool_call:b"]);
  });
});

describe("extractTargetFacts", () => {
  const contacts: InteractionsPanel = {
    kind: "interactions",
    tool: "profile_interactions",
    name: null,
    structurePath: "out/x.cif",
    smiles: "CCO",
    error: null,
    counts: {},
    totalContacts: 3,
    contacts: [
      { type: "hbond", subtype: null, chain: "A", resname: "ASN", resnum: 140, label: "ASN140", receptorAtom: null, ligandAtom: null, ligandAtomIndex: null, distance: 2.9, fragment: null, confidence: null },
      { type: "hydrophobic", subtype: null, chain: "A", resname: "ASN", resnum: 140, label: "ASN140", receptorAtom: null, ligandAtom: null, ligandAtomIndex: null, distance: 3.8, fragment: null, confidence: null },
      { type: "hydrophobic", subtype: null, chain: "A", resname: "TYR", resnum: 97, label: "TYR97", receptorAtom: null, ligandAtom: null, ligandAtomIndex: null, distance: 3.6, fragment: null, confidence: null },
    ],
  };

  it("combines UniProt, TTP target and contacts", () => {
    const facts = extractTargetFacts([
      { type: "tool_call", name: "fetch_target_info", arguments: { query: "O60885" } },
      {
        type: "tool_result",
        name: "fetch_target_info",
        result: JSON.stringify({
          accession: "O60885",
          entry_name: "BRD4_HUMAN",
          protein_name: "Bromodomain-containing protein 4",
          gene: "BRD4",
          organism: "Homo sapiens",
          reviewed: true,
          sequence: SEQ,
          sequence_length: SEQ.length,
          pdb_ids: ["2OSS", "3MXF"],
          pdb_count: 400,
        }),
      },
      { type: "tool_call", name: "create_ttp", arguments: { ttp } },
      { type: "tool_result", name: "profile_interactions", result: "{}", panels: [contacts] },
    ]);
    expect(facts).toMatchObject({
      name: "Bromodomain-containing protein 4",
      gene: "BRD4",
      uniprot: "O60885",
      sequenceLength: SEQ.length,
      chains: ["A"],
      pdbIds: ["2OSS", "3MXF"],
      pdbCount: 400,
      ttpId: "brd4-lead",
      profiledComplexes: 1,
    });
    expect(facts.pocket.map((r) => [r.label, r.contacts, r.types])).toEqual([
      ["ASN140", 2, ["hbond", "hydrophobic"]],
      ["TYR97", 1, ["hydrophobic"]],
    ]);
    expect(facts.sources).toEqual(["fetch_target_info", "create_ttp", "profile_interactions"]);
  });

  it("ignores failed lookups and finds sequences in arguments", () => {
    const facts = extractTargetFacts([
      { type: "tool_result", name: "fetch_target_info", result: JSON.stringify({ error: "Could not reach UniProt" }) },
      { type: "tool_call", name: "cofold", arguments: { inputs: [{ protein_sequence: SEQ.toLowerCase() }], smiles: ["CCCCCC"] } },
    ]);
    expect(facts.uniprot).toBeNull();
    expect(facts.sequence).toBe(SEQ);
    expect(facts.pocket).toEqual([]);
  });

  it("is empty for a session without a target", () => {
    expect(extractTargetFacts([{ type: "tool_call", name: "list_jobs", arguments: {} }]).sources).toEqual([]);
  });
});
