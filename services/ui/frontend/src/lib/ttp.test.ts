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
  weightRange,
  objectiveSource,
  ttpCreated,
  ttpGeneration,
  buildTtp,
  createdTtp,
  formProblems,
  nativeObjective,
  normaliseSequence,
  objectiveCatalogue,
  parseComponents,
  parseServices,
  presetObjectives,
  renameObjective,
  sequenceProblem,
  serviceObjective,
  slugify,
  uniqueName,
  type ToolEvent,
  type Ttp,
  type TtpForm,
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

describe("TTP display helpers", () => {
  const make = (weights: number[], extra: Record<string, unknown> = {}): Ttp =>
    ({
      id: "t",
      name: "t",
      version: 1,
      scoring: { objectives: weights.map((weight, i) => ({ name: `o${i}`, type: "native", component: "QED", weight })) },
      ...extra,
    }) as Ttp;

  it("uses a 0-1 weight range unless a weight exceeds 1", () => {
    expect(weightRange(make([0.3, 1]))).toEqual({ min: 0, max: 1, step: 0.01 });
    expect(weightRange(make([0.3, 2.5]))).toEqual({ min: 0, max: 5, step: 0.05 });
  });
  it("labels objectives native, service or external", () => {
    expect(objectiveSource({ type: "native" })).toBe("native");
    expect(objectiveSource({ type: "external", service: "boltz" })).toBe("service");
    expect(objectiveSource({ type: "external", service: "my_api" })).toBe("external");
  });
  it("reads created and generation only when present", () => {
    expect(ttpCreated(make([1]))).toBeNull();
    expect(ttpCreated(make([1], { created_at: "2026-10-26" }))).toBe("2026-10-26");
    expect(ttpGeneration(make([1]))).toEqual([]);
    expect(ttpGeneration(make([1], { generation: { mode: "libinvent", libinvent_input: { scaffold: "c1ccccc1[*]" }, seeds: ["CCO", "CCN"] } }))).toEqual([
      { label: "Mode", value: "libinvent" },
      { label: "Scaffold", value: "c1ccccc1[*]" },
      { label: "Seeds", value: "CCO, CCN" },
    ]);
  });
});

const SERVICES = JSON.stringify({
  boltz: { url: "http://localhost:8091/score", status: "ok" },
  posebusters: { url: "http://localhost:8082/score", status: "ok" },
  admet: { url: "http://localhost:8084/score", status: "ok", endpoints: ["logp", "logd", "caco2_permeability"] },
  retrodmta: { url: "http://localhost:8085/score", status: "ok", targets: ["FXA"], targets_detail: { FXA: { endpoints: ["FXA_pKi", "FXA_logD"] } } },
});

describe("builder inputs", () => {
  it("reads services with their selectors", () => {
    const services = parseServices(SERVICES);
    expect(Object.keys(services)).toEqual(["boltz", "posebusters", "admet", "retrodmta"]);
    expect(services.admet.endpoints).toEqual(["logp", "logd", "caco2_permeability"]);
    expect(services.retrodmta.targets).toEqual({ FXA: ["FXA_pKi", "FXA_logD"] });
    expect(parseServices("not json")).toEqual({});
  });

  it("keeps the native scorers that need no settings", () => {
    const components = parseComponents([
      { name: "qed", class: "Qed", component_type: "scorer", parameters: [] },
      { name: "sascore", class: "SAScore", component_type: "scorer", parameters: null },
      { name: "customalerts", class: "CustomAlerts", component_type: "filter", parameters: [] },
      { name: "fragmentqed", class: "FragmentQed", component_type: "scorer", parameters: [] },
      { name: "tanimotosimilarity", class: "TanimotoSimilarity", component_type: "scorer", parameters: [{ name: "smiles" }] },
    ]);
    expect(components).toEqual(["Qed", "SAScore"]);
  });

  it("slugs names into ids and cleans sequences", () => {
    expect(slugify("BRD4 Cofolding (v2)!")).toBe("brd4-cofolding-v2");
    expect(slugify("  ")).toBe("");
    expect(normaliseSequence(">sp|O60885|BRD4_HUMAN\nmsae sgpg\n10 TRLR*")).toBe("MSAESGPGTRLR");
    expect(sequenceProblem("MSAE")).toMatch(/at least 10/);
    expect(sequenceProblem("MSAEXGPGTRLRB")).toMatch(/B, X/);
    expect(sequenceProblem(SEQ)).toBeNull();
    expect(sequenceProblem("")).toBeNull();
  });
});

describe("objectives", () => {
  const services = parseServices(SERVICES);

  it("uses the configured service URLs and chains PoseBusters after Boltz", () => {
    const [boltz, posebusters] = presetObjectives("cofolding", services);
    expect(boltz).toMatchObject({ type: "external", service: "boltz", service_url: "http://localhost:8091/score" });
    expect(boltz.transform?.type).toBe("reverse_sigmoid");
    expect(posebusters).toMatchObject({ service: "posebusters", depends_on: boltz.name });
  });

  it("selects ADMET endpoints and RetroDMTA targets through service_params", () => {
    expect(serviceObjective(services, "admet", "logd")).toMatchObject({ name: "LogD", service_params: { endpoint: "logd" } });
    expect(serviceObjective(services, "retrodmta", "FXA_pKi", "FXA")).toMatchObject({ service_params: { target: "FXA", endpoint: "FXA_pKi" }, transform: { type: "sigmoid" } });
    expect(nativeObjective("TanimotoSimilarity").params).toMatchObject({ smiles: [], radius: 3 });
    expect(nativeObjective("QED").transform).toBeNull();
  });

  it("offers every service selector and native scorer", () => {
    const keys = objectiveCatalogue(services, ["QED", "Csp3"]).map((e) => e.key);
    expect(keys).toContain("boltz");
    expect(keys).toContain("admet:caco2_permeability");
    expect(keys).toContain("retrodmta:FXA:FXA_logD");
    expect(keys).toContain("native:QED");
    expect(keys).toContain("native:Csp3");
    expect(keys).toContain("native:TanimotoSimilarity");
    expect(keys).not.toContain("native:SlogP");
    expect(keys).not.toContain("ip:hbond");
  });

  it("names and renames without breaking links", () => {
    expect(uniqueName("QED", ["QED", "QED 2"])).toBe("QED 3");
    expect(uniqueName("MW", ["QED"])).toBe("MW");
    const renamed = renameObjective(presetObjectives("cofolding", services), 0, "Affinity");
    expect(renamed[0].name).toBe("Affinity");
    expect(renamed[1].depends_on).toBe("Affinity");
  });
});

describe("formProblems and buildTtp", () => {
  const services = parseServices(SERVICES);
  const form = (patch: Partial<TtpForm> = {}): TtpForm => ({
    name: "BRD4 cofold",
    id: "brd4-cofold",
    description: "",
    sequence: SEQ,
    pdbId: "",
    aggregation: "geometric_mean",
    objectives: presetObjectives("cofolding", services),
    ...patch,
  });

  it("accepts a complete form", () => {
    expect(formProblems(form())).toEqual([]);
  });

  it("explains what the server would reject", () => {
    expect(formProblems(form({ id: "Bad id" }))[0]).toMatch(/lowercase/);
    expect(formProblems(form(), ["brd4-cofold"])[0]).toMatch(/already exists/);
    expect(formProblems(form({ sequence: "" }))).toContain("Boltz scoring needs the target's protein sequence.");
    expect(formProblems(form({ objectives: [] }))).toContain("Add at least one objective.");
    const objectives = presetObjectives("cofolding", services);
    expect(formProblems(form({ objectives: objectives.slice(1) })).join(" ")).toMatch(/runs after "Boltz affinity"/);
    expect(formProblems(form({ objectives: [objectives[2], { ...objectives[2] }] })).join(" ")).toMatch(/unique: QED/);
    expect(formProblems(form({ objectives: [{ ...objectives[2], weight: 0 }] })).join(" ")).toMatch(/weight above 0/);
    expect(formProblems(form({ objectives: [nativeObjective("TanimotoSimilarity")] })).join(" ")).toMatch(/reference SMILES/);
  });

  it("builds a create_ttp argument without empty fields", () => {
    const ttp = buildTtp(form({ description: " For the demo ", pdbId: "3mxf", sequence: `>x\n${SEQ.toLowerCase()}` }));
    expect(ttp).toMatchObject({
      id: "brd4-cofold",
      name: "BRD4 cofold",
      target: { protein_sequence: SEQ, pdb_id: "3MXF" },
      scoring: { comment: "For the demo", aggregation: "geometric_mean" },
    });
    const [boltz, posebusters, qed] = (ttp.scoring as { objectives: Record<string, unknown>[] }).objectives;
    expect(boltz).toEqual({
      name: "Boltz affinity",
      type: "external",
      weight: 1,
      service: "boltz",
      service_url: "http://localhost:8091/score",
      transform: { type: "reverse_sigmoid", low: -3, high: 1, k: 1 },
      comment: "log10 IC50 in µM; lower binds tighter",
    });
    expect(posebusters.depends_on).toBe("Boltz affinity");
    expect(qed).toEqual({ name: "QED", type: "native", weight: 0.6, component: "QED" });
    expect(buildTtp(form({ sequence: "", objectives: [qed as never] })).target).toBeUndefined();
  });

  it("reads create_ttp's answer", () => {
    expect(createdTtp(JSON.stringify({ status: "created", id: "x", name: "X", version: 1 }))).toEqual({ id: "x", name: "X" });
    expect(createdTtp({ error: "Invalid TTP: boom" })).toEqual({ error: "Invalid TTP: boom" });
    expect(createdTtp("Tool failed")).toEqual({ error: "Tool failed" });
  });
});

