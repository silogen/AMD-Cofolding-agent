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
  objectiveKey,
  objectiveGroups,
  defaultObjectives,
  groupColour,
  objectiveGroup,
  toggleObjective,
  dropObjectives,
  parseComponents,
  parseServices,
  presetObjectives,
  renameObjective,
  sequenceProblem,
  serviceObjective,
  slugify,
  uniqueName,
  advancedParams,
  chatPrompt,
  createdTimes,
  curveDomain,
  curveLabel,
  defaultTransform,
  filterTtps,
  formFromTtp,
  hasCofolding,
  rangeFields,
  rememberCreated,
  retypeObjective,
  setRange,
  sortTtps,
  targetSummary,
  ttpFileName,
  withCofolding,
  versionDiff,
  isSameVersion,
  type ToolEvent,
  type Objective,
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
      { name: "tpsa", class: "TPSA", component_type: "scorer", parameters: [{ name: "includeSandP", required: false }] },
    ]);
    expect(components).toEqual(["Qed", "SAScore", "TPSA"]);
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

  it("offers every service selector and native scorer, keyed as objectiveKey", () => {
    const catalogue = objectiveCatalogue(services, ["QED", "Csp3"]);
    const keys = catalogue.map((e) => e.key);
    expect(keys).toEqual(expect.arrayContaining(["boltz:", "posebusters:", "admet:caco2_permeability", "retrodmta:FXA:FXA_logD", "native:QED", "native:Csp3", "native:TanimotoSimilarity"]));
    expect(keys).toContain("native:SlogP");
    expect(keys).not.toContain("native:HBondDonors");
    expect(keys).not.toContain("interaction_profiler:hbond");
    for (const e of catalogue) expect(objectiveKey(e.make())).toBe(e.key);
    expect(catalogue.find((e) => e.key === "native:Csp3")).toMatchObject({ extra: true });
    expect(catalogue.find((e) => e.key === "posebusters:")).toMatchObject({ after: "boltz:" });
  });

  it("matches REINVENT's class names to the usual metrics, whatever their case", () => {
    const keys = objectiveCatalogue(services, ["Qed", "SlogP", "Csp3"]).filter((e) => e.group === "Drug-likeness").map((e) => e.key);
    expect(keys).toEqual(["native:QED", "native:MolecularWeight", "native:SlogP", "native:TPSA", "native:SAScore", "native:TanimotoSimilarity", "native:Csp3"]);
  });

  it("labels the catalogue by metric, grouped Cofold, Drug-likeness, ADMET, RetroDMTA", () => {
    const catalogue = objectiveCatalogue(services, ["QED", "MolecularWeight"]);
    const labels = catalogue.map((e) => `${e.group}/${e.label}`);
    expect(labels).toContain("Drug-likeness/Molecular weight");
    expect(labels).toContain("Cofold/Binding affinity");
    expect(labels.join(" ")).not.toMatch(/RDKit|PoseBusters|Boltz|\(MolecularWeight\)/);
    expect(objectiveGroups(catalogue).map((g) => g.name)).toEqual(["Cofold", "Drug-likeness", "ADMET", "RetroDMTA"]);
    expect(objectiveCatalogue({ admet: services.admet }, []).some((e) => e.group === "Cofold")).toBe(false);
  });

  it("starts each metric at its preset weight", () => {
    const groups = objectiveGroups(objectiveCatalogue(services, []));
    const druglike = groups.find((g) => g.name === "Drug-likeness")!;
    const preset = new Set(presetObjectives("druglike", services).map(objectiveKey));
    const made = druglike.entries.filter((e) => preset.has(e.key)).map((e) => e.make());
    expect(made).toEqual(presetObjectives("druglike", services));
    expect(druglike.entries.find((e) => e.key === "native:HBondDonors")?.make().weight).toBe(0.5);
  });

  it("starts a new profile with cofolding, pose checks, drug-likeness and key ADMET endpoints the catalogue offers", () => {
    const list = defaultObjectives(objectiveCatalogue(services, []));
    expect(list.map(objectiveKey)).toEqual(["boltz:", "posebusters:", "native:QED", "native:SAScore", "admet:logd"]);
    expect(list[1].depends_on).toBe(list[0].name);
    expect(list[0].service_url).toBe("http://localhost:8091/score");
    expect(list.map((o) => o.weight)).toEqual([1, 0.6, 1, 0.6, 0.8]);
    expect(defaultObjectives(objectiveCatalogue({}, [])).map(objectiveKey)).toContain("admet:aqueous_solubility");
    expect(defaultObjectives(objectiveCatalogue({ admet: services.admet }, [])).map(objectiveKey)).toEqual(["native:QED", "native:SAScore", "admet:logd"]);
  });

  it("gives each picker group its own theme colour", () => {
    const catalogue = objectiveCatalogue(services, []);
    for (const e of catalogue) expect(objectiveGroup(e.make())).toBe(e.group);
    const colours = objectiveGroups(catalogue).map((g) => groupColour(g.name));
    expect(new Set(colours).size).toBe(4);
    for (const c of colours) expect(c).toMatch(/^var\(--color-/);
    expect(groupColour(objectiveGroup({ type: "external", service: "my_api" }))).toBe("var(--text-faint)");
  });

  describe("picking", () => {
    const groups = objectiveGroups(objectiveCatalogue(services, []));
    const pick = (key: string) => groups.flatMap((g) => g.entries).find((e) => e.key === key)!;
    const names = (list: Objective[]) => list.map((o) => o.name);

    it("adds or removes exactly the metric", () => {
      const start = presetObjectives("druglike", services);
      const more = toggleObjective(start, pick("admet:logd"), true);
      expect(more.slice(0, start.length)).toEqual(start);
      expect(names(more)).toEqual([...names(start), "LogD"]);
      expect(toggleObjective(more, pick("admet:logd"), true)).toBe(more);
      expect(toggleObjective(more, pick("native:TPSA"), false)).toEqual(more.filter((o) => o.name !== "TPSA"));
      expect(toggleObjective(start, pick("admet:logp"), false)).toBe(start);
    });

    it("finds a metric whatever it is named", () => {
      const renamed: Objective = { ...nativeObjective("QED"), name: "Renamed" };
      expect(objectiveKey(renamed)).toBe("native:QED");
      expect(toggleObjective([renamed], pick("native:QED"), false)).toEqual([]);
      expect(objectiveKey(serviceObjective(services, "admet", "logd"))).toBe("admet:logd");
    });

    it("adds pose checks only after Boltz, and drops them with it", () => {
      expect(toggleObjective([], pick("posebusters:"), true)).toEqual([]);
      const boltz = toggleObjective([], pick("boltz:"), true);
      const both = toggleObjective(boltz, pick("posebusters:"), true);
      expect(both.map((o) => [o.service, o.depends_on ?? null])).toEqual([
        ["boltz", null],
        ["posebusters", "Boltz affinity"],
      ]);
      expect(toggleObjective(both, pick("boltz:"), false)).toEqual([]);
      const renamed = renameObjective(boltz, 0, "Affinity");
      expect(toggleObjective(renamed, pick("posebusters:"), true)[1].depends_on).toBe("Affinity");
    });

    it("drops what runs after a removed objective", () => {
      const list = [...toggleObjective(toggleObjective([], pick("boltz:"), true), pick("posebusters:"), true), nativeObjective("QED")];
      expect(names(list)).toEqual(["Boltz affinity", "Pose quality", "QED"]);
      expect(names(dropObjectives(list, (o) => o.service === "boltz"))).toEqual(["QED"]);
      expect(dropObjectives(list, () => false)).toBe(list);
    });
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
    expect(formProblems(form({ sequence: "" }))).toContain("Cofolding needs the protein: a structure or a sequence.");
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

  it("writes the MSA and a carried-over structure path into the target", () => {
    expect(buildTtp(form({ msaPath: " msas/brd4.a3m " })).target).toEqual({ protein_sequence: SEQ, msa_path: "msas/brd4.a3m" });
    expect(buildTtp(form({ proteinPdbPath: "/shared/p.pdb" })).target).toEqual({ protein_sequence: SEQ, protein_pdb_path: "/shared/p.pdb" });
    expect(buildTtp(form({ msaPath: "", proteinPdbPath: null })).target).toEqual({ protein_sequence: SEQ });
    expect(formProblems(form({ sequence: "", objectives: [nativeObjective("QED")], msaPath: "msas/brd4.a3m" }))).toEqual(["An MSA needs the protein it was built for."]);
  });

  it("sends a structure's chain instead of the sequence", () => {
    const structured = form({ sequence: "not sent", structurePath: " uploads/s-1/1M17.cif ", structureChain: "A", pdbId: "1m17", msaPath: "msas/egfr.a3m" });
    expect(formProblems(structured)).toEqual([]);
    expect(buildTtp(structured).target).toEqual({ structure_path: "uploads/s-1/1M17.cif", structure_chain: "A", pdb_id: "1M17", msa_path: "msas/egfr.a3m" });
    expect(buildTtp(form({ sequence: "", structurePath: "x.pdb", structureChain: null })).target).toEqual({ structure_path: "x.pdb" });
    expect(formProblems(form({ sequence: "", structurePath: "msas/x.a3m" }))).toEqual(["x.a3m is not a .cif or .pdb file."]);
    expect(buildTtp(form({ structurePath: "" })).target).toEqual({ protein_sequence: SEQ });
  });

  it("writes templates into the target, leaving out empty fields", () => {
    const templates = [
      { path: " uploads/s-1/1M17_A.cif ", chain_id: "A", force: false, threshold: null },
      { path: "structures/egfr.pdb", chain_id: null, force: true, threshold: 2 },
    ];
    expect(buildTtp(form({ pdbId: "1m17", templates })).target).toEqual({
      protein_sequence: SEQ,
      pdb_id: "1M17",
      templates: [{ path: "uploads/s-1/1M17_A.cif", chain_id: "A" }, { path: "structures/egfr.pdb", force: true, threshold: 2 }],
    });
    expect(buildTtp(form({ templates: [] })).target).toEqual({ protein_sequence: SEQ });
    expect(formProblems(form({ templates }))).toEqual([]);
  });

  it("explains templates the server would reject", () => {
    expect(formProblems(form({ templates: [{ path: "a.cif", force: true }] }))).toEqual(["Forcing the template a.cif needs a threshold in Å."]);
    expect(formProblems(form({ templates: [{ path: "a.cif", force: true, threshold: Number("") || NaN }] }))).toEqual(["The template a.cif needs a threshold above 0 Å."]);
    expect(formProblems(form({ templates: [{ path: "msas/a.a3m" }] }))).toEqual(["The template a.a3m is not a .cif or .pdb file."]);
  });

  it("keeps constraints when given", () => {
    expect(buildTtp(form({ constraints: [{ type: "custom_alerts", smarts: ["[N+](=O)[O-]"] }] })).constraints).toEqual([{ type: "custom_alerts", smarts: ["[N+](=O)[O-]"] }]);
    expect(buildTtp(form({ constraints: [] })).constraints).toBeUndefined();
  });
});

describe("curve editing", () => {
  const services = parseServices(SERVICES);

  it("labels curves and splits simple from advanced parameters", () => {
    expect(curveLabel("reverse_sigmoid")).toBe("Lower is better");
    expect(curveLabel(null)).toBe("Score as is");
    expect(rangeFields("sigmoid").map((f) => f.key)).toEqual(["low", "high"]);
    expect(rangeFields("right_step")).toEqual([{ key: "low", label: "at least" }]);
    expect(advancedParams("sigmoid")).toEqual(["k"]);
    expect(advancedParams("double_sigmoid")).toEqual(["coef_div", "coef_si", "coef_se"]);
    expect(advancedParams("exponential_decay")).toEqual([]);
  });

  it("keeps a double sigmoid finite far from zero", () => {
    const t = { type: "double_sigmoid", low: 200, high: 500, coef_div: 5, coef_si: 10, coef_se: 10 };
    expect(evaluateTransform(t, 350)).toBeCloseTo(1);
    expect(evaluateTransform(t, 900)).toBe(0);
    expect(transformCurve(t).every((p) => Number.isFinite(p.y))).toBe(true);
  });

  it("samples a fixed domain while a threshold is dragged", () => {
    const t = { type: "sigmoid", low: 0, high: 10, k: 0.5 };
    expect(curveDomain(t)).toEqual({ from: -6, to: 16 });
    const points = transformCurve({ ...t, high: 20 }, 4, { from: -6, to: 16 });
    expect(points[0].x).toBe(-6);
    expect(points[4].x).toBe(16);
  });

  it("starts each objective from its usual window", () => {
    expect(defaultTransform(nativeObjective("MolecularWeight"))).toMatchObject({ low: 200, high: 500 });
    expect(defaultTransform(nativeObjective("QED"))).toMatchObject({ type: "sigmoid", low: 0.4, high: 0.8 });
    expect(defaultTransform(serviceObjective(services, "admet", "aqueous_solubility"))).toMatchObject({ low: -6, high: -3 });
    expect(defaultTransform(serviceObjective(services, "boltz"))).toMatchObject({ type: "reverse_sigmoid", low: -3, high: 1 });
    expect(defaultTransform(nativeObjective("Csp3"))).toMatchObject({ low: 0.2, high: 0.6 });
  });

  it("changes curve type without losing the window", () => {
    const qed = nativeObjective("QED");
    expect(retypeObjective(qed, "double_sigmoid")).toMatchObject({ type: "double_sigmoid", low: 0.4, high: 0.8, coef_div: 0.8 });
    expect(retypeObjective(qed, null)).toBeNull();
    const mw = nativeObjective("MolecularWeight");
    expect(retypeObjective(mw, "right_step")).toEqual({ type: "right_step", low: 200 });
  });

  it("keeps a double sigmoid's coef_div in scale with its window", () => {
    const t = { type: "double_sigmoid", low: 1, high: 5, coef_div: 5, coef_si: 10, coef_se: 10 };
    expect(setRange(t, { high: 8 }).coef_div).toBe(8);
    expect(setRange({ ...t, coef_div: 3 }, { high: 8 }).coef_div).toBe(3);
    expect(setRange({ type: "sigmoid", low: 0, high: 1, k: 0.5 }, { low: 0.2 })).toEqual({ type: "sigmoid", low: 0.2, high: 1, k: 0.5 });
  });

  it("switches cofolding on and off", () => {
    const base = presetObjectives("druglike", services);
    const on = withCofolding(base, services, true);
    expect(hasCofolding(on)).toBe(true);
    expect(on.slice(0, 2).map((o) => [o.service, o.depends_on ?? null])).toEqual([
      ["boltz", null],
      ["posebusters", "Boltz affinity"],
    ]);
    expect(on[0].service_url).toBe("http://localhost:8091/score");
    expect(withCofolding(on, services, true)).toBe(on);
    const withContacts = [...on, serviceObjective(services, "interaction_profiler", "hbond")];
    expect(withCofolding(withContacts, services, false)).toEqual(base);
  });
});

describe("browsing stored TTPs", () => {
  const items = [
    { id: "a-old", name: "Alpha", version: 1 },
    { id: "b-new", name: "Beta", version: 2 },
    { id: "c-none", name: "Gamma solubility", version: 1 },
  ];

  it("searches name and id by every word", () => {
    expect(filterTtps(items, "solub gam").map((t) => t.id)).toEqual(["c-none"]);
    expect(filterTtps(items, "B-NEW").map((t) => t.id)).toEqual(["b-new"]);
    expect(filterTtps(items, "  ")).toHaveLength(3);
  });

  it("puts the most recently used first, then by name", () => {
    expect(sortTtps(items, { "a-old": "2026-10-01T00:00:00Z", "b-new": "2026-10-27T00:00:00Z" }).map((t) => t.id)).toEqual(["b-new", "a-old", "c-none"]);
  });

  it("remembers local creations", () => {
    const store = memory();
    rememberCreated("x", "2026-10-27T08:00:00Z", store);
    rememberCreated("y", "2026-10-27T09:00:00Z", store);
    expect(createdTimes(store)).toEqual({ x: "2026-10-27T08:00:00Z", y: "2026-10-27T09:00:00Z" });
    store.setItem("cofold.ttp-created", "[1]");
    expect(createdTimes(store)).toEqual({});
  });

  it("duplicates a stored TTP into a builder form", () => {
    const copy = formFromTtp(ttp, ["brd4-lead-copy"]);
    expect(copy).toMatchObject({ name: "BRD4 lead optimisation (copy)", id: "brd4-lead-copy-2", sequence: SEQ, pdbId: "3MXF", aggregation: "geometric_mean" });
    expect(copy.objectives.map((o) => o.name)).toEqual(["QED", "MW", "Affinity"]);
    expect(copy.constraints).toEqual(ttp.constraints);
    expect(formProblems(copy)).toEqual([]);
    const withFiles = formFromTtp({ ...ttp, target: { protein_sequence: SEQ, msa_path: "msas/a.a3m", protein_pdb_path: "/shared/p.pdb" } });
    expect(buildTtp(withFiles).target).toEqual({ protein_sequence: SEQ, msa_path: "msas/a.a3m", protein_pdb_path: "/shared/p.pdb" });
    const templated = formFromTtp({ ...ttp, target: { protein_sequence: SEQ, templates: [{ path: "uploads/s/1M17_A.cif", chain_id: "A", force: true, threshold: 1.5 }] } });
    expect(buildTtp(templated).target).toEqual({ protein_sequence: SEQ, templates: [{ path: "uploads/s/1M17_A.cif", chain_id: "A", force: true, threshold: 1.5 }] });
    expect(targetSummary({ ...ttp, target: { protein_sequence: SEQ, templates: [{ path: "a.cif" }] } })).toMatch(/· template$/);
    const fromStructure = formFromTtp({ ...ttp, target: { structure_path: "uploads/s/1M17.cif", structure_chain: "B", pdb_id: "1M17" } });
    expect(fromStructure).toMatchObject({ sequence: "", structurePath: "uploads/s/1M17.cif", structureChain: "B" });
    expect(buildTtp(fromStructure).target).toEqual({ structure_path: "uploads/s/1M17.cif", structure_chain: "B", pdb_id: "1M17" });
    expect(targetSummary({ ...ttp, target: { structure_path: "uploads/s/1M17.cif", structure_chain: "B", msa_path: "m.a3m" } })).toBe("1M17.cif chain B · MSA");
  });

  it("diffs a target read from a structure", () => {
    const structured: Ttp = { ...ttp, target: { structure_path: "uploads/s/1M17.cif", structure_chain: "A" } };
    expect(versionDiff(ttp, structured).other).toEqual(["target from 1M17.cif chain A", "PDB 3MXF → —"]);
    expect(versionDiff(structured, { ...ttp, target: null }).other).toEqual(["target removed"]);
    expect(versionDiff({ ...ttp, target: null }, ttp).other).toEqual([`target added (${SEQ.length} aa)`, "PDB — → 3MXF"]);
  });

  it("diffs two versions", () => {
    expect(isSameVersion(versionDiff(ttp, ttp))).toBe(true);
    const next: Ttp = {
      ...removeObjective(ttp, "pre_scoring", 0),
      target: { protein_sequence: SEQ + "AAAA", pdb_id: "3MXF" },
      scoring: { ...ttp.scoring, aggregation: "arithmetic_mean", objectives: [...ttp.scoring.objectives.slice(0, 1), { ...ttp.scoring.objectives[1], weight: 1.5, transform: { type: "double_sigmoid", low: -3, high: 1, coef_div: 3, coef_si: 10, coef_se: 10 } }, nativeObjective("SAScore")] },
    };
    const diff = versionDiff(ttp, next);
    expect(diff.added).toEqual(["Synthetic accessibility"]);
    expect(diff.removed).toEqual(["QED"]);
    expect(diff.changed).toEqual([{ name: "Affinity", parts: ["weight 2 → 1.5", "curve Lower is better → Within a range"] }]);
    expect(diff.other).toEqual([`target sequence ${SEQ.length} → ${SEQ.length + 4} aa`, "geometric mean → arithmetic mean"]);
    const moved = versionDiff(ttp, patchObjective(ttp, "scoring", 0, { transform: { ...ttp.scoring.objectives[0].transform!, high: 550 } }));
    expect(moved.changed).toEqual([{ name: "MW", parts: ["high 500 → 550"] }]);
  });

  it("describes a TTP for reading and reuse", () => {
    expect(targetSummary(ttp)).toBe(`${SEQ.length} aa · PDB 3MXF`);
    expect(targetSummary({ ...ttp, target: null })).toBeNull();
    expect(chatPrompt(ttp)).toBe('Use the TTP "BRD4 lead optimisation" (brd4-lead)');
    expect(ttpFileName(ttp)).toBe("brd4-lead-v2.json");
  });
});

