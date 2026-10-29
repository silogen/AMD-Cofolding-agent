// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import type { InteractionsPanel } from "../types/api";
import { extractTargets, orderTargets, sameTarget, scopedTarget, targetKey, targetName, withLookup } from "./targets";
import { emptyTargetFacts, structureHint, targetLigands, type TargetFacts, type ToolEvent, type Ttp } from "./ttp";

const BRD4_SEQ =
  "MSAESGPGTRLRNLPVMGDGLETSQMSTTQAQAQPQPANAASTNPPPPETSNPNKPKRQTNQLQYLLRVVLKTLWKHQFAWPFQQPVDAVKLNLPDYYKIIKTPMDMGTIKKRLENNYYWNAQECIQDFNTMFTNCYIYNKPGDDIVLMAEALEKLFLQKI";
/** The bromodomain actually crystallised, a slice of the canonical sequence. */
const BRD4_BD1 = BRD4_SEQ.slice(43, 160);
const EGFR_SEQ = "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNY";

const BRD4 = {
  accession: "O60885",
  entry_name: "BRD4_HUMAN",
  protein_name: "Bromodomain-containing protein 4",
  gene: "BRD4",
  organism: "Homo sapiens",
  reviewed: true,
  function: "Chromatin reader that recognises acetylated histones.",
  sequence: BRD4_SEQ,
  sequence_length: BRD4_SEQ.length,
  pdb_ids: ["3MXF", "2OSS"],
  pdb_count: 400,
  pdb_structures: [
    { pdb_id: "3MXF", method: "X-ray", resolution: 1.85, uniprot_range: [44, 168] },
    { pdb_id: "2OSS", method: "X-ray", resolution: 2.3, uniprot_range: [349, 460] },
  ],
  constructs: [
    {
      pdb_id: "3MXF",
      title: "BRD4 bromodomain 1 with JQ1",
      method: "X-RAY DIFFRACTION",
      resolution: 1.85,
      chains: ["A", "B"],
      length: BRD4_BD1.length,
      uniprot_range: [44, 168],
      ligands: ["JQ1 ((S)-tert-butyl triazolodiazepine)", "SO4 (SULFATE ION)"],
    },
  ],
};

const EGFR = {
  accession: "P00533",
  entry_name: "EGFR_HUMAN",
  protein_name: "Epidermal growth factor receptor",
  gene: "EGFR",
  organism: "Homo sapiens",
  reviewed: true,
  sequence: EGFR_SEQ,
  sequence_length: 1210,
  pdb_ids: ["1M17"],
  pdb_count: 200,
};

const ttp = (id: string, sequence: string, pdbId?: string): Ttp => ({
  id,
  name: id,
  version: 1,
  target: { protein_sequence: sequence, ...(pdbId ? { pdb_id: pdbId } : {}) },
  scoring: { objectives: [{ name: "QED", type: "native", component: "QED", weight: 1 }] },
});

const contacts: InteractionsPanel = {
  kind: "interactions",
  tool: "profile_interactions",
  name: null,
  structurePath: "out/x.cif",
  smiles: "CCO",
  error: null,
  counts: {},
  totalContacts: 2,
  contacts: [
    { type: "hbond", subtype: null, chain: "A", resname: "ASN", resnum: 140, label: "ASN140", receptorAtom: null, ligandAtom: null, ligandAtomIndex: null, distance: 2.9, fragment: null, confidence: null },
    { type: "hydrophobic", subtype: null, chain: "A", resname: "TYR", resnum: 97, label: "TYR97", receptorAtom: null, ligandAtom: null, ligandAtomIndex: null, distance: 3.6, fragment: null, confidence: null },
  ],
};

/** A conversation that looked BRD4 up, profiled it, then turned to EGFR. */
const events: ToolEvent[] = [
  { type: "tool_call", name: "fetch_target_info", arguments: { query: "O60885" } },
  { type: "tool_result", name: "fetch_target_info", result: JSON.stringify(BRD4) },
  { type: "tool_call", name: "create_ttp", arguments: { ttp: ttp("brd4-lead", BRD4_BD1, "3MXF") } },
  { type: "tool_result", name: "profile_interactions", result: "{}", panels: [contacts] },
  { type: "tool_call", name: "fetch_target_info", arguments: { query: "EGFR human" } },
  { type: "tool_result", name: "fetch_target_info", result: JSON.stringify(EGFR) },
];

describe("extractTargets", () => {
  it("keeps the conversation's proteins apart", () => {
    const targets = extractTargets(events);
    expect(targets.map((t) => t.key)).toEqual(["O60885", "P00533"]);
    expect(targets.map((t) => targetName(t.facts))).toEqual(["BRD4", "EGFR"]);
    expect(targets.every((t) => t.inSession)).toBe(true);
  });

  it("joins a lookup, its profile and its contacts into one target", () => {
    const [brd4] = extractTargets(events);
    expect(brd4.facts).toMatchObject({
      gene: "BRD4",
      uniprot: "O60885",
      organism: "Homo sapiens",
      // The profile folds a construct, so that is the sequence in play; the canonical one stays for numbering.
      sequence: BRD4_BD1,
      sequenceLength: BRD4_BD1.length,
      uniprotSequence: BRD4_SEQ,
      ttpId: "brd4-lead",
      pdbIds: ["3MXF", "2OSS"],
      profiledComplexes: 1,
      chains: ["A"],
    });
    expect(brd4.facts.pocket.map((r) => r.label)).toEqual(["TYR97", "ASN140"]);
    expect(extractTargets(events)[1].facts.pocket).toEqual([]);
  });

  it("reads the names, function and cross-references of the same lookup", () => {
    const { facts } = extractTargets(events)[0];
    expect(facts.name).toBe("Bromodomain-containing protein 4");
    expect(facts.entryName).toBe("BRD4_HUMAN");
    expect(facts.function).toMatch(/Chromatin reader/);
    expect(facts.structures.map((s) => s.pdbId)).toEqual(["3MXF", "2OSS"]);
    expect(facts.constructs[0]).toMatchObject({ pdbId: "3MXF", chains: ["A", "B"], length: BRD4_BD1.length, range: [44, 168] });
    expect(structureHint(facts, "3mxf")).toBe("X-RAY DIFFRACTION 1.85 Å · UniProt 44–168 · JQ1");
    expect(structureHint(facts, "2OSS")).toBe("X-ray 2.3 Å · UniProt 349–460");
    expect(structureHint(facts, "9ZZZ")).toBeNull();
    expect(targetLigands(facts)).toEqual([{ code: "JQ1", name: "(S)-tert-butyl triazolodiazepine", pdbIds: ["3MXF"] }]);
  });

  it("ignores a failed lookup and the accession it asked about", () => {
    const failed: ToolEvent[] = [
      { type: "tool_call", name: "fetch_target_info", arguments: { query: "O60885" } },
      { type: "tool_result", name: "fetch_target_info", result: JSON.stringify({ error: "Could not reach UniProt" }), isError: false },
      { type: "tool_call", name: "cofold", arguments: { inputs: [{ protein_sequence: BRD4_BD1 }], smiles: ["CCO"] } },
    ];
    const targets = extractTargets(failed);
    expect(targets).toHaveLength(2);
    expect(targets.map((t) => t.key)).toEqual(["O60885", `seq:${BRD4_BD1.length}:${BRD4_BD1.slice(0, 8)}`]);
  });

  it("is empty for a conversation without a target", () => {
    expect(extractTargets([{ type: "tool_call", name: "list_jobs", arguments: {} }])).toEqual([]);
  });
});

describe("choosing a target", () => {
  const targets = extractTargets(events);

  it("puts the conversation's own first, the latest leading", () => {
    const withOther = withLookup(targets, JSON.stringify({ accession: "P24941", gene: "CDK2", sequence: "MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRL", sequence_length: 298 }));
    expect(withOther.map((t) => t.inSession)).toEqual([true, true, false]);
    expect(orderTargets(withOther).map((t) => targetName(t.facts))).toEqual(["EGFR", "BRD4", "CDK2"]);
  });

  it("merges a lookup of a target the conversation already has", () => {
    const again = withLookup(targets, JSON.stringify({ ...BRD4, function: "Updated summary." }));
    expect(again).toHaveLength(2);
    const brd4 = again.find((t) => t.key === "O60885");
    expect(brd4?.inSession).toBe(true);
    expect(brd4?.facts.function).toBe("Updated summary.");
    expect(brd4?.facts.ttpId).toBe("brd4-lead");
  });

  it("ignores a lookup that answered with nothing", () => {
    expect(withLookup(targets, JSON.stringify({ error: "No UniProt entry found" }))).toHaveLength(2);
  });

  it("defaults to the target the current profile uses, and follows a pick", () => {
    expect(scopedTarget(targets, null, "brd4-lead")?.key).toBe("O60885");
    expect(scopedTarget(targets, null, null)?.key).toBe("P00533");
    expect(scopedTarget(targets, "P00533", "brd4-lead")?.key).toBe("P00533");
    expect(scopedTarget(targets, "gone", null)?.key).toBe("P00533");
    expect(scopedTarget([], null, "brd4-lead")).toBeNull();
  });
});

describe("target identity", () => {
  const facts = (patch: Partial<TargetFacts>): TargetFacts => ({ ...emptyTargetFacts(), ...patch });

  it("never merges two known accessions", () => {
    expect(sameTarget(facts({ uniprot: "O60885", sequence: BRD4_SEQ }), facts({ uniprot: "P00533", sequence: BRD4_SEQ }))).toBe(false);
  });

  it("recognises a construct, a shared entry and a shared profile", () => {
    expect(sameTarget(facts({ sequence: BRD4_SEQ }), facts({ sequence: BRD4_BD1 }))).toBe(true);
    expect(sameTarget(facts({ sequence: BRD4_SEQ }), facts({ sequence: EGFR_SEQ }))).toBe(false);
    expect(sameTarget(facts({ pdbIds: ["3MXF", "2OSS"] }), facts({ pdbIds: ["2OSS"] }))).toBe(true);
    expect(sameTarget(facts({ ttpId: "brd4-lead" }), facts({ ttpId: "brd4-lead" }))).toBe(true);
    expect(sameTarget(facts({}), facts({}))).toBe(false);
  });

  it("names and keys a target by what is known of it", () => {
    expect(targetKey(facts({ uniprot: "O60885", pdbIds: ["3MXF"] }))).toBe("O60885");
    expect(targetKey(facts({ pdbIds: ["3MXF"] }))).toBe("3MXF");
    expect(targetKey(facts({ ttpId: "brd4-lead" }))).toBe("ttp:brd4-lead");
    expect(targetKey(facts({}))).toBe("target");
    expect(targetName(facts({ name: "Bromodomain-containing protein 4" }))).toBe("Bromodomain-containing protein 4");
    expect(targetName(facts({ sequenceLength: 130 }))).toBe("130 aa");
    expect(targetName(facts({}))).toBe("Target");
  });
});
