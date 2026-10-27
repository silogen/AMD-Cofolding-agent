// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import type { InteractionsPanel, Panel, StructurePanel, TurnBlock } from "../types/api";
import type { JobStatus } from "./jobs";
import type { Numbering } from "./numbering";
import {
  answerExcerpt,
  buildReport,
  conversationRuns,
  conversationTtps,
  escapeHtml,
  parseResidues,
  reportMolecules,
  reportStructures,
  reportTurns,
  structureContacts,
  type ReportInput,
} from "./report";
import { emptyTargetFacts, type Ttp } from "./ttp";

type ToolStep = Extract<TurnBlock, { kind: "tool" }>;

let at = 0;
function step(name: string, args: Record<string, unknown>, result: unknown, extra: { panels?: Panel[]; isError?: boolean } = {}): ToolStep {
  at += 1;
  return {
    kind: "tool",
    id: `c${at}`,
    name,
    arguments: args,
    at,
    result: {
      type: "tool_result",
      seq: at,
      ts: "",
      id: `c${at}`,
      name,
      result: typeof result === "string" ? result : JSON.stringify(result),
      truncated: false,
      isError: extra.isError ?? false,
      panels: extra.panels ?? [],
      elapsedMs: 10,
    },
  };
}

const TTP: Ttp = {
  id: "sol",
  name: "Solubility",
  version: 1,
  scoring: {
    aggregation: "geometric_mean",
    objectives: [
      { name: "Aqueous solubility", type: "external", service: "admet", weight: 0.6, transform: { type: "sigmoid", low: -6, high: -3, k: 0.5 } },
      { name: "QED", type: "native", component: "QED", weight: 0.4, transform: null },
    ],
  },
};

const STATUS: JobStatus = {
  id: "3814ad67",
  ttpId: "sol",
  ttpVersion: 1,
  model: ".m2m_medium",
  type: "optimization",
  state: "running",
  createdAt: null,
  startedAt: "2026-10-27T07:24:04Z",
  finishedAt: null,
  error: null,
  maxSteps: 300,
  numEpochs: null,
  csvPath: null,
  checkpointPath: null,
  hasCheckpoint: false,
  workingDir: null,
  currentStep: 280,
  molecules: 17792,
  valid: 17190,
  validity: 0.966,
  meanScore: null,
  bestScore: null,
  latestMean: null,
  components: [{ name: "Score", mean: 0.86, max: 1, min: 0 }],
  trajectory: [],
  componentTrajectories: {
    Score: [
      { step: 1, mean: 0.5 },
      { step: 100, mean: 0.8 },
      { step: 279, mean: 0.9 },
    ],
  },
  diagnostics: null,
};

const PDB_PATH = "predictions/out_1888c8910823/mol_0_model_0.pdb";
const JQ1 = "CC1=C(SC2=C1C(=N[C@H](C3=NN=C(N32)C)CC(=O)OC(C)(C)C)C4=CC=C(C=C4)Cl)C";

const structure: StructurePanel = {
  kind: "structure",
  tool: "score_molecules",
  name: "mol_0_model_0.pdb",
  path: PDB_PATH,
  url: `/api/artifacts/${PDB_PATH}`,
  format: "pdb",
  smiles: JQ1,
  metrics: { confidence: 0.98, ligandIptm: 0.99, affinity: -1.82, posebustersPassed: 18, posebustersTotal: 22 },
};

const contact = (type: string, resname: string, resnum: number) => ({
  type,
  subtype: null,
  chain: "A",
  resname,
  resnum,
  label: `${resname}${resnum}`,
  receptorAtom: null,
  ligandAtom: null,
  ligandAtomIndex: null,
  distance: 3.5,
  fragment: null,
  confidence: "confident",
});

const profile: InteractionsPanel = {
  kind: "interactions",
  tool: "profile_interactions",
  name: "mol_0",
  structurePath: PDB_PATH,
  smiles: JQ1,
  error: null,
  counts: { hydrophobic: 3, ch_pi: 1 },
  contacts: [contact("hydrophobic", "LEU", 51), contact("hydrophobic", "TRP", 38), contact("hydrophobic", "LEU", 51), contact("ch_pi", "LEU", 49)],
  totalContacts: 4,
};

const base = (extra: Partial<ReportInput> = {}): ReportInput => ({
  title: "BRD4 work",
  id: "s-1",
  created: "2026-10-26T21:14:29Z",
  generated: new Date("2026-10-27T08:00:00Z"),
  target: null,
  ttps: [],
  runs: [],
  molecules: [],
  structures: [],
  turns: [],
  ...extra,
});

describe("buildReport", () => {
  it("leaves out every empty section", () => {
    const html = buildReport(base());
    expect(html).toContain("<h1>BRD4 work</h1>");
    expect(html).not.toContain("<section");
  });

  it("has a section per kind of result, each with its inline SVG", () => {
    const target = { ...emptyTargetFacts(), gene: "BRD4", name: "Bromodomain-containing protein 4", uniprot: "O60885", uniprotSequence: "M".repeat(1362), sequence: "Q".repeat(125) };
    const html = buildReport(
      base({
        target,
        ttps: [TTP],
        runs: [STATUS],
        molecules: [{ smiles: JQ1, name: "JQ1", score: 0.9, objectives: [], admet: [{ label: "LogP", value: "2.25", unit: null }], admetNote: null, metrics: null, source: "score_molecules" }],
        structures: reportStructures({ structures: [{ item: structure, tool: "score_molecules", at: 1 }], interactions: [{ item: profile, tool: "profile_interactions", at: 2 }] }, () => null),
        turns: [{ question: "Cofold JQ1", title: "JQ1 in BRD4", answer: "It binds." }],
        depict: () => "<?xml version='1.0'?><svg class='mol'></svg>",
      }),
    );
    for (const id of ["target", "ttp", "runs", "molecules", "structures", "answers"]) expect(html).toContain(`<section id="${id}">`);
    expect(html).toContain("O60885");
    expect(html).toContain("1,362 aa");
    expect(html).toContain("125 aa");
    expect(html).toContain('aria-label="Higher is better"');
    expect(html).toContain('aria-label="Score per step"');
    expect(html).toContain("<svg class='mol'></svg>");
    expect(html).not.toContain("<?xml");
    expect(html).toContain("PoseBusters 18/22");
    expect(html).toContain("Gate failed");
    expect(html).toContain("ranking signal");
    expect(html).toContain("LEU49");
    expect(html).toContain("TRP38 LEU51");
    expect(html).not.toMatch(/https?:\/\/(?!www\.w3\.org)/);
    expect(html).toContain("break-inside:avoid");
  });

  it("does not caution about an affinity whose pose passes the gate", () => {
    const passing = { ...structure, metrics: { confidence: 0.9, ligandIptm: 0.8, affinity: -1, posebustersPassed: 22, posebustersTotal: 22 } };
    const html = buildReport(base({ structures: reportStructures({ structures: [{ item: passing, tool: "t", at: 1 }], interactions: [] }, () => null) }));
    expect(html).toContain("Gate passed");
    expect(html).not.toContain("ranking signal");
  });

  it("escapes everything that came from the conversation", () => {
    const html = buildReport(
      base({
        title: `<script>alert("x")</script> & co`,
        turns: [{ question: "<img src=x onerror=alert(1)>", title: "A & B", answer: "x < y" }],
        molecules: [{ smiles: "C<C", name: `"quoted"`, score: null, objectives: [], admet: [], admetNote: null, metrics: null, source: "t" }],
      }),
    );
    expect(html).not.toContain("<script>");
    expect(html).not.toContain("<img");
    expect(html).toContain("&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; co");
    expect(html).toContain("x &lt; y");
    expect(html).toContain("C&lt;C");
    expect(html).toContain("&quot;quoted&quot;");
    expect(escapeHtml(`'`)).toBe("&#39;");
  });
});

describe("answerExcerpt", () => {
  it("skips headings, tables, code and preambles", () => {
    const answer = [
      "I have all three pieces. Let me compile the analysis.",
      "## Results",
      "| Metric | Value |\n|---|---|\n| Confidence | 0.98 |",
      "**Confidence gate** (all three must pass):",
      "1. confidence > 0.6\n2. ipTM > 0.5",
      "> ⚠️ **The gate is not fully satisfied.** Treat the affinity as a ranking signal.",
      "```\n/shared/x.pdb\n```",
    ].join("\n\n");
    expect(answerExcerpt(answer)).toBe("⚠️ The gate is not fully satisfied. Treat the affinity as a ranking signal.");
  });

  it("keeps what follows an announcement and drops labels and lists run into a paragraph", () => {
    const answer = [
      "Here's what the profile reveals. Note: the ligand sits in the **acetyl-lysine** pocket.",
      "**Ligand:** `CC1S[C@@H]2CCC2`",
      "**Key fragments:**\n- thiophene\n- triazole",
    ].join("\n\n");
    expect(answerExcerpt(answer)).toBe("Note: the ligand sits in the acetyl-lysine pocket.");
  });

  it("prefers a summary section", () => {
    const answer = "Some long intro.\n\n## Summary\n\n- JQ1 binds\n- the pose is fine\n\n## Details\n\nMore.";
    expect(answerExcerpt(answer)).toBe("JQ1 binds; the pose is fine");
  });

  it("cuts long prose at a sentence", () => {
    const text = `${"Word ".repeat(30).trim()}. ${"More words here ".repeat(40)}`;
    const excerpt = answerExcerpt(text, 200);
    expect(excerpt.length).toBeLessThanOrEqual(200);
    expect(excerpt.endsWith(".") || excerpt.endsWith("…")).toBe(true);
  });
});

describe("reading the conversation", () => {
  it("keeps the profiles of calls that worked and names the ones it never saw", () => {
    const steps = [
      step("score_molecules", { ttp: { ...TTP, id: "bad" } }, { error: "invalid TTP" }),
      step("create_ttp", { ttp: TTP }, { id: "sol" }),
      step("run_optimization", { ttp_id: "other" }, { job_id: "abc12345" }),
    ];
    const { ttps, missing } = conversationTtps(steps);
    expect(ttps.map((t) => t.id)).toEqual(["sol"]);
    expect(missing).toEqual(["other"]);
    expect(conversationRuns([...steps, step("get_optimization_status", { job_id: "def67890" }, {})])).toEqual(["abc12345", "def67890"]);
  });

  it("titles each answer and pairs it with its question", () => {
    const turns = reportTurns([
      { role: "user", id: "u1", content: "Predict ADMET for caffeine", at: 1 },
      { role: "agent", id: "a1", at: 2, done: true, blocks: [{ kind: "text", id: "t1", content: "Caffeine is soluble." }] },
      { role: "user", id: "u2", content: "Stop", at: 3 },
      { role: "agent", id: "a2", at: 4, done: true, blocks: [{ kind: "cancelled", id: "x" }] },
    ]);
    expect(turns).toEqual([{ question: "Predict ADMET for caffeine", title: "Answer", answer: "Caffeine is soluble." }]);
  });

  it("lists each run's best, then what was measured, once per molecule", () => {
    const molecules = reportMolecules({
      runs: [
        {
          jobId: "3814ad67",
          candidates: [
            { smiles: "CCO", total: null, components: [{ name: "Score", transformed: 0.5, raw: null }, { name: "Aqueous solubility", transformed: 0.5, raw: -4.4 }, { name: "Agent", transformed: 4, raw: null }] },
            { smiles: "CCN", total: 0.9, components: [] },
          ],
        },
      ],
      artifacts: {
        tables: [],
        steps: [],
        molecules: [
          { item: { kind: "molecule", tool: "predict_admet", smiles: "OCC", label: null }, tool: "predict_admet", at: 2 },
          { item: { kind: "molecule", tool: "fetch_compound", smiles: "C", label: null }, tool: "fetch_compound", at: 1 },
        ],
        structures: [],
        properties: [
          {
            item: {
              kind: "properties",
              tool: "predict_admet",
              service: "admet",
              target: null,
              endpoints: ["logp", "aqueous_solubility"],
              molecules: [
                { smiles: "OCC", standardized: null, label: null, ok: true, error: null, adPass: false, values: { logp: { mean: -0.31, unit: "logP (dimensionless)" }, aqueous_solubility: { mean: 0.5, unit: "log10(mol/L)" } } },
              ],
            },
            tool: "predict_admet",
            at: 2,
          },
        ],
      },
      canonical: (smiles) => (smiles === "OCC" ? "CCO" : smiles),
    });
    expect(molecules.map((m) => m.smiles)).toEqual(["CCN", "CCO"]);
    const ethanol = molecules[1];
    expect(ethanol.score).toBe(0.5);
    expect(ethanol.objectives).toEqual([{ name: "Aqueous solubility", value: -4.4 }]);
    expect(ethanol.admet).toEqual([
      { label: "LogP", value: "-0.31", unit: null },
      { label: "Aq. solubility", value: "0.50", unit: "log10(mol/L)" },
    ]);
    expect(ethanol.admetNote).toMatch(/applicability domain/);
  });

  it("numbers contact residues as in UniProt when the structure maps", () => {
    const numbering: Numbering = { toUniprot: new Map([["A:51", 94], ["A:38", 81], ["A:49", 92]]), toFile: new Map(), names: new Map(), same: false };
    expect(structureContacts(profile, numbering)).toEqual([
      { type: "hydrophobic", count: 3, residues: ["TRP81", "LEU94"] },
      { type: "ch_pi", count: 1, residues: ["LEU92"] },
    ]);
    expect(structureContacts(profile, null)[0].residues).toEqual(["TRP38", "LEU51"]);
  });
});

describe("parseResidues", () => {
  it("reads polymer residues from a PDB file", () => {
    const pdb = [
      "ATOM      1  N   ASN A   1       9.009   1.098 -15.561  1.00 98.96           N",
      "ATOM      2  CA  ASN A   1       8.365   1.414 -14.281  1.00 98.96           C",
      "ATOM      9  N   PRO A   2       7.365   1.414 -14.281  1.00 98.96           N",
      "HETATM  100  C1  LIG B   1       1.000   1.000   1.000  1.00 90.00           C",
    ].join("\n");
    expect(parseResidues(pdb, "pdb")).toEqual([
      { chain: "A", resnum: 1, resname: "ASN" },
      { chain: "A", resnum: 2, resname: "PRO" },
    ]);
  });

  it("reads the author numbering from an mmCIF atom_site loop", () => {
    const cif = [
      "data_x",
      "loop_",
      "_atom_site.group_PDB",
      "_atom_site.id",
      "_atom_site.label_comp_id",
      "_atom_site.label_asym_id",
      "_atom_site.label_seq_id",
      "_atom_site.auth_seq_id",
      "_atom_site.auth_asym_id",
      "ATOM 1 ASN A 1 44 A",
      "ATOM 2 ASN A 1 44 A",
      "ATOM 3 LEU A 2 45 A",
      "HETATM 4 LIG B . 1 B",
      "#",
    ].join("\n");
    expect(parseResidues(cif, "cif")).toEqual([
      { chain: "A", resnum: 44, resname: "ASN" },
      { chain: "A", resnum: 45, resname: "LEU" },
    ]);
  });
});
