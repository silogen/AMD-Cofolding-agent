// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { InteractionsPanel, StructureMetrics, Turn, TurnBlock } from "../types/api";
import { answerTitle, withoutLeadingHeading } from "./answer";
import { contactsFor, runOf, structureTitle, type Artifacts } from "./artifacts";
import { buildCandidates } from "./candidates";
import { contactColor, contactLabel, sortedTypes } from "./contacts";
import { affinityConcentration, affinityText, gate } from "./gate";
import { jobTitle, linear, linePath, parseStatus, runningBest, scoreDomain, ticks, typeLabel, type Candidate as RunCandidate, type JobStatus } from "./jobs";
import { displayNumber, residueLabel, type FileResidue, type Numbering } from "./numbering";
import { endpointLabel, formatValue } from "./properties";
import { conversationJobIds } from "./runProgress";
import {
  asTtp,
  curveDomain,
  curveLabel,
  objectivesOf,
  parseResult,
  payloadError,
  rangeFields,
  sourceLabel,
  targetRange,
  transformCurve,
  ttpGeneration,
  weightShares,
  type Objective,
  type TargetFacts,
  type Transform,
  type Ttp,
} from "./ttp";

/**
 * A conversation as one self-contained HTML page: the target, the profiles,
 * the runs, the best molecules and complexes, and what the agent concluded.
 * Everything is inline (CSS, SVG, depictions), so the file opens offline and
 * prints to PDF one section per page.
 */

type ToolStep = Extract<TurnBlock, { kind: "tool" }>;

export interface ReportMolecule {
  smiles: string;
  name: string | null;
  score: number | null;
  /** Objective values from a run, raw where the run reports them. */
  objectives: { name: string; value: number }[];
  admet: { label: string; value: string; unit: string | null }[];
  admetNote: string | null;
  metrics: StructureMetrics | null;
  /** Where it came from: "run 3814ad67", "score_molecules". */
  source: string;
}

export interface ReportContacts {
  type: string;
  count: number;
  /** "TRP81", in UniProt numbering when the structure could be mapped. */
  residues: string[];
}

export interface ReportStructure {
  title: string;
  run: string | null;
  smiles: string | null;
  metrics: StructureMetrics | null;
  contacts: ReportContacts[];
  numbering: "uniprot" | "file";
}

export interface ReportTurn {
  question: string;
  title: string;
  answer: string;
}

export interface ReportInput {
  title: string;
  id: string;
  /** ISO 8601. */
  created: string;
  generated: Date;
  target: TargetFacts | null;
  ttps: Ttp[];
  runs: JobStatus[];
  molecules: ReportMolecule[];
  structures: ReportStructure[];
  turns: ReportTurn[];
  /** 2D SVG markup of a molecule, null when it cannot be drawn. */
  depict?: (smiles: string) => string | null;
}

// ---------------------------------------------------------------------------
// reading the conversation
// ---------------------------------------------------------------------------

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value);

const failed = (step: ToolStep) => !step.result || step.result.isError || payloadError(step.result.result) !== null;

const byTime = (steps: readonly ToolStep[]) => [...steps].sort((a, b) => a.at - b.at);

/**
 * The profiles the conversation used, by id with the newest definition, and
 * the ids it named without ever showing their definition. Calls the server
 * refused do not count.
 */
export function conversationTtps(steps: readonly ToolStep[]): { ttps: Ttp[]; missing: string[] } {
  const found = new Map<string, Ttp>();
  const named: string[] = [];
  for (const step of byTime(steps)) {
    if (failed(step)) continue;
    const id = step.arguments.ttp_id;
    if (typeof id === "string" && id.trim() && !named.includes(id.trim())) named.push(id.trim());
    const payload = parseResult(step.result?.result);
    const seen = [asTtp(step.arguments.ttp), step.name === "get_ttp" ? asTtp(payload) : isRecord(payload) ? asTtp(payload.ttp) : null];
    for (const ttp of seen) {
      if (!ttp) continue;
      const known = found.get(ttp.id);
      if (!known || ttp.version >= known.version) found.set(ttp.id, ttp);
      if (!named.includes(ttp.id)) named.push(ttp.id);
    }
  }
  return { ttps: named.flatMap((id) => found.get(id) ?? []), missing: named.filter((id) => !found.has(id)) };
}

/** The REINVENT jobs the conversation started or asked about, oldest first. */
export function conversationRuns(steps: readonly ToolStep[]): string[] {
  const ids = conversationJobIds(steps);
  for (const step of byTime(steps)) {
    const id = step.arguments.job_id;
    if (typeof id === "string" && id.trim() && !failed(step) && !ids.includes(id.trim())) ids.push(id.trim());
  }
  return ids;
}

/** The last status the conversation saw of each job, for when the job manager no longer knows it. */
export function lastStatuses(steps: readonly ToolStep[]): Map<string, JobStatus> {
  const out = new Map<string, JobStatus>();
  for (const step of byTime(steps)) {
    if (step.name !== "get_optimization_status" || failed(step)) continue;
    const status = parseStatus(parseResult(step.result?.result));
    if (status) out.set(status.id, status);
  }
  return out;
}

/** Names the conversation looked compounds up by, by the SMILES it got back. */
export function compoundNames(steps: readonly ToolStep[]): Map<string, string> {
  const out = new Map<string, string>();
  for (const step of steps) {
    const name = step.arguments.name ?? step.arguments.query;
    if (step.name !== "fetch_compound" || typeof name !== "string" || failed(step)) continue;
    for (const panel of step.result?.panels ?? []) if (panel.kind === "molecule") out.set(panel.smiles, name);
  }
  return out;
}

/** Scores REINVENT reports per molecule that are not objectives. */
const NOT_OBJECTIVES = new Set(["Agent", "Prior", "Target", "Score", "total_score"]);
const RUN_TOP = 5;

export const runScore = (candidate: RunCandidate): number | null =>
  candidate.total ?? candidate.components.find((c) => c.name === "Score")?.transformed ?? null;

/** "log10(mol/L)"; dimensionless units say nothing. */
const shortUnit = (unit: string | undefined) => (unit && !/dimensionless/i.test(unit) ? unit : null);

/**
 * The molecules worth listing: each run's best, then what the conversation
 * scored, cofolded or predicted, once each. Molecules nothing was measured
 * on are left out unless there is nothing else.
 */
export function reportMolecules({
  runs,
  artifacts,
  canonical = (smiles) => smiles,
  limit = 12,
}: {
  runs: { jobId: string; candidates: RunCandidate[] }[];
  artifacts: Pick<Artifacts, "tables" | "molecules" | "structures" | "properties" | "steps">;
  canonical?: (smiles: string) => string | null;
  limit?: number;
}): ReportMolecule[] {
  const byKey = new Map<string, ReportMolecule>();
  const entry = (smiles: string, source: string) => {
    const key = canonical(smiles) ?? smiles;
    let molecule = byKey.get(key);
    if (!molecule) {
      molecule = { smiles, name: null, score: null, objectives: [], admet: [], admetNote: null, metrics: null, source };
      byKey.set(key, molecule);
    }
    return molecule;
  };

  for (const { jobId, candidates } of runs) {
    const ranked = [...candidates].sort((a, b) => (runScore(b) ?? -Infinity) - (runScore(a) ?? -Infinity)).slice(0, RUN_TOP);
    for (const candidate of ranked) {
      const molecule = entry(candidate.smiles, `run ${jobId}`);
      molecule.score ??= runScore(candidate);
      if (molecule.objectives.length) continue;
      molecule.objectives = candidate.components
        .filter((c) => !NOT_OBJECTIVES.has(c.name) && !/^std \(/.test(c.name))
        .flatMap((c) => {
          const value = c.raw ?? c.transformed;
          return value === null ? [] : [{ name: c.name, value }];
        });
    }
  }

  const names = compoundNames(artifacts.steps);
  for (const candidate of buildCandidates(artifacts)) {
    const molecule = entry(candidate.smiles, candidate.origin);
    molecule.name ??= candidate.name ?? names.get(candidate.smiles) ?? null;
    molecule.score ??= candidate.score ?? null;
    if (!molecule.metrics && candidate.metrics) molecule.metrics = candidate.metrics;
  }

  for (const { item } of artifacts.properties) {
    for (const predicted of item.molecules) {
      const smiles = predicted.smiles ?? predicted.standardized;
      if (!smiles) continue;
      const molecule = entry(smiles, item.tool);
      if (molecule.admet.length || molecule.admetNote) continue;
      if (!predicted.ok) {
        molecule.admetNote = "prediction failed";
        continue;
      }
      molecule.admet = item.endpoints.flatMap((endpoint) => {
        const value = predicted.values[endpoint];
        return value ? [{ label: endpointLabel(endpoint), value: formatValue({ mean: value.mean }), unit: shortUnit(value.unit) }] : [];
      });
      if (predicted.adPass === false) molecule.admetNote = "outside the models' applicability domain";
    }
  }

  const all = [...byKey.values()];
  const measured = all.filter((m) => m.score !== null || m.objectives.length || m.admet.length || m.metrics);
  return (measured.length ? measured : all.slice(0, 4)).slice(0, limit);
}

/** The polymer residues of a PDB or mmCIF file, in file order, with the author numbering the profiler uses. */
export function parseResidues(text: string, format: "pdb" | "cif"): FileResidue[] {
  const out: FileResidue[] = [];
  const add = (chain: string | null, resnum: number, resname: string) => {
    const last = out[out.length - 1];
    if (!Number.isFinite(resnum) || (last && last.chain === chain && last.resnum === resnum)) return;
    out.push({ chain, resnum, resname });
  };
  const lines = text.split(/\r?\n/);
  if (format === "pdb") {
    for (const line of lines) {
      if (!line.startsWith("ATOM  ")) continue;
      add(line.slice(21, 22).trim() || null, Number.parseInt(line.slice(22, 26), 10), line.slice(17, 20).trim());
    }
    return out;
  }
  const fields: string[] = [];
  let i = 0;
  for (; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line.startsWith("_atom_site.")) fields.push(line.slice(11).split(/\s/)[0]);
    else if (fields.length) break;
  }
  const at = (...names: string[]) => names.map((n) => fields.indexOf(n)).find((index) => index >= 0) ?? -1;
  const group = at("group_PDB");
  const chainAt = at("auth_asym_id", "label_asym_id");
  const numAt = at("auth_seq_id", "label_seq_id");
  const nameAt = at("auth_comp_id", "label_comp_id");
  if (numAt < 0 || nameAt < 0) return out;
  for (; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line || line.startsWith("#") || line.startsWith("loop_") || line.startsWith("_")) break;
    const cells = line.match(/'[^']*'|"[^"]*"|\S+/g) ?? [];
    if (group >= 0 && cells[group] !== "ATOM") continue;
    const chain = chainAt >= 0 ? cells[chainAt]?.replace(/^['"]|['"]$/g, "") : null;
    add(chain && chain !== "." && chain !== "?" ? chain : null, Number.parseInt(cells[numAt] ?? "", 10), cells[nameAt] ?? "");
  }
  return out;
}

/** The residues a ligand touches, by interaction type, specific interactions first. */
export function structureContacts(profile: InteractionsPanel, numbering: Numbering | null): ReportContacts[] {
  const byType = new Map<string, Map<string, number>>();
  for (const contact of profile.contacts) {
    if (contact.resnum === null) continue;
    const residues = byType.get(contact.type) ?? new Map<string, number>();
    residues.set(residueLabel(numbering, contact.resname, contact.chain, contact.resnum).label, displayNumber(numbering, contact.chain, contact.resnum));
    byType.set(contact.type, residues);
  }
  const counts: Record<string, number> = { ...profile.counts };
  for (const type of byType.keys()) counts[type] ??= profile.contacts.filter((c) => c.type === type).length;
  return sortedTypes(counts)
    .filter((type) => byType.has(type))
    .map((type) => ({
      type,
      count: counts[type],
      residues: [...(byType.get(type) ?? new Map<string, number>()).entries()].sort((a, b) => a[1] - b[1]).map(([label]) => label),
    }));
}

export function reportStructures(
  artifacts: Pick<Artifacts, "structures" | "interactions">,
  numberingOf: (path: string) => Numbering | null,
  nameOf: (smiles: string) => string | null = () => null,
  limit = 6,
): ReportStructure[] {
  return artifacts.structures.slice(0, limit).map(({ item }) => {
    const profile = contactsFor(item, artifacts.interactions);
    const numbering = numberingOf(item.path);
    const smiles = item.smiles ?? profile?.smiles ?? null;
    const name = smiles ? nameOf(smiles) : null;
    return {
      title: name ? `${name} · ${structureTitle(item)}` : structureTitle(item),
      run: runOf(item.path),
      smiles,
      metrics: item.metrics ?? null,
      contacts: profile ? structureContacts(profile, numbering) : [],
      numbering: numbering ? "uniprot" : "file",
    };
  });
}

/** Each answer, titled as in the chat and cut to its gist. */
export function reportTurns(turns: Turn[]): ReportTurn[] {
  const out: ReportTurn[] = [];
  let question = "";
  for (const turn of turns) {
    if (turn.role === "user") {
      question = turn.content;
      continue;
    }
    const texts = turn.blocks.flatMap((block) => (block.kind === "text" && block.content.trim() ? [block.content] : []));
    const tools = turn.blocks.flatMap((block) => (block.kind === "tool" ? [block.name] : []));
    const asked = question;
    question = "";
    if (!texts.length) continue;
    const title = answerTitle(texts, tools);
    const answer = answerExcerpt(withoutLeadingHeading(texts[texts.length - 1], title));
    if (answer) out.push({ question: oneLine(asked, 160), title, answer });
  }
  return out;
}

const oneLine = (text: string, max: number) => {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length > max ? `${flat.slice(0, max - 1).trimEnd()}…` : flat;
};

const HEADING_LINE = /^\s{0,3}#{1,6}\s/;
const TABLE_LINE = /^\s*\|/;
const RULE_LINE = /^\s*([-*_])(\s*\1){2,}\s*$/;
const LIST_LINE = /^\s*([-*+]|\d+[.)])\s+/;
/** Sentences that announce work rather than report it. */
const PREAMBLE = /^(I have|I've|I'll|I will|Let me|Now I|Here is|Here's|I'm going)\b/i;
const SUMMARY = /summary|conclusion|bottom line|tl;?dr|verdict|key (findings|takeaways)|in short|recommendation/i;

const plain = (text: string) =>
  text
    .replace(/^\s*>\s?/gm, "")
    .replace(/\[!\w+\]/g, "")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/(\*\*|__)(.+?)\1/g, "$2")
    .replace(/`([^`]*)`/g, "$1")
    .replace(/\s+/g, " ")
    .trim();

/** A paragraph without the sentences that only announce what comes next. */
const withoutPreamble = (text: string) => {
  const sentences = text.split(/(?<=[.!?])\s+/);
  while (sentences.length && PREAMBLE.test(sentences[0])) sentences.shift();
  return sentences.join(" ");
};

/** Fewer words than this outside code is a label ("Ligand: `CC…`"), not prose. */
const MIN_WORDS = 4;

function prose(markdown: string, lists: boolean): string[] {
  const out: string[] = [];
  for (const block of markdown.replace(/```[\s\S]*?(```|$)/g, "\n\n").split(/\n\s*\n/)) {
    let lines = block.split("\n").filter((line) => line.trim() && !HEADING_LINE.test(line) && !TABLE_LINE.test(line) && !RULE_LINE.test(line));
    const listed = lines.length > 0 && lines.every((line) => LIST_LINE.test(line));
    if (!listed) lines = lines.filter((line) => !LIST_LINE.test(line));
    if (!lines.length || (listed && !lists)) continue;
    const text = listed ? lines.map((line) => plain(line.replace(LIST_LINE, ""))).join("; ") : withoutPreamble(plain(lines.join(" ")));
    const raw = lines.join(" ");
    const label = raw.includes("`") && raw.replace(/`[^`]*`/g, "").split(/\s+/).filter((word) => /\p{L}/u.test(word)).length < MIN_WORDS;
    if (text && !text.endsWith(":") && !label) out.push(text);
  }
  return out;
}

/**
 * The gist of an answer in plain text: its summary section when it has one,
 * else its first paragraphs of prose, cut at a sentence near `max` characters.
 */
export function answerExcerpt(markdown: string, max = 420): string {
  const lines = markdown.split("\n");
  const start = lines.findIndex((line) => HEADING_LINE.test(line) && SUMMARY.test(line));
  let paragraphs: string[] = [];
  if (start >= 0) {
    const level = /#+/.exec(lines[start])?.[0].length ?? 1;
    const end = lines.findIndex((line, i) => i > start && HEADING_LINE.test(line) && (/#+/.exec(line)?.[0].length ?? 1) <= level);
    paragraphs = prose(lines.slice(start + 1, end < 0 ? undefined : end).join("\n"), true);
  }
  if (!paragraphs.length) paragraphs = prose(markdown, false);
  if (!paragraphs.length) paragraphs = prose(markdown, true);

  let text = "";
  for (const paragraph of paragraphs) {
    if (text && text.length + paragraph.length + 1 > max) break;
    text = text ? `${text} ${paragraph}` : paragraph;
  }
  if (text.length <= max) return text;
  const cut = text.slice(0, max);
  const sentence = Math.max(...[...cut.matchAll(/[.!?](?=\s)/g)].map((m) => m.index), -1);
  if (sentence > max * 0.4) return cut.slice(0, sentence + 1);
  return `${cut.slice(0, cut.lastIndexOf(" ")).trimEnd()}…`;
}

// ---------------------------------------------------------------------------
// HTML
// ---------------------------------------------------------------------------

const ESCAPES: Record<string, string> = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

export const escapeHtml = (text: string) => text.replace(/[&<>"']/g, (ch) => ESCAPES[ch]);

const esc = escapeHtml;

const num = (value: number | null | undefined, digits = 2) => (value == null || !Number.isFinite(value) ? "—" : value.toFixed(digits).replace(/^-/, "−"));

const date = (value: string | Date) => {
  const d = new Date(value);
  return Number.isNaN(d.getTime())
    ? String(value)
    : d.toLocaleString("en-GB", { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
};

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

/** RDKit's SVG without its XML prolog, so it can sit inline in HTML. */
const inlineSvg = (svg: string | null | undefined) => (svg ? svg.replace(/<\?xml[^>]*\?>\s*/, "").replace(/<script[\s\S]*?<\/script>/gi, "") : "");

const ACCENT = "#ed1c24";
const CURVE = "#a38a3e";

const CSS = `
*{box-sizing:border-box}
html{background:#f6f6f7}
body{margin:0;font:11px/1.45 Arial,"Helvetica Neue",Helvetica,sans-serif;color:#1d1d20}
main{max-width:860px;margin:24px auto;padding:28px 36px 36px;background:#fff;border:1px solid #eaeaed;border-radius:8px}
header{padding-left:12px;border-left:3px solid ${ACCENT};margin-bottom:18px}
.brand{font-size:9px;letter-spacing:.08em;text-transform:uppercase;color:#85878a;font-weight:bold}
h1{font-size:17px;line-height:1.3;margin:2px 0 4px;overflow-wrap:anywhere}
h2{font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:#5e5e5e;margin:0 0 8px;padding-bottom:4px;border-bottom:1px solid #eaeaed}
h3{font-size:12px;margin:0}
section{margin-top:22px}
p{margin:0}
.meta,.muted{color:#5e5e5e}
.faint{color:#85878a}
.small{font-size:10px}
.mono{font-family:Consolas,"SF Mono",Menlo,monospace;font-size:9.5px;overflow-wrap:anywhere;color:#3a3a3e}
.card{border:1px solid #eaeaed;border-radius:6px;padding:10px 12px;margin-bottom:8px;break-inside:avoid}
.row{display:flex;gap:12px;align-items:flex-start}
.grow{flex:1;min-width:0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:8px}
.legend{margin-top:8px;font-size:9.5px;color:#85878a}
.grid .card{margin:0}
.depict{flex:none;border-radius:4px;background:#fafafa}
.depict svg{display:block;width:100%;height:100%}
.kv{display:flex;flex-wrap:wrap;gap:2px 12px;margin-top:4px}
.kv span b{font-weight:normal;color:#85878a;margin-right:3px}
table{border-collapse:collapse;width:100%;margin-top:6px}
th{text-align:left;font-weight:normal;color:#85878a;font-size:9.5px;padding:2px 8px 3px 0;border-bottom:1px solid #eaeaed}
td{padding:4px 8px 4px 0;border-bottom:1px solid #f2f2f4;vertical-align:middle}
tr:last-child td{border-bottom:0}
.bar{display:inline-block;width:40px;height:4px;border-radius:2px;background:#eeeef0;vertical-align:middle;margin-right:5px;overflow:hidden}
.bar i{display:block;height:100%;background:${CURVE}}
.pill{display:inline-block;padding:0 6px;border-radius:8px;font-size:9.5px;line-height:15px;white-space:nowrap}
.ok{background:#e6f5ee;color:#16784f}
.bad{background:#fdecec;color:#b3141a}
.unknown{background:#f0f0f2;color:#5e5e5e}
.note{margin-top:4px;font-size:10px;color:#8a5a00}
.contacts{margin-top:6px}
.contacts p{margin-top:2px}
.dot{display:inline-block;width:6px;height:6px;border-radius:3px;margin-right:4px;vertical-align:1px}
.q{color:#5e5e5e;font-style:italic;margin-bottom:2px}
.answer{margin-top:3px}
footer{margin-top:28px;color:#85878a;font-size:9px}
@page{margin:14mm}
@media print{
html{background:#fff}
main{margin:0;padding:0;border:0;max-width:none}
h2{break-after:avoid}
}
`;

function header(input: ReportInput): string {
  const counts = [
    describesTarget(input.target) ? "target" : null,
    input.ttps.length ? plural(input.ttps.length, "profile") : null,
    input.runs.length ? plural(input.runs.length, "run") : null,
    input.molecules.length ? plural(input.molecules.length, "molecule") : null,
    input.structures.length ? plural(input.structures.length, "structure") : null,
    input.turns.length ? plural(input.turns.length, "answer") : null,
  ].filter(Boolean);
  return `<header>
<div class="brand">AMD · Cofolding agent · report</div>
<h1>${esc(input.title || "Untitled conversation")}</h1>
<p class="meta">Started ${esc(date(input.created))} · exported ${esc(date(input.generated))}${counts.length ? ` · ${esc(counts.join(", "))}` : ""}</p>
</header>`;
}

/** Whether there is more to say about the target than the residues ligands touched, which the structures already list. */
const describesTarget = (facts: TargetFacts | null): facts is TargetFacts =>
  facts !== null && Boolean(facts.gene || facts.name || facts.uniprot || facts.sequence || facts.pdbIds.length);

function targetSection(facts: TargetFacts | null): string {
  if (!describesTarget(facts)) return "";
  const heading = [facts.gene, facts.name].filter(Boolean).join(" · ") || "Protein target";
  const uniprotLength = facts.uniprotSequence?.length ?? null;
  const constructLength = facts.sequence && facts.sequence !== facts.uniprotSequence ? facts.sequence.length : null;
  const facts1 = [
    facts.uniprot && `<span><b>UniProt</b>${esc(facts.uniprot)}${facts.entryName ? ` (${esc(facts.entryName)})` : ""}</span>`,
    facts.organism && `<span><b>Organism</b>${esc(facts.organism)}</span>`,
    (uniprotLength ?? facts.sequenceLength) && `<span><b>Sequence</b>${(uniprotLength ?? facts.sequenceLength ?? 0).toLocaleString("en")} aa</span>`,
    constructLength && uniprotLength && `<span><b>Construct</b>${constructLength.toLocaleString("en")} aa</span>`,
    facts.pdbIds.length && `<span><b>PDB</b>${esc(facts.pdbIds.slice(0, 6).join(", "))}${facts.pdbIds.length > 6 || (facts.pdbCount ?? 0) > 6 ? ` +${Math.max(facts.pdbIds.length, facts.pdbCount ?? 0) - 6}` : ""}</span>`,
  ].filter(Boolean);
  return `<section id="target"><h2>Target</h2><div class="card"><h3>${esc(heading)}</h3><div class="kv">${facts1.join("")}</div></div></section>`;
}

/** A transform's curve, 0 to 1, with the rewarded window shaded. */
export function curveSvg(transform: Transform, width = 96, height = 26): string {
  const domain = curveDomain(transform);
  const points = transformCurve(transform, 40, domain);
  if (!points.length) return `<span class="faint small">${esc(transform.type)}</span>`;
  const x = linear([domain.from, domain.to], [2, width - 2]);
  const y = linear([0, 1], [height - 2, 2]);
  const fields = rangeFields(transform.type).map((f) => f.key);
  const low = typeof transform.low === "number" && fields.includes("low") ? transform.low : null;
  const high = typeof transform.high === "number" && fields.includes("high") ? transform.high : null;
  const band =
    low !== null && high !== null
      ? `<rect x="${x(Math.min(low, high)).toFixed(1)}" y="2" width="${Math.max(1, Math.abs(x(high) - x(low))).toFixed(1)}" height="${height - 4}" fill="${CURVE}" opacity="0.1"/>`
      : "";
  return `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(curveLabel(transform.type))}">${band}<path d="M2 ${height - 2}H${width - 2}" stroke="#d7d7dc" fill="none"/><path d="${linePath(points, x, y)}" fill="none" stroke="${CURVE}" stroke-width="1.5" stroke-linejoin="round"/></svg>`;
}

function objectiveRow(o: Objective, share: number, curves: boolean): string {
  const range = targetRange(o.transform);
  return `<tr>
<td>${esc(o.name)}${o.depends_on ? `<div class="faint small">after ${esc(o.depends_on)}</div>` : ""}</td>
<td class="muted">${esc(sourceLabel(o))}</td>
<td><span class="bar"><i style="width:${Math.round(share * 100)}%"></i></span>${esc(String(+o.weight.toPrecision(3)))}</td>
${curves ? `<td>${o.transform ? curveSvg(o.transform) : ""}</td>` : ""}
<td class="muted">${esc(o.transform ? curveLabel(o.transform.type) : "Score as is")}${range ? ` <span class="faint">${esc(range)}</span>` : ""}</td>
</tr>`;
}

function ttpSection(ttps: Ttp[]): string {
  if (!ttps.length) return "";
  const cards = ttps.map((ttp) => {
    const shares = weightShares(ttp);
    const objectives = [...objectivesOf(ttp, "pre_scoring"), ...objectivesOf(ttp, "scoring")];
    const curves = objectives.some((o) => o.transform);
    const rows = objectives.map((o) => objectiveRow(o, shares.get(o.name) ?? 0, curves));
    const aggregation = (ttp.scoring.aggregation ?? "geometric_mean").replace("_", " ");
    const extra = [
      ttp.target?.protein_sequence && `target ${ttp.target.protein_sequence.length.toLocaleString("en")} aa${ttp.target.pdb_id ? `, PDB ${ttp.target.pdb_id}` : ""}`,
      ttp.constraints?.length && plural(ttp.constraints.length, "constraint"),
      ...ttpGeneration(ttp).map((g) => `${g.label.toLowerCase()} ${g.value}`),
    ].filter(Boolean) as string[];
    return `<div class="card">
<h3>${esc(ttp.name)} <span class="faint small">${esc(ttp.id)} v${ttp.version} · ${esc(aggregation)}</span></h3>
${ttp.scoring.comment ? `<p class="muted small">${esc(ttp.scoring.comment)}</p>` : ""}
${extra.length ? `<p class="faint small">${esc(extra.join(" · "))}</p>` : ""}
<table><thead><tr><th>Objective</th><th>Source</th><th>Weight</th>${curves ? "<th>Curve</th>" : ""}<th>Rewards</th></tr></thead><tbody>${rows.join("")}</tbody></table>
</div>`;
  });
  return `<section id="ttp"><h2>Target product ${ttps.length === 1 ? "profile" : "profiles"}</h2>${cards.join("")}</section>`;
}

/** Mean and best score per step: the run's trajectory, else its total score component. */
export function runSeries(status: JobStatus): { mean: { x: number; y: number }[]; best: { x: number; y: number }[] } {
  if (status.trajectory.length) {
    return { mean: status.trajectory.map((p) => ({ x: p.step, y: p.mean })), best: runningBest(status.trajectory) };
  }
  const total = status.componentTrajectories.Score ?? status.componentTrajectories.total_score ?? [];
  return { mean: total.map((p) => ({ x: p.step, y: p.mean })), best: [] };
}

function runChart(status: JobStatus): string {
  const { mean, best } = runSeries(status);
  if (mean.length < 2) return "";
  const width = 520;
  const height = 110;
  const left = 26;
  const bottom = height - 16;
  const lastStep = Math.max(...mean.map((p) => p.x), ...best.map((p) => p.x));
  const [lo, hi] = scoreDomain([...mean, ...best].map((p) => p.y));
  const x = linear([0, Math.max(status.maxSteps ?? 0, lastStep, 1)], [left, width - 6]);
  const y = linear([lo, hi], [bottom, 6]);
  const grid = ticks(lo, hi, 3)
    .map((t) => `<path d="M${left} ${y(t).toFixed(1)}H${width - 6}" stroke="#f0f0f2"/><text x="${left - 4}" y="${(y(t) + 3).toFixed(1)}" text-anchor="end">${num(t, 1)}</text>`)
    .join("");
  const axis = `<text x="${left}" y="${height - 3}">0</text><text x="${width - 6}" y="${height - 3}" text-anchor="end">${esc(String(Math.max(status.maxSteps ?? 0, lastStep)))} steps</text>`;
  const legend = `<p class="legend"><span style="color:#5e5e5e">━</span> mean score${best.length ? ` <span style="color:${ACCENT}">━</span> best so far` : ""}</p>`;
  return `${legend}<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" font-size="8.5" fill="#85878a" role="img" aria-label="Score per step">${grid}${axis}<path d="${linePath(mean, x, y)}" fill="none" stroke="#5e5e5e" stroke-width="1.4"/>${best.length ? `<path d="${linePath(best, x, y)}" fill="none" stroke="${ACCENT}" stroke-width="1.4"/>` : ""}</svg>`;
}

const STATE_TONE: Record<string, string> = { completed: "ok", failed: "bad" };

function runsSection(runs: JobStatus[]): string {
  if (!runs.length) return "";
  const cards = runs.map((status) => {
    const best = status.bestScore ?? status.components.find((c) => c.name === "Score")?.max ?? null;
    const mean = status.meanScore ?? status.components.find((c) => c.name === "Score")?.mean ?? null;
    const steps = status.currentStep != null ? `${status.state === "completed" && status.maxSteps ? status.maxSteps : Math.max(0, status.currentStep - 1)}${status.maxSteps ? ` / ${status.maxSteps}` : ""}` : null;
    const stats = [
      steps && `<span><b>Steps</b>${esc(steps)}</span>`,
      status.molecules != null && `<span><b>Molecules</b>${status.molecules.toLocaleString("en")}</span>`,
      status.validity != null && `<span><b>Valid</b>${Math.round(status.validity * 100)}%</span>`,
      best != null && `<span><b>Best</b>${num(best, 3)}</span>`,
      mean != null && `<span><b>Mean</b>${num(mean, 3)}</span>`,
      status.model && `<span><b>Model</b>${esc(status.model)}</span>`,
      status.startedAt && `<span><b>Started</b>${esc(date(status.startedAt))}</span>`,
    ].filter(Boolean);
    const convergence = status.diagnostics?.convergence;
    return `<div class="card">
<h3>${esc(jobTitle(status))} <span class="faint small">${esc(status.id)} · ${esc(typeLabel(status.type))}</span> <span class="pill ${STATE_TONE[status.state] ?? "unknown"}">${esc(status.state)}</span>${convergence ? ` <span class="faint small">${esc(convergence)}</span>` : ""}</h3>
<div class="kv">${stats.join("")}</div>
${runChart(status)}
${status.error ? `<p class="note">${esc(oneLine(status.error, 240))}</p>` : ""}
</div>`;
  });
  return `<section id="runs"><h2>REINVENT ${runs.length === 1 ? "run" : "runs"}</h2>${cards.join("")}</section>`;
}

/**
 * Gate pill, affinity and, when the pose fails the gate, a word of caution.
 * Compact for a molecule whose complex is described in full further on.
 */
function verdict(metrics: StructureMetrics | null, compact = false): string {
  if (!metrics) return "";
  const g = gate(metrics);
  const label = g.status === "pass" ? "Gate passed" : g.status === "fail" ? "Gate failed" : "Gate incomplete";
  const tone = g.status === "pass" ? "ok" : g.status === "fail" ? "bad" : "unknown";
  const value = affinityText(metrics);
  const affinity =
    value && metrics.affinity !== undefined
      ? `<span><b>Affinity</b>${esc(value.replace(/-/g, "−"))}${compact ? "" : " log10(IC50/µM)"}, ${esc(affinityConcentration(metrics.affinity))}</span>`
      : "";
  if (compact) {
    return `<p class="kv"><span class="pill ${tone}">${label}</span>${affinity}</p>${affinity && g.status !== "pass" ? `<p class="note">Affinity of a pose that fails the gate: a ranking signal only.</p>` : ""}`;
  }
  const lines = [`<p class="kv"><span class="pill ${tone}">${label}</span>${g.reasons.length ? ` <span class="faint small">${esc(g.reasons.join("; "))}</span>` : ""}</p>`];
  if (affinity) {
    lines.push(`<p class="kv">${affinity}${metrics.binderProbability !== undefined ? `<span><b>Binder p</b>${num(metrics.binderProbability)}</span>` : ""}</p>`);
    if (g.status !== "pass") lines.push(`<p class="note">The pose does not pass the gate, so read this affinity as a ranking signal, not a measured IC50.</p>`);
  }
  return lines.join("");
}

function depiction(input: ReportInput, smiles: string | null, width: number, height: number): string {
  const svg = smiles ? inlineSvg(input.depict?.(smiles)) : "";
  return svg ? `<div class="depict" style="width:${width}px;height:${height}px">${svg}</div>` : "";
}

function moleculesSection(input: ReportInput): string {
  if (!input.molecules.length) return "";
  const cards = input.molecules.map((m, i) => {
    const values = [
      m.score !== null && `<span><b>Score</b>${num(m.score, 3)}</span>`,
      ...m.objectives.map((o) => `<span><b>${esc(o.name)}</b>${num(o.value)}</span>`),
      ...m.admet.map((a) => `<span><b>${esc(a.label)}</b>${esc(a.value.replace(/^-/, "−"))}${a.unit ? ` <span class="faint">${esc(a.unit)}</span>` : ""}</span>`),
    ].filter(Boolean);
    const metrics = m.metrics
      ? [
          m.metrics.confidence !== undefined && `<span><b>Confidence</b>${num(m.metrics.confidence)}</span>`,
          m.metrics.ligandIptm !== undefined && `<span><b>Ligand ipTM</b>${num(m.metrics.ligandIptm)}</span>`,
        ].filter(Boolean)
      : [];
    return `<div class="card row">
${depiction(input, m.smiles, 120, 92)}
<div class="grow">
<h3>${m.name ? `${esc(m.name)} ` : ""}<span class="faint small">#${i + 1} · ${esc(m.source)}</span></h3>
<p class="mono">${esc(m.smiles)}</p>
${values.length || metrics.length ? `<div class="kv">${[...values, ...metrics].join("")}</div>` : ""}
${m.admetNote ? `<p class="faint small">ADMET: ${esc(m.admetNote)}</p>` : ""}
${verdict(m.metrics, true)}
</div>
</div>`;
  });
  return `<section id="molecules"><h2>Top molecules</h2><div class="grid">${cards.join("")}</div></section>`;
}

function structuresSection(input: ReportInput): string {
  if (!input.structures.length) return "";
  const cards = input.structures.map((s) => {
    const m = s.metrics;
    const pb = m?.posebustersPassed !== undefined && m.posebustersTotal !== undefined ? `${m.posebustersPassed}/${m.posebustersTotal}` : null;
    const values = m
      ? [
          m.confidence !== undefined && `<span><b>Confidence</b>${num(m.confidence)}</span>`,
          m.iptm !== undefined && `<span><b>ipTM</b>${num(m.iptm)}</span>`,
          m.ligandIptm !== undefined && `<span><b>Ligand ipTM</b>${num(m.ligandIptm)}</span>`,
          m.plddt !== undefined && `<span><b>pLDDT</b>${num(m.plddt)}</span>`,
          pb && `<span><b>PoseBusters</b>${pb}</span>`,
        ].filter(Boolean)
      : [];
    const contacts = s.contacts.length
      ? `<div class="contacts">${s.contacts
          .map(
            (c) =>
              `<p><span class="dot" style="background:${contactColor(c.type, "light")}"></span>${esc(contactLabel(c.type))} <span class="faint">${c.count}</span> <span class="mono">${esc(c.residues.join(" "))}</span></p>`,
          )
          .join("")}<p class="faint small">${s.numbering === "uniprot" ? "UniProt numbering" : "Numbering of the structure file"}</p></div>`
      : "";
    return `<div class="card row">
${depiction(input, s.smiles, 150, 112)}
<div class="grow">
<h3>${esc(s.title)}${s.run ? ` <span class="faint small">run ${esc(s.run)}</span>` : ""}</h3>
${values.length ? `<div class="kv">${values.join("")}</div>` : ""}
${verdict(m)}
${contacts}
</div>
</div>`;
  });
  return `<section id="structures"><h2>Cofolded ${input.structures.length === 1 ? "structure" : "structures"}</h2>${cards.join("")}</section>`;
}

function turnsSection(turns: ReportTurn[]): string {
  if (!turns.length) return "";
  const cards = turns.map(
    (t) => `<div class="card">
${t.question ? `<p class="q">${esc(t.question)}</p>` : ""}
<h3>${esc(t.title)}</h3>
<p class="answer">${esc(t.answer)}</p>
</div>`,
  );
  return `<section id="answers"><h2>Answers</h2>${cards.join("")}</section>`;
}

export function buildReport(input: ReportInput): string {
  const body = [
    header(input),
    targetSection(input.target),
    ttpSection(input.ttps),
    runsSection(input.runs),
    moleculesSection(input),
    structuresSection(input),
    turnsSection(input.turns),
  ]
    .filter(Boolean)
    .join("\n");
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(input.title || "Untitled conversation")} · report</title>
<style>${CSS}</style>
</head>
<body>
<main>
${body}
<footer>${esc(input.id)} · generated by the AMD Cofolding agent. Predictions are computational; verify before acting on them.</footer>
</main>
</body>
</html>
`;
}
