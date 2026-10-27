// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { ConversationEvent, SessionSummary, StructurePanel } from "../types/api";
import { callTool, download, sessionEvents } from "./api";
import { collectArtifacts } from "./artifacts";
import { parseCandidates, parseStatus, type JobStatus } from "./jobs";
import { buildNumbering, type Numbering } from "./numbering";
import { canonical, depict, loadRDKit } from "./rdkit";
import { buildReport, conversationRuns, conversationTtps, lastStatuses, parseResidues, reportMolecules, reportStructures, reportTurns } from "./report";
import { asTtp, extractTargetFacts, parseResult, toolEvents, type Ttp } from "./ttp";
import { applyEvents, EMPTY_TRANSCRIPT } from "./transcript";

/**
 * A conversation as Markdown: questions, tool calls with their arguments,
 * and answers, in order. Meant for a lab notebook or a report, so the raw
 * tool output is left out; the JSON export has everything.
 */
export function sessionToMarkdown(session: SessionSummary, events: ConversationEvent[]): string {
  const { turns } = applyEvents(EMPTY_TRANSCRIPT, events);
  const lines = [`# ${session.title || "Untitled conversation"}`, "", `_${session.id} · started ${session.created}_`, ""];
  for (const turn of turns) {
    if (turn.role === "user") {
      lines.push("## Question", "", turn.content, "");
      continue;
    }
    for (const block of turn.blocks) {
      switch (block.kind) {
        case "tool": {
          const status = !block.result ? "did not finish" : block.result.isError ? "failed" : "ok";
          const took = block.result?.elapsedMs != null ? `, ${(block.result.elapsedMs / 1000).toFixed(1)} s` : "";
          lines.push(`- \`${block.name}\` (${status}${took})`, "", "  ```json", indent(JSON.stringify(block.arguments, null, 2)), "  ```", "");
          break;
        }
        case "text":
          lines.push("## Answer", "", block.content, "");
          break;
        case "error":
          lines.push(`> **Error:** ${block.message.split("\n")[0]}`, "");
          break;
        case "cancelled":
          lines.push("> Stopped by the user.", "");
          break;
        case "thinking":
          break;
      }
    }
  }
  return lines.join("\n");
}

function indent(text: string): string {
  return text
    .split("\n")
    .map((line) => `  ${line}`)
    .join("\n");
}

/** A file name from a title: lower case, dashes, no punctuation. */
export function slug(title: string, fallback: string): string {
  const cleaned = title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 60);
  return cleaned || fallback;
}

/** Structures the report lists at most; each one is a file to fetch for its numbering. */
const REPORT_STRUCTURES = 6;

/** UniProt numbering of each structure, read from its file; null when the file or the sequence is missing. */
async function numberings(structures: StructurePanel[], uniprot: string | null): Promise<Map<string, Numbering | null>> {
  const out = new Map<string, Numbering | null>();
  if (!uniprot) return out;
  await Promise.all(
    structures.map(async (structure) => {
      try {
        const response = await fetch(structure.url);
        if (!response.ok) return;
        out.set(structure.path, buildNumbering(parseResidues(await response.text(), structure.format), uniprot));
      } catch {
        // Without the file the contacts keep the file's numbers.
      }
    }),
  );
  return out;
}

/**
 * The conversation's report as one HTML file. Runs are read live from the
 * job manager, falling back to the last status the conversation saw;
 * molecules are drawn with RDKit when it loads, and listed without a
 * drawing when it does not.
 */
export async function sessionReport(sessionId: string, now = new Date()): Promise<{ name: string; html: string }> {
  const { session, events } = await sessionEvents(sessionId);
  const { turns } = applyEvents(EMPTY_TRANSCRIPT, events);
  const artifacts = collectArtifacts(turns);
  const facts = extractTargetFacts(toolEvents(artifacts.steps));
  const seen = conversationTtps(artifacts.steps);
  const known = lastStatuses(artifacts.steps);
  const structures = artifacts.structures.slice(0, REPORT_STRUCTURES);

  const [rdkit, fetched, runs, numbered] = await Promise.all([
    loadRDKit().catch(() => null),
    Promise.all(
      seen.missing.map((id) =>
        callTool("get_ttp", { ttp_id: id }).then(
          (response) => asTtp(parseResult(response.result)),
          () => null,
        ),
      ),
    ),
    Promise.all(
      conversationRuns(artifacts.steps).map(async (jobId) => {
        const [status, candidates] = await Promise.all([
          callTool("get_optimization_status", { job_id: jobId }).then(
            (response) => parseStatus(parseResult(response.result)),
            () => null,
          ),
          callTool("get_top_candidates", { job_id: jobId, top_n: 10 }).then(
            (response) => parseCandidates(parseResult(response.result)),
            () => [],
          ),
        ]);
        return { jobId, status: status ?? known.get(jobId) ?? null, candidates };
      }),
    ),
    numberings(
      structures.map(({ item }) => item),
      facts.uniprotSequence,
    ),
  ]);

  const key = (smiles: string) => (rdkit ? canonical(rdkit, smiles) : null) ?? smiles;
  const molecules = reportMolecules({ runs, artifacts, canonical: key });
  const names = new Map(molecules.flatMap((m) => (m.name ? [[key(m.smiles), m.name] as const] : [])));
  const drawings = new Map<string, string | null>();
  const ttps: Ttp[] = [...seen.ttps, ...fetched.filter((ttp): ttp is Ttp => ttp !== null)];

  const html = buildReport({
    title: session.title,
    id: session.id,
    created: session.created,
    generated: now,
    target: facts,
    ttps,
    runs: runs.map((run) => run.status).filter((status): status is JobStatus => status !== null),
    molecules,
    structures: reportStructures(
      { structures, interactions: artifacts.interactions },
      (path) => numbered.get(path) ?? null,
      (smiles) => names.get(key(smiles)) ?? null,
    ),
    turns: reportTurns(turns),
    depict: (smiles) => {
      if (!rdkit) return null;
      if (!drawings.has(smiles)) drawings.set(smiles, depict(rdkit, smiles, { width: 240, height: 180, dark: false }));
      return drawings.get(smiles) ?? null;
    },
  });
  return { name: `${slug(session.title, session.id)}-report.html`, html };
}

export async function downloadReport(sessionId: string): Promise<void> {
  const { name, html } = await sessionReport(sessionId);
  download(name, html, "text/html");
}

/**
 * The report in a new window, with the print dialog open. The window is
 * opened before anything is awaited: browsers only allow pop-ups that
 * follow a click directly.
 */
export async function printReport(sessionId: string): Promise<void> {
  const win = window.open("", "_blank");
  if (!win) throw new Error("The browser blocked the report window. Allow pop-ups for this site and try again.");
  win.document.write('<!doctype html><title>Report</title><p style="font:12px Arial,sans-serif;color:#5e5e5e;padding:24px">Preparing the report…</p>');
  try {
    const { html } = await sessionReport(sessionId);
    win.document.open();
    win.document.write(html);
    win.document.close();
    win.focus();
    win.setTimeout(() => win.print(), 200);
  } catch (reason) {
    win.close();
    throw reason;
  }
}
