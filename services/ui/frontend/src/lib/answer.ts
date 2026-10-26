// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { Panel, StructurePanel } from "../types/api";
import type { ViewerRequest } from "./viewerBus";
import type { WorkbenchTab } from "./workbenchBus";

/**
 * How an answer is presented: its title, the links to what it produced and
 * its callouts. The agent gives none of these, so they are read off the
 * answer text and the turn's tool calls.
 */

/** What a turn is about, by its main tool; the first match in the turn wins. */
const TOOL_TITLES: [RegExp, string][] = [
  [/^score_molecules$/, "Cofolding and checking pose quality"],
  [/^profile_interactions$/, "Profiling interactions"],
  [/^predict_admet$/, "Predicting ADMET"],
  [/^predict_retrodmta$/, "Predicting with RetroDMTA"],
  [/^(run|start|launch)_(optimization|reinvent|generation)/, "Running REINVENT"],
  [/^get_optimization_status$/, "Checking the run"],
  [/^get_top_candidates$/, "Ranking candidates"],
  [/^get_ttp$/, "Reading the target product profile"],
  [/^fetch_target_info$/, "Looking up the target"],
  [/^fetch_compound$/, "Looking up a compound"],
  [/^list_shared_assets$/, "Listing shared assets"],
  [/^(list_available_models|list_external_services)$/, "Checking available services"],
];

/** Tools that only gather context; a turn is titled by the work it did, not by these. */
const MINOR = /^(list_|get_ttp$|fetch_)/;

const HEADING = /^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$/m;

/** The text of the answer's first heading, without inline markdown. */
export function firstHeading(markdown: string): string | null {
  const match = HEADING.exec(markdown);
  if (!match) return null;
  const text = match[1].replace(/[*_`]/g, "").replace(/\[([^\]]*)\]\([^)]*\)/g, "$1").trim();
  return text || null;
}

/** A bold title for a turn: the answer's first heading, else its main tool, else "Answer". */
export function answerTitle(texts: string[], tools: string[]): string {
  for (const text of texts) {
    const heading = firstHeading(text);
    if (heading) return heading;
  }
  const titled = (names: string[]) => {
    for (const name of names) {
      const hit = TOOL_TITLES.find(([pattern]) => pattern.test(name));
      if (hit) return hit[1];
    }
    return null;
  };
  return titled(tools.filter((name) => !MINOR.test(name))) ?? titled(tools) ?? "Answer";
}

/** The answer without its first line when that line is the heading used as the title. */
export function withoutLeadingHeading(markdown: string, title: string): string {
  const match = /^\s*#{1,6}\s+(.+?)\s*#*\s*(\n|$)/.exec(markdown);
  if (!match || firstHeading(match[0]) !== title) return markdown;
  return markdown.slice(match[0].length).replace(/^\s*\n/, "");
}

export type ArtifactTarget = { kind: "viewer"; request: ViewerRequest } | { kind: "workbench"; tab: WorkbenchTab };

export interface ArtifactLink {
  label: string;
  target: ArtifactTarget;
}

/** The pose to open first: the most confident one. */
export function bestStructure(structures: StructurePanel[]): StructurePanel | null {
  let best: StructurePanel | null = null;
  for (const panel of structures) {
    if (!best || (panel.metrics?.confidence ?? -1) > (best.metrics?.confidence ?? -1)) best = panel;
  }
  return best;
}

/** Links under an answer to the places that show what it produced. */
export function artifactLinks(panels: Panel[], tools: string[]): ArtifactLink[] {
  const out: ArtifactLink[] = [];
  const add = (label: string, target: ArtifactTarget) => {
    if (!out.some((link) => link.label === label)) out.push({ label, target });
  };
  const structures = panels.filter((panel): panel is StructurePanel => panel.kind === "structure");
  const best = bestStructure(structures);
  if (best) add(structures.length > 1 ? "Open best pose" : "Open pose", { kind: "viewer", request: { path: best.path, source: "answer" } });

  const profile = panels.find((panel) => panel.kind === "interactions" && panel.structurePath);
  if (profile?.kind === "interactions" && profile.structurePath) {
    add("Show contacts", {
      kind: "viewer",
      request: { path: profile.structurePath, action: { kind: "contacts", type: null }, source: "answer" },
    });
  }

  const has = (pattern: RegExp) => tools.some((name) => pattern.test(name));
  if (has(/^get_top_candidates$/)) add("Open candidates", { kind: "workbench", tab: "candidates" });
  if (has(/^(run|start|launch)_|_status$/)) {
    add("Open candidates", { kind: "workbench", tab: "candidates" });
    add("Open jobs", { kind: "workbench", tab: "jobs" });
  }
  if (panels.some((panel) => panel.kind === "properties") || has(/^predict_(admet|retrodmta)$/)) {
    add("Open ADMET", { kind: "workbench", tab: "properties" });
  }
  if (panels.some((panel) => panel.kind === "molecule")) add("Open molecules", { kind: "workbench", tab: "molecules" });
  if (has(/_ttps?$/)) add("Open TTP", { kind: "workbench", tab: "ttp" });
  if (has(/^fetch_target_info$/)) add("Open target", { kind: "workbench", tab: "target" });
  if (has(/^list_shared_assets$/)) add("Open assets", { kind: "workbench", tab: "assets" });
  if (panels.some((panel) => panel.kind === "table")) add("Open tables", { kind: "workbench", tab: "tables" });
  return out;
}

export type CalloutTone = "info" | "warn" | "bad" | "ok";

/** GitHub alert markers and the tone each one gets. */
const MARKERS: Record<string, { tone: CalloutTone; title: string }> = {
  NOTE: { tone: "info", title: "Note" },
  IMPORTANT: { tone: "info", title: "Important" },
  TIP: { tone: "ok", title: "Tip" },
  WARNING: { tone: "warn", title: "Warning" },
  CAUTION: { tone: "bad", title: "Caution" },
};

/** The callout a blockquote's leading text asks for, and that text without the marker. */
export function parseCalloutMarker(text: string): { tone: CalloutTone; title: string; rest: string } | null {
  const match = /^\s*\[!(\w+)\][ \t]*\n?/.exec(text);
  const marker = match && MARKERS[match[1].toUpperCase()];
  if (!match || !marker) return null;
  return { ...marker, rest: text.slice(match[0].length) };
}

interface MdNode {
  type: string;
  value?: string;
  children?: MdNode[];
  data?: { hProperties?: Record<string, unknown> };
}

/**
 * Remark plugin: marks every blockquote as a callout, typed by a leading
 * "[!NOTE]"-style marker (which it removes), "info" otherwise.
 */
export function remarkCallouts() {
  return (tree: MdNode) => {
    const walk = (node: MdNode) => {
      if (node.type === "blockquote") {
        let tone: CalloutTone = "info";
        let title: string | null = null;
        const first = node.children?.[0];
        const text = first?.type === "paragraph" ? first.children?.[0] : undefined;
        if (text?.type === "text" && text.value !== undefined) {
          const parsed = parseCalloutMarker(text.value);
          if (parsed) {
            tone = parsed.tone;
            title = parsed.title;
            text.value = parsed.rest;
            if (!parsed.rest && first?.children?.length === 1) node.children = node.children?.slice(1);
            else if (!parsed.rest && first?.children?.[1]?.type === "break") first.children.splice(0, 2);
          }
        }
        node.data = { ...node.data, hProperties: { ...node.data?.hProperties, "data-tone": tone, "data-title": title ?? undefined } };
      }
      node.children?.forEach(walk);
    };
    walk(tree);
  };
}
