// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { unified } from "unified";
import remarkParse from "remark-parse";
import type { Panel, StructurePanel } from "../types/api";
import { answerTitle, artifactLinks, firstHeading, parseCalloutMarker, remarkCallouts, withoutLeadingHeading } from "./answer";

const structure = (path: string, confidence?: number): StructurePanel => ({
  kind: "structure",
  tool: "score_molecules",
  name: path,
  path,
  url: `/files/${path}`,
  format: "cif",
  metrics: confidence === undefined ? undefined : { confidence },
});

describe("answerTitle", () => {
  it("uses the first heading of the answer", () => {
    expect(answerTitle(["Intro\n\n## Pose **quality** check\n\ntext"], ["score_molecules"])).toBe("Pose quality check");
  });

  it("falls back to the main tool, skipping context lookups", () => {
    expect(answerTitle(["no heading"], ["get_ttp", "score_molecules"])).toBe("Cofolding and checking pose quality");
    expect(answerTitle([], ["profile_interactions"])).toBe("Profiling interactions");
    expect(answerTitle([], ["predict_admet"])).toBe("Predicting ADMET");
    expect(answerTitle([], ["run_optimization"])).toBe("Running REINVENT");
    expect(answerTitle([], ["get_ttp"])).toBe("Reading the therapeutic target profile");
  });

  it("says Answer when nothing tells", () => {
    expect(answerTitle(["hello"], ["mystery_tool"])).toBe("Answer");
  });
});

describe("headings", () => {
  it("reads a heading without markup", () => {
    expect(firstHeading("# [Link](http://x) `code`")).toBe("Link code");
    expect(firstHeading("plain")).toBeNull();
  });

  it("drops a leading heading only when it is the title", () => {
    expect(withoutLeadingHeading("## Result\n\nBody", "Result")).toBe("Body");
    expect(withoutLeadingHeading("Body\n## Result", "Result")).toBe("Body\n## Result");
  });
});

describe("artifactLinks", () => {
  it("opens the most confident pose and its contacts", () => {
    const panels: Panel[] = [
      structure("a.cif", 0.4),
      structure("b.cif", 0.8),
      { kind: "interactions", tool: "profile_interactions", name: null, structurePath: "b.cif", smiles: null, error: null, counts: {}, contacts: [], totalContacts: 0 },
    ];
    const links = artifactLinks(panels, ["score_molecules", "profile_interactions"]);
    expect(links.map((link) => link.label)).toEqual(["Open best pose", "Show contacts"]);
    expect(links[0].target).toEqual({ kind: "viewer", request: { path: "b.cif", source: "answer" } });
  });

  it("does not link tables", () => {
    const table: Panel = { kind: "table", tool: "get_top_candidates", columns: ["smiles"], rows: [{ smiles: "CCO" }], totalRows: 1, truncated: false };
    expect(artifactLinks([table], ["list_jobs"])).toEqual([]);
  });

  it("links runs, ADMET and the TTP to workbench tabs", () => {
    const links = artifactLinks([], ["run_optimization", "predict_admet", "get_ttp"]);
    expect(links.map((link) => [link.label, link.target.kind === "workbench" ? link.target.tab : null])).toEqual([
      ["Open candidates", "candidates"],
      ["Open jobs", "jobs"],
      ["Open ADMET", "properties"],
      ["Open TTP", "ttp"],
    ]);
  });
});

describe("callouts", () => {
  it("parses GitHub alert markers", () => {
    expect(parseCalloutMarker("[!WARNING]\nCareful")).toEqual({ tone: "warn", title: "Warning", rest: "Careful" });
    expect(parseCalloutMarker("[!caution] x")?.tone).toBe("bad");
    expect(parseCalloutMarker("[!BOGUS] x")).toBeNull();
    expect(parseCalloutMarker("plain")).toBeNull();
  });

  it("marks blockquotes with their tone and strips the marker", () => {
    const run = (markdown: string) => {
      const tree = unified().use(remarkParse).parse(markdown);
      remarkCallouts()(tree as never);
      return (tree as unknown as { children: { data?: { hProperties?: Record<string, unknown> }; children: { children: { value?: string }[] }[] }[] }).children[0];
    };
    const typed = run("> [!TIP]\n> Use a bigger box");
    expect(typed.data?.hProperties).toMatchObject({ "data-tone": "ok", "data-title": "Tip" });
    expect(typed.children[0].children[0].value).toBe("Use a bigger box");
    expect(run("> just a note").data?.hProperties).toMatchObject({ "data-tone": "info" });
  });
});
