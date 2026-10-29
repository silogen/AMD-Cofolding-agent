// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { moleculesOf } from "../../lib/runAnalysis";
import { parseRunResults } from "../../lib/runResults";
import { SelectionTable } from "../RunAnalysis/SelectionTable";
import { MoleculeTable } from "./MoleculeTable";

describe("MoleculeTable", () => {
  it("puts the SMILES on the drawing's tooltip and a copy button beside it, not in a column", () => {
    const html = renderToStaticMarkup(
      <MoleculeTable
        columns={[{ key: "Score", label: "Total", strong: true }, { key: "MW", label: "MW", unit: "Da" }]}
        rows={[{ smiles: "CCO", cells: { Score: { text: "0.82" }, MW: { text: "46", score: 0.4 } }, complex: "Show the complex" }]}
        theme="light"
        onOpenStructure={() => {}}
      />,
    );
    expect(html).not.toContain(">SMILES<");
    expect(html).not.toContain(">CCO<");
    expect(html).toContain('title="CCO"');
    expect(html).toContain('aria-label="Copy SMILES"');
    expect(html).toContain(">3D</button>");
    expect(html).toContain(">Da<");
    expect(html).toContain("width:40%");
  });

  it("draws a text column as a name over a faint line, and a footer in a box that grows with its parent", () => {
    const html = renderToStaticMarkup(
      <MoleculeTable
        columns={[{ key: "name", label: "Name", text: true }]}
        rows={[{ smiles: "CCO", cells: { name: { text: "ethanol", sub: "Scored · ADMET" } } }]}
        theme="light"
        maxHeight={null}
        footer={<p>Show more</p>}
      />,
    );
    expect(html).toContain('<span class="block truncate font-medium">ethanol</span>');
    expect(html).toContain(">Scored · ADMET<");
    expect(html).toContain("<p>Show more</p>");
    expect(html).not.toContain("max-height");
  });

  it("leaves the 3D column out when nothing has a complex", () => {
    const html = renderToStaticMarkup(<MoleculeTable columns={[]} rows={[{ smiles: "CCO", cells: {} }]} theme="light" onOpenStructure={() => {}} />);
    expect(html).not.toContain("Complex");
    expect(html).not.toContain(">3D<");
  });
});

describe("SelectionTable", () => {
  it("draws the analysis molecules without a SMILES column", () => {
    const molecules = moleculesOf(parseRunResults(["Score,SMILES,SMILES_state,QED,QED (raw),step", "0.5,CCO,1,0.6,0.6,1", "0.7,CCN,1,0.8,0.8,2"].join("\n")));
    const html = renderToStaticMarkup(
      <SelectionTable molecules={molecules} mask={null} columns={molecules.endpoints} colours={new Map()} hovered={null} onHover={() => {}} onOpen={() => {}} theme="light" />,
    );
    expect(html).not.toContain(">SMILES<");
    expect(html.indexOf('title="CCN"')).toBeLessThan(html.indexOf('title="CCO"'));
    expect(html).toContain(">Step<");
    expect(html).toContain(">0.70<");
  });
});
