// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { DetailsColumns } from "./DetailsColumns";

const files = [
  { label: "Structure", type: "PDB", url: "/files/a.pdb", name: "a.pdb" },
  { label: "MSA", type: "A3M", url: "/files/a.a3m", name: "a.a3m" },
];

describe("DetailsColumns", () => {
  it("puts the files first, then cofold and predictions before the properties", () => {
    const html = renderToStaticMarkup(
      <DetailsColumns
        files={files}
        cofold={[{ label: "Confidence", value: "0.82", bar: { fraction: 0.82, tone: "ok" } }]}
        predictions={[{ title: "ADMET", facts: [{ label: "hERG", value: "0.12" }] }]}
        properties={[{ label: "MW", value: "412.6", unit: "Da" }]}
      />,
    );
    const order = ['aria-label="Files"', 'aria-label="Cofold"', 'aria-label="ADMET"', 'aria-label="Properties"'].map((mark) => html.indexOf(mark));
    expect(order.every((at) => at >= 0)).toBe(true);
    expect([...order].sort((a, b) => a - b)).toEqual(order);
    expect(html).toContain('download="a.a3m"');
    expect(html).toContain("width:82%");
    expect(html).not.toContain("/files/a.pdb</");
  });

  it("shows only the files when nothing else is known", () => {
    const html = renderToStaticMarkup(<DetailsColumns files={files} cofold={[]} predictions={[]} properties={[]} />);
    expect(html).toContain("Structure");
    expect(html).not.toContain("<section");
  });
});
