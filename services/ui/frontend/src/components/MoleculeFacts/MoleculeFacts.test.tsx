// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { MoleculeFacts } from "./MoleculeFacts";

describe("MoleculeFacts", () => {
  it("shows cofold, predictions and score before the properties, one bar row per value", () => {
    const html = renderToStaticMarkup(
      <MoleculeFacts
        score={0.49}
        components={[{ label: "QED", value: "0.29" }]}
        cofold={{ confidence: 0.82, posebustersPassed: 18, posebustersTotal: 20 }}
        onOpenComplex={() => {}}
        predictions={[{ title: "ADMET", facts: [{ label: "hERG", value: "0.12" }, { label: "LogD", value: "2.10" }] }]}
        properties={[{ label: "MW", value: "612.4", unit: "Da" }]}
      />,
    );
    const order = ['aria-label="Cofold"', 'aria-label="ADMET"', 'aria-label="Score"', 'aria-label="Properties"'].map((mark) => html.indexOf(mark));
    expect(order.every((at) => at >= 0)).toBe(true);
    expect([...order].sort((a, b) => a - b)).toEqual(order);
    expect(html).toContain(">3D</button>");
    expect(html).toContain("width:49%");
    expect(html).toContain("width:82%");
    expect(html).toContain("width:12%");
    expect(html).toContain("color:var(--color-warn)");
    expect(html.match(/<dt/g)).toHaveLength(7);
  });

  it("keeps the cofold heading for its 3D button when the complex came without values", () => {
    const html = renderToStaticMarkup(<MoleculeFacts onOpenComplex={() => {}} />);
    expect(html).toContain('aria-label="Cofold"');
    expect(html).not.toContain("<dt");
  });

  it("renders nothing when nothing is known", () => {
    expect(renderToStaticMarkup(<MoleculeFacts />)).toBe("");
  });
});
