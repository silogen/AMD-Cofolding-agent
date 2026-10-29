// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { Campaign } from "../../types/api";
import { CampaignCard } from "./CampaignCard";

const at = (minute: number) => new Date(Date.UTC(2026, 9, 29, 10, minute)).toISOString();

const LOG: Campaign = {
  goal: "Find a BRD4 binder that beats JQ1 on affinity",
  findings: [
    { ts: at(0), text: "Finding one" },
    { ts: at(1), text: "Finding two" },
    { ts: at(2), text: "Finding three" },
    { ts: at(3), text: "Finding four" },
  ],
  next: "Cofold the top 5 with Boltz",
  updated: at(3),
};

const render = (campaign: Campaign | null, defaultOpen = false) => renderToStaticMarkup(<CampaignCard campaign={campaign} defaultOpen={defaultOpen} />);

describe("CampaignCard", () => {
  it("renders nothing while the log is empty", () => {
    expect(render(null)).toBe("");
    expect(render({ goal: null, findings: [], next: null, updated: null })).toBe("");
  });

  it("shows the goal, the next step and the two latest findings, newest first", () => {
    const html = render(LOG);
    expect(html).toContain(LOG.goal!);
    expect(html).toContain(LOG.next!);
    expect(html.indexOf("Finding four")).toBeGreaterThan(-1);
    expect(html.indexOf("Finding four")).toBeLessThan(html.indexOf("Finding three"));
    expect(html).not.toContain("Finding two");
    expect(html).toContain("+2");
    expect(html).toContain('aria-expanded="false"');
  });

  it("lists every finding with when it was written once unfolded", () => {
    const html = render(LOG, true);
    for (const finding of LOG.findings) expect(html).toContain(finding.text);
    expect(html).toContain(" ago");
    expect(html).toContain('aria-expanded="true"');
  });

  it("never names the machinery behind it", () => {
    for (const open of [false, true]) {
      // Text and tooltips, not the class names.
      const seen = render({ ...LOG, goal: "[AUTO] Beat JQ1" }, open)
        .replace(/ class="[^"]*"/g, "")
        .toLowerCase();
      for (const word of ["auto", "budget", "cycle"]) expect(seen).not.toContain(word);
    }
  });
});
