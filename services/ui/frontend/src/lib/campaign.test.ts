// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import type { Campaign, ConversationEvent } from "../types/api";
import { campaignText, campaignView, latestCampaign, relativeTime } from "./campaign";

const EMPTY: Campaign = { goal: null, findings: [], next: null, updated: null };
const at = (minute: number) => new Date(Date.UTC(2026, 9, 29, 10, minute)).toISOString();

const log = (extra: Partial<Campaign>): Campaign => ({ ...EMPTY, ...extra });

describe("latestCampaign", () => {
  it("takes the last campaign event, which carries the whole log", () => {
    const events = [
      { type: "campaign", seq: 1, ts: at(0), campaign: log({ goal: "first" }) },
      { type: "user_message", seq: 2, ts: at(1), content: "hi" },
      { type: "campaign", seq: 3, ts: at(2), campaign: log({ goal: "second" }) },
      { type: "status", seq: 4, ts: at(3), state: "idle" },
    ] as ConversationEvent[];
    expect(latestCampaign(events)?.goal).toBe("second");
    expect(latestCampaign(events.slice(1, 2))).toBeNull();
  });
});

describe("campaignView", () => {
  it("is null for an empty log, so the card stays hidden", () => {
    expect(campaignView(null)).toBeNull();
    expect(campaignView(EMPTY)).toBeNull();
    expect(campaignView(log({ goal: "  ", next: "", findings: [{ ts: at(0), text: " " }] }))).toBeNull();
  });

  it("lists findings newest first", () => {
    const view = campaignView(
      log({
        goal: "Beat JQ1",
        findings: [
          { ts: at(0), text: "a" },
          { ts: at(5), text: "c" },
          { ts: at(2), text: "b" },
        ],
      }),
    );
    expect(view?.findings.map((f) => f.text)).toEqual(["c", "b", "a"]);
  });

  it("keeps the written order, reversed, when times are equal or missing", () => {
    const view = campaignView(log({ findings: [{ ts: "", text: "a" }, { ts: "", text: "b" }] }));
    expect(view?.findings.map((f) => f.text)).toEqual(["b", "a"]);
  });

  it("drops the agent's own tags from what it wrote", () => {
    const view = campaignView(log({ goal: "[AUTO] Beat JQ1", next: "[auto]  Dock the top 5", findings: [{ ts: at(0), text: "[AUTO]" }] }));
    expect(view).toMatchObject({ goal: "Beat JQ1", next: "Dock the top 5", findings: [] });
  });

  it("survives a malformed log", () => {
    const view = campaignView({ goal: "g", findings: null, next: null, updated: null } as unknown as Campaign);
    expect(view).toEqual({ goal: "g", next: null, findings: [], updated: null });
  });
});

describe("relativeTime", () => {
  const now = Date.parse(at(0)) + 3 * 86_400_000;
  it("stays short", () => {
    expect(relativeTime(new Date(now - 20_000).toISOString(), now)).toBe("just now");
    expect(relativeTime(new Date(now - 5 * 60_000).toISOString(), now)).toBe("5 min ago");
    expect(relativeTime(new Date(now - 3 * 3_600_000).toISOString(), now)).toBe("3 h ago");
    expect(relativeTime(at(0), now)).toBe("3 d ago");
    expect(relativeTime(null, now)).toBe("");
    expect(relativeTime("not a time", now)).toBe("");
  });
});

describe("campaignText", () => {
  it("puts the log on a few plain lines", () => {
    const view = campaignView(log({ goal: "Beat JQ1", next: "Dock", findings: [{ ts: at(0), text: "Methyl helps" }] }));
    expect(campaignText(view!)).toBe("Goal: Beat JQ1\nNext: Dock\nFindings:\n- Methyl helps");
  });
});
