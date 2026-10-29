// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { Campaign, CampaignFinding, ConversationEvent } from "../types/api";

/** The campaign log as the card shows it; findings newest first. */
export interface CampaignView {
  goal: string | null;
  next: string | null;
  findings: CampaignFinding[];
  updated: string | null;
}

/** The current log of a list of events: the last campaign event carries all of it. */
export function latestCampaign(events: readonly ConversationEvent[]): Campaign | null {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (event.type === "campaign") return event.campaign;
  }
  return null;
}

/** The agent's own tags ("[AUTO] ..."), which are bookkeeping and not for the reader. */
function clean(text: unknown): string | null {
  if (typeof text !== "string") return null;
  const stripped = text.replace(/\[auto\]\s*/gi, "").trim();
  return stripped || null;
}

/** What there is to show of *campaign*, or null when the log is empty. */
export function campaignView(campaign: Campaign | null | undefined): CampaignView | null {
  if (!campaign) return null;
  const listed = Array.isArray(campaign.findings) ? campaign.findings : [];
  const findings = listed
    .map((finding, order) => ({ ts: typeof finding?.ts === "string" ? finding.ts : "", text: clean(finding?.text), order }))
    .filter((finding): finding is { ts: string; text: string; order: number } => finding.text !== null)
    .sort((a, b) => (Date.parse(b.ts) || 0) - (Date.parse(a.ts) || 0) || b.order - a.order)
    .map(({ ts, text }) => ({ ts, text }));
  const goal = clean(campaign.goal);
  const next = clean(campaign.next);
  if (!goal && !next && findings.length === 0) return null;
  return { goal, next, findings, updated: typeof campaign.updated === "string" ? campaign.updated : null };
}

/** "just now", "5 min ago", "3 h ago", "2 d ago"; empty for a missing time. */
export function relativeTime(ts: string | null, now: number): string {
  const at = ts ? Date.parse(ts) : NaN;
  if (Number.isNaN(at)) return "";
  const minutes = Math.floor(Math.max(0, now - at) / 60_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  return `${Math.floor(hours / 24)} d ago`;
}

/** The log as plain text, for the clipboard. */
export function campaignText(view: CampaignView): string {
  const lines: string[] = [];
  if (view.goal) lines.push(`Goal: ${view.goal}`);
  if (view.next) lines.push(`Next: ${view.next}`);
  if (view.findings.length) lines.push("Findings:", ...view.findings.map((finding) => `- ${finding.text}`));
  return lines.join("\n");
}
