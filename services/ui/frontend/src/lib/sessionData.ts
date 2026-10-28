// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { request } from "./api";

/**
 * What one conversation produced or used, and deleting it with its data
 * (reinvent4_agent/web/session_data.py). The gateway decides what may be
 * deleted; the browser only picks among the items it marked deletable.
 */

export type DataGroupKind = "uploads" | "predictions" | "structures" | "analyses" | "jobs" | "ttps";

export interface DataLink {
  label: string;
  url: string;
}

export interface DataItem {
  id: string;
  group: DataGroupKind;
  name: string;
  detail: string | null;
  path: string | null;
  bytes: number;
  /** ISO 8601. */
  modified: string | null;
  url: string | null;
  links: DataLink[];
  onDisk: boolean;
  deletable: boolean;
  /** Why it is kept, when it is not deletable. */
  reason: string | null;
  state?: string | null;
  withConversation?: boolean;
}

export interface DataGroup {
  kind: DataGroupKind;
  label: string;
  items: DataItem[];
  bytes: number;
}

export interface SessionData {
  session: string;
  busy: boolean;
  groups: DataGroup[];
  totals: {
    items: number;
    bytes: number;
    deletableItems: number;
    deletableBytes: number;
    conversationBytes: number;
  };
}

export interface PurgeEntry {
  id: string;
  group: DataGroupKind | null;
  name: string;
  bytes: number;
  reason?: string;
}

export interface PurgeResult {
  deleted: string;
  conversationBytes: number;
  removed: PurgeEntry[];
  kept: PurgeEntry[];
  freedBytes: number;
  error?: string;
}

export const fetchSessionData = (id: string) => request<SessionData>(`/api/sessions/${encodeURIComponent(id)}/data`);

export const purgeSession = (id: string, items: string[]) =>
  request<PurgeResult>(`/api/sessions/${encodeURIComponent(id)}/purge`, {
    method: "POST",
    body: JSON.stringify({ items }),
  });

const UNITS = ["B", "KB", "MB", "GB", "TB"];

/** A size the way a file manager shows it: 0 B, 812 KB, 1.4 GB. */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const power = Math.min(UNITS.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  const value = bytes / 1024 ** power;
  return `${power === 0 ? value : value < 10 ? value.toFixed(1) : Math.round(value)} ${UNITS[power]}`;
}

export function allItems(data: SessionData): DataItem[] {
  return data.groups.flatMap((group) => group.items);
}

export function deletableIds(data: SessionData, group?: DataGroupKind): string[] {
  return allItems(data)
    .filter((item) => item.deletable && (!group || item.group === group))
    .map((item) => item.id);
}

/** How many of the picked items can go, and how much space that frees. */
export function selectionTotals(data: SessionData, picked: ReadonlySet<string>): { count: number; bytes: number } {
  const chosen = allItems(data).filter((item) => item.deletable && picked.has(item.id));
  return { count: chosen.length, bytes: chosen.reduce((sum, item) => sum + item.bytes, 0) };
}

/** The confirmation question, with what will be removed. */
export function confirmText(title: string, data: SessionData, picked: ReadonlySet<string>): string {
  const { count, bytes } = selectionTotals(data, picked);
  const record = formatBytes(data.totals.conversationBytes);
  if (count === 0) return `Delete "${title}" (${record} of conversation record)? Its data stays where it is. This cannot be undone.`;
  return `Delete "${title}" and ${count} item${count === 1 ? "" : "s"} of its data (${formatBytes(bytes)} plus ${record} of conversation record)? This cannot be undone.`;
}

/** One line on what the deletion did. */
export function purgeSummary(result: PurgeResult): string {
  const freed = formatBytes(result.freedBytes + result.conversationBytes);
  const removed = result.removed.length;
  const kept = result.kept.filter((entry) => entry.reason !== "Not selected").length;
  const parts = [`Conversation deleted`, `${removed} item${removed === 1 ? "" : "s"} removed`];
  if (kept) parts.push(`${kept} kept`);
  return `${parts.join(", ")}; ${freed} freed.`;
}

export function shortDate(iso: string | null): string {
  if (!iso) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleDateString([], { day: "numeric", month: "short", year: "numeric" });
}
