// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { afterEach, describe, expect, it, vi } from "vitest";
import {
  confirmText,
  deletableIds,
  formatBytes,
  purgeSession,
  purgeSummary,
  selectionTotals,
  type DataItem,
  type PurgeResult,
  type SessionData,
} from "./sessionData";

const item = (id: string, group: DataItem["group"], bytes: number, deletable: boolean, reason: string | null = null): DataItem => ({
  id,
  group,
  name: id,
  detail: null,
  path: null,
  bytes,
  modified: null,
  url: null,
  links: [],
  onDisk: group !== "ttps",
  deletable,
  reason,
});

const data: SessionData = {
  session: "s-1",
  busy: false,
  groups: [
    { kind: "uploads", label: "Uploads", bytes: 300, items: [item("upload:a", "uploads", 100, true), item("upload:b", "uploads", 200, false, "Also used by 1 other conversation")] },
    { kind: "jobs", label: "REINVENT jobs", bytes: 4096, items: [item("job:1", "jobs", 4096, true), item("job:2", "jobs", 9, false, "Still running: stop the run first")] },
    { kind: "ttps", label: "TTPs", bytes: 0, items: [item("ttp:brd4", "ttps", 0, false, "TTPs are shared and never deleted here")] },
  ],
  totals: { items: 5, bytes: 4405, deletableItems: 2, deletableBytes: 4196, conversationBytes: 2048 },
};

afterEach(() => vi.unstubAllGlobals());

describe("formatBytes", () => {
  it("reads like a file manager", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(1023)).toBe("1023 B");
    expect(formatBytes(812 * 1024)).toBe("812 KB");
    expect(formatBytes(1.4 * 1024 ** 3)).toBe("1.4 GB");
    expect(formatBytes(Number.NaN)).toBe("0 B");
  });
});

describe("selection", () => {
  it("only offers what the gateway marked deletable", () => {
    expect(deletableIds(data)).toEqual(["upload:a", "job:1"]);
    expect(deletableIds(data, "jobs")).toEqual(["job:1"]);
  });
  it("ignores picked items that cannot go", () => {
    expect(selectionTotals(data, new Set(["upload:a", "job:2", "ttp:brd4"]))).toEqual({ count: 1, bytes: 100 });
  });
  it("states the totals before deleting", () => {
    expect(confirmText("BRD4", data, new Set(["upload:a", "job:1"]))).toBe(
      'Delete "BRD4" and 2 items of its data (4.1 KB plus 2.0 KB of conversation record)? This cannot be undone.',
    );
    expect(confirmText("BRD4", data, new Set())).toContain("Its data stays where it is");
  });
});

describe("purge", () => {
  it("posts the picked ids", async () => {
    const fetch = vi.fn(async () => new Response(JSON.stringify({ deleted: "s-1", removed: [], kept: [], freedBytes: 0, conversationBytes: 0 })));
    vi.stubGlobal("fetch", fetch);
    await purgeSession("s-1", ["job:1"]);
    expect(fetch).toHaveBeenCalledWith("/api/sessions/s-1/purge", expect.objectContaining({ method: "POST", body: '{"items":["job:1"]}' }));
  });
  it("summarises what was removed and kept, without counting unpicked items", () => {
    const result: PurgeResult = {
      deleted: "s-1",
      conversationBytes: 1024,
      freedBytes: 1024,
      removed: [{ id: "upload:a", group: "uploads", name: "a", bytes: 1024 }],
      kept: [
        { id: "job:2", group: "jobs", name: "2", bytes: 9, reason: "Still running: stop the run first" },
        { id: "upload:c", group: "uploads", name: "c", bytes: 1, reason: "Not selected" },
      ],
    };
    expect(purgeSummary(result)).toBe("Conversation deleted, 1 item removed, 1 kept; 2.0 KB freed.");
  });
});
