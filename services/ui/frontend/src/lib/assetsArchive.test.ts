// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { clearToolCache } from "./api";
import { archiveError, downloadArchive, listFiles } from "./assetsArchive";

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

beforeEach(() => clearToolCache());
afterEach(() => vi.unstubAllGlobals());

describe("listFiles", () => {
  it("reads every file from the gateway", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => json({ files: [{ path: "a.sdf", size: 3, mtime: 1 }], truncated: false })));
    expect(await listFiles()).toEqual({ files: [{ path: "a.sdf", size: 3, mtime: 1 }], partial: false, truncated: false });
  });

  it("falls back to list_shared_assets on a gateway without /api/files", async () => {
    const fetcher = vi.fn(async (path: string) =>
      path === "/api/files" ? json({ detail: "Not Found" }, 404) : json({ name: "list_shared_assets", result: { structures: ["t/a.pdb"], msas: [] }, panels: [] }),
    );
    vi.stubGlobal("fetch", fetcher);
    const listing = await listFiles();
    expect(listing.partial).toBe(true);
    expect(listing.files.map((f) => f.path)).toEqual(["t/a.pdb"]);
  });
});

describe("downloadArchive", () => {
  it("explains a gateway that has no archive route yet", () => {
    expect(archiveError(404, "Not Found")).toMatch(/updated gateway/);
    expect(archiveError(404, "Not found or not readable: x.pdb")).toBe("Not found or not readable: x.pdb");
    expect(archiveError(413, "")).toBe("The archive failed (HTTP 413).");
  });

  it("posts the paths and name, and reports the gateway's refusal", async () => {
    const fetcher = vi.fn(async () => json({ detail: "5001 files asked for; one archive holds at most 5000" }, 413));
    vi.stubGlobal("fetch", fetcher);
    await expect(downloadArchive(["a.sdf"], "all-sdf")).rejects.toThrow("at most 5000");
    expect(fetcher).toHaveBeenCalledWith("/api/archive", expect.objectContaining({ method: "POST", body: JSON.stringify({ paths: ["a.sdf"], name: "all-sdf" }) }));
  });
});
