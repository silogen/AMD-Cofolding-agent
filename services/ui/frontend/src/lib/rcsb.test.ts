// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { afterEach, describe, expect, it, vi } from "vitest";

const added = vi.hoisted(() => [] as unknown[]);
vi.mock("./userStructures", () => ({ addUserStructure: (_session: string, panel: unknown) => added.push(panel) }));
vi.mock("./viewerBus", () => ({ sendToViewer: () => undefined }));

import { fetchPdbEntry, isPdbId } from "./rcsb";

const respond = (status: number, body: unknown = {}) =>
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(body), { status, statusText: `status ${status}` })));

afterEach(() => {
  vi.unstubAllGlobals();
  added.length = 0;
});

describe("isPdbId", () => {
  it("accepts four characters starting with a digit", () => {
    expect(isPdbId("3MXF")).toBe(true);
    expect(isPdbId(" 1abc ")).toBe(true);
    expect(isPdbId("BRD4")).toBe(false);
    expect(isPdbId("3MXFA")).toBe(false);
  });
});

describe("fetchPdbEntry", () => {
  it("uses the copy the gateway made", async () => {
    respond(200, { path: "uploads/s-1/3MXF.cif", url: "/files/uploads/s-1/3MXF.cif" });
    const panel = await fetchPdbEntry("3mxf", "s-1");
    expect(panel).toMatchObject({ tool: "rcsb", path: "uploads/s-1/3MXF.cif", format: "cif" });
    expect(added).toEqual([panel]);
  });
  it("falls back to RCSB when the gateway has no endpoint", async () => {
    respond(404);
    const panel = await fetchPdbEntry("3mxf", "s-1");
    expect(panel).toMatchObject({ path: "rcsb/3MXF.cif", url: "https://files.rcsb.org/download/3MXF.cif" });
  });
  it("reports other failures", async () => {
    respond(502, { detail: "RCSB is down" });
    await expect(fetchPdbEntry("3MXF", "s-1")).rejects.toThrow("Could not fetch 3MXF: RCSB is down");
    await expect(fetchPdbEntry("BRD4", "s-1")).rejects.toThrow("not a PDB id");
    expect(added).toEqual([]);
  });
});
