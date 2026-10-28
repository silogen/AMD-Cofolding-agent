// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { afterEach, describe, expect, it, vi } from "vitest";
import { rcsbCifUrl, sharedPath, sharedRcsbCif, sliceRegion, uniprotSpans } from "./region";

afterEach(() => vi.unstubAllGlobals());

const REFERENCE = "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQ";

describe("sliceRegion", () => {
  it("cuts UniProt residues, both ends included", () => {
    expect(sliceRegion(REFERENCE, 1, 10)).toEqual({ sequence: "MRPSGTAGAA" });
    expect(sliceRegion(REFERENCE, 61, REFERENCE.length)).toEqual({ sequence: REFERENCE.slice(60) });
  });

  it("rejects ranges outside the sequence, reversed, too short or not whole", () => {
    expect(sliceRegion(REFERENCE, 0, 20)).toEqual({ error: `Residues run from 1 to ${REFERENCE.length}.` });
    expect(sliceRegion(REFERENCE, 5, REFERENCE.length + 1)).toHaveProperty("error");
    expect(sliceRegion(REFERENCE, 30, 20)).toEqual({ error: "The first residue comes after the last." });
    expect(sliceRegion(REFERENCE, 1, 9)).toEqual({ error: "At least 10 residues." });
    expect(sliceRegion(REFERENCE, 1.5, 20)).toHaveProperty("error");
    expect(sliceRegion(REFERENCE, Number(""), 20)).toHaveProperty("error");
  });
});

const MAPPED = `data_1M17
#
loop_
_struct_ref.id
_struct_ref.db_name
_struct_ref.db_code
_struct_ref.pdbx_db_accession
1 UNP EGFR_HUMAN P00533
2 GB  XYZ        AAA111
#
loop_
_struct_ref_seq.align_id
_struct_ref_seq.ref_id
_struct_ref_seq.pdbx_strand_id
_struct_ref_seq.seq_align_beg
_struct_ref_seq.seq_align_end
_struct_ref_seq.pdbx_db_accession
_struct_ref_seq.db_align_beg
_struct_ref_seq.db_align_end
1 1 A 1   200 P00533 671 870
2 1 A 210 333 P00533 880 1003
3 2 B 1   50  AAA111 1   50
#
`;

describe("uniprotSpans", () => {
  it("spans every UniProt segment of a chain and skips other databases", () => {
    const spans = uniprotSpans(MAPPED);
    expect(spans.get("A")).toEqual({ accession: "P00533", start: 671, end: 1003 });
    expect(spans.has("B")).toBe(false);
  });

  it("is empty without the records", () => {
    expect(uniprotSpans("data_x\n_entry.id x\n").size).toBe(0);
  });
});

describe("sharedRcsbCif", () => {
  const stored = (name: string) => Response.json({ path: `/shared/uploads/new/${name}`, url: `/api/artifacts/uploads/new/${name}`, name, size: 9 });

  it("has the gateway keep a copy on the shared volume", async () => {
    const fetch = vi.fn(async (url: string) => (url === "/api/uploads/rcsb" ? stored("1M17.cif") : new Response("data_1M17")));
    vi.stubGlobal("fetch", fetch);
    await expect(sharedRcsbCif(" 1m17 ", "new")).resolves.toEqual({ path: "uploads/new/1M17.cif", text: "data_1M17" });
    expect(fetch).toHaveBeenLastCalledWith("/api/artifacts/uploads/new/1M17.cif");
    expect(fetch).not.toHaveBeenCalledWith(rcsbCifUrl("1M17"));
  });

  it("uploads what the browser fetched when the gateway cannot reach RCSB", async () => {
    const fetch = vi.fn(async (url: string) => {
      if (url === "/api/uploads/rcsb") return Response.json({ detail: "RCSB unreachable" }, { status: 502 });
      if (url.startsWith("/api/uploads?")) return stored("1M17.cif");
      return new Response("data_1M17");
    });
    vi.stubGlobal("fetch", fetch);
    await expect(sharedRcsbCif("1M17", "s 1")).resolves.toEqual({ path: "uploads/new/1M17.cif", text: "data_1M17" });
    expect(fetch).toHaveBeenCalledWith(rcsbCifUrl("1M17"));
    expect(fetch.mock.calls.at(-1)?.[0]).toBe("/api/uploads?session=s+1&name=1M17.cif");
  });

  it("says when the entry does not exist", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("", { status: 404 })));
    await expect(sharedRcsbCif("9ZZZ", "new")).rejects.toThrow("9ZZZ is not in the PDB.");
    await expect(sharedRcsbCif("hello", "new")).rejects.toThrow("not a PDB id");
  });
});

describe("sharedPath", () => {
  it("reads the path from the artifact URL", () => {
    expect(sharedPath({ path: "/shared/uploads/a b/x.cif", url: "/api/artifacts/uploads/a%20b/x.cif", name: "x.cif", size: 1 })).toBe("uploads/a b/x.cif");
  });
});
