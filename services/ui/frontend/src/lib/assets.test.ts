// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { assetKind, assetUrl, canOpen, groupAssets, groupByFolder, topFolder, mergeAssets, parseSharedAssets, servedPath, sessionAssets, sharedRelative } from "./assets";

const listing = {
  shared_dir: "/shared",
  structures: ["targets/brd4.pdb", "predictions/run1/complex_model_0.cif"],
  msas: ["msa/brd4_bd1.a3m"],
  msa_details: [{ path: "msa/brd4_bd1.a3m", query_length: 127 }],
  cofold_outputs: ["predictions/run1/complex_model_0.cif"],
  counts: { structures: 2, msas: 1, cofold_outputs: 1 },
};

describe("parseSharedAssets", () => {
  it("lists each file once, cofold outputs apart", () => {
    const assets = parseSharedAssets(JSON.stringify(listing));
    expect(assets.map((a) => [a.path, a.kind])).toEqual([
      ["targets/brd4.pdb", "structure"],
      ["predictions/run1/complex_model_0.cif", "cofold"],
      ["msa/brd4_bd1.a3m", "sequence"],
    ]);
    expect(assets[2].queryLength).toBe(127);
    expect(assets[0]).toMatchObject({ name: "brd4.pdb", dir: "targets", url: "/api/artifacts/targets/brd4.pdb", source: "shared" });
  });

  it("is empty for anything else", () => {
    expect(parseSharedAssets("nope")).toEqual([]);
    expect(parseSharedAssets({ note: "empty" })).toEqual([]);
  });
});

describe("paths", () => {
  it("classifies by extension", () => {
    expect(assetKind("a/b.CIF")).toBe("structure");
    expect(assetKind("x/predictions/y.pdb")).toBe("cofold");
    expect(assetKind("seeds.smi")).toBe("smiles");
    expect(assetKind("config.toml")).toBe("config");
    expect(assetKind("README")).toBe("other");
  });

  it("serves shared and job files only", () => {
    expect(sharedRelative("/shared/a/b.pdb")).toBe("a/b.pdb");
    expect(sharedRelative("/etc/passwd")).toBeNull();
    expect(sharedRelative("../x.pdb")).toBeNull();
    expect(assetUrl("dir/my file.pdb")).toBe("/api/artifacts/dir/my%20file.pdb");
    expect(assetUrl("/data/reinvent4_mcp/jobs/job_d0cbeb98/config.toml")).toBe("/api/jobs/d0cbeb98/files/config.toml");
    expect(assetUrl("/data/elsewhere/x.csv")).toBeNull();
    expect(servedPath("results.csv")).toBeNull();
    expect(servedPath("/shared/out/a.csv")).toBe("out/a.csv");
  });

  it("opens structures on the shared volume", () => {
    const [structure, , msa] = parseSharedAssets(listing);
    expect(canOpen(structure)).toBe(true);
    expect(canOpen(msa)).toBe(false);
  });
});

describe("session files", () => {
  it("collects files named by tool calls and results", () => {
    const assets = sessionAssets([
      { type: "tool_call", name: "profile_interactions", arguments: { structure_paths: ["/shared/predictions/run1/complex_model_0.cif"] } },
      {
        type: "tool_result",
        name: "run_optimization",
        result: JSON.stringify({ job_dir: "/data/reinvent4_mcp/jobs/job_d0cbeb98", config: "/data/reinvent4_mcp/jobs/job_d0cbeb98/config.toml", note: "wrote results.csv" }),
      },
      { type: "tool_result", name: "failed", result: "seeds/x.smi", isError: true },
    ]);
    expect(assets.map((a) => [a.path, a.kind, a.tool])).toEqual([
      ["predictions/run1/complex_model_0.cif", "cofold", "profile_interactions"],
      ["/data/reinvent4_mcp/jobs/job_d0cbeb98/config.toml", "config", "run_optimization"],
    ]);
  });

  it("merges without duplicates and groups by kind", () => {
    const shared = parseSharedAssets(listing);
    const session = sessionAssets([{ type: "tool_call", name: "t", arguments: { p: "/shared/targets/brd4.pdb", q: "seeds/lib.smi" } }]);
    const all = mergeAssets(shared, session);
    expect(all).toHaveLength(4);
    const groups = groupAssets(all);
    expect(groups.map((g) => [g.kind, g.assets.length])).toEqual([
      ["structure", 1],
      ["cofold", 1],
      ["sequence", 1],
      ["smiles", 1],
    ]);
    expect(groupAssets(all, "brd4").flatMap((g) => g.assets.map((a) => a.name))).toEqual(["brd4.pdb", "brd4_bd1.a3m"]);
  });
});

describe("groupByFolder", () => {
  it("groups by top-level folder, sorted", () => {
    const assets = parseSharedAssets(listing);
    const groups = groupByFolder(assets);
    expect(groups.map((g) => g.folder)).toEqual(["/msa", "/predictions", "/targets"]);
    expect(groupByFolder(assets, "brd4").map((g) => g.folder)).toEqual(["/msa", "/targets"]);
  });
  it("puts root files under /", () => {
    expect(topFolder("x.pdb")).toBe("/");
    expect(topFolder("/shared/structures/a/b.pdb")).toBe("/structures");
  });
});