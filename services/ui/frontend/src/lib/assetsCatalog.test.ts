// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";

import { sessionAssets } from "./assets";
import { archiveName, archivePath, buildCatalog, fileLabel, fileType, groupFiles, groupOf, inScope, ofType, totalBytes, typeCounts } from "./assetsCatalog";
import type { ToolEvent } from "./ttp";

const RUN = "predictions/out_7aa793db23f9/boltz_results_batch_7aa793db23f9/predictions/mol_0";

describe("fileType", () => {
  it("tells predictions from input structures, and keeps molecules, tables and MSAs apart", () => {
    expect(fileType(`${RUN}/mol_0_model_0.pdb`)).toBe("prediction");
    expect(fileType(`${RUN}/mol_0_model_0.cif`)).toBe("prediction");
    expect(fileType(`${RUN}/confidence_mol_0_model_0.json`)).toBe("prediction");
    expect(fileType(`${RUN}/affinity_mol_0.json`)).toBe("prediction");
    expect(fileType("uploads/s1/brd4.cif")).toBe("structure");
    expect(fileType("batches/b1/ligands.sdf")).toBe("sdf");
    expect(fileType("hits.csv")).toBe("table");
    expect(fileType("seeds.smi")).toBe("table");
    expect(fileType("reports/run.PDF")).toBe("pdf");
    expect(fileType("msa_cache/brd4.a3m")).toBe("msa");
    expect(fileType("/data/jobs/job_abc123/reinvent.log")).toBe("config");
  });

  it("puts engine internals in Other, the raw arrays of a prediction included", () => {
    for (const path of [
      `${RUN}/pae_mol_0_model_0.npz`,
      `${RUN}/plddt_mol_0_model_0.npz`,
      "predictions/out_1/processed/manifest.json",
      "msa_cache/brd4.pkl",
      "msa_cache/brd4.lock",
      "batches/b1/input.yaml",
      "batches/b1/records.json",
    ])
      expect(fileType(path), path).toBe("other");
  });
});

describe("groupOf", () => {
  it("groups by run, job, uploads, or top-level folder", () => {
    expect(groupOf(`${RUN}/mol_0_model_0.pdb`)).toEqual({ key: "predictions/out_7aa793db23f9", label: "Cofold 7aa793db", run: true });
    expect(groupOf("/data/jobs/job_abcdef123456/results_1.csv")).toEqual({ key: "job_abcdef123456", label: "Job abcdef12", run: true });
    expect(groupOf("/shared/uploads/s1/x.pdb")).toEqual({ key: "uploads", label: "Uploads", run: true });
    expect(groupOf("targets/a/b.pdb")).toEqual({ key: "targets", label: "targets", run: false });
    expect(groupOf("top.pdb")).toEqual({ key: "", label: "Top level", run: false });
  });
});

describe("fileLabel", () => {
  it("names Boltz outputs by molecule and model, and cuts long hashes", () => {
    expect(fileLabel(`${RUN}/mol_0_model_0.pdb`)).toBe("mol_0 \u00b7 model 0");
    expect(fileLabel(`${RUN}/confidence_mol_0_model_1.json`)).toBe("mol_0 \u00b7 model 1 \u00b7 confidence");
    expect(fileLabel(`${RUN}/affinity_mol_0.json`)).toBe("mol_0 \u00b7 affinity");
    expect(fileLabel("msas/0123456789abcdef0123.a3m")).toBe("01234567\u2026");
    expect(fileLabel("targets/brd4.pdb")).toBe("brd4");
  });
});

describe("archivePath", () => {
  it("is shared-relative, or job_<id>/<name> for a job file, or null", () => {
    expect(archivePath("/shared/a/b.pdb")).toBe("a/b.pdb");
    expect(archivePath("/data/reinvent4_mcp/jobs/job_abc123/results_1.csv")).toBe("job_abc123/results_1.csv");
    expect(archivePath("/etc/passwd")).toBeNull();
  });
});

describe("catalog", () => {
  const events: ToolEvent[] = [
    { type: "tool_call", name: "profile_interactions", arguments: { structure_paths: [`/shared/${RUN}/mol_0_model_0.pdb`] } },
    { type: "tool_call", name: "read_results", arguments: { path: "/data/jobs/job_abc123/results_1.csv" } },
  ];
  const listed = [
    { path: `${RUN}/mol_0_model_0.pdb`, size: 1000, mtime: 10 },
    { path: `${RUN}/mol_0_model_1.pdb`, size: 1000, mtime: 10 },
    { path: `${RUN}/pae_mol_0_model_0.npz`, size: 9000, mtime: 10 },
    { path: "targets/brd4.pdb", size: 500, mtime: 99 },
    { path: "ligands.sdf", size: 20, mtime: 5 },
  ];
  const catalog = buildCatalog(listed, sessionAssets(events));

  it("adds the job files only the chat named, and puts whole runs it touched in the chat", () => {
    expect(catalog.map((f) => [f.path, f.inChat])).toEqual([
      [`${RUN}/mol_0_model_0.pdb`, true],
      [`${RUN}/mol_0_model_1.pdb`, true],
      [`${RUN}/pae_mol_0_model_0.npz`, true],
      ["targets/brd4.pdb", false],
      ["ligands.sdf", false],
      ["/data/jobs/job_abc123/results_1.csv", true],
    ]);
    expect(catalog.find((f) => f.type === "table")?.archivePath).toBe("job_abc123/results_1.csv");
    expect(catalog.filter((f) => f.openable).map((f) => f.path)).toHaveLength(3);
  });

  it("filters by scope and search, and counts the types left", () => {
    const chat = inScope(catalog, { scope: "chat", query: "" });
    expect(chat).toHaveLength(4);
    expect(typeCounts(chat)).toEqual([
      { type: "prediction", label: "Predictions", count: 2 },
      { type: "table", label: "CSV", count: 1 },
      { type: "other", label: "Other", count: 1 },
    ]);
    expect(inScope(catalog, { scope: "all", query: "model 1" }).map((f) => f.path)).toEqual([`${RUN}/mol_0_model_1.pdb`]);
    expect(ofType(inScope(catalog, { scope: "all", query: "" }), "sdf").map((f) => f.path)).toEqual(["ligands.sdf"]);
  });

  it("leaves Other out unless it is asked for, downloads included", () => {
    const all = inScope(catalog, { scope: "all", query: "" });
    expect(ofType(all, null).some((f) => f.type === "other")).toBe(false);
    expect(totalBytes(ofType(all, null))).toBe(2520);
    expect(ofType(all, "other").map((f) => f.ext)).toEqual(["npz"]);
    expect(totalBytes([])).toBeNull();
  });

  it("lists the chat's groups first, then the newest, with sizes", () => {
    const groups = groupFiles(ofType(catalog, null));
    expect(groups.map((g) => [g.label, g.files.length, g.bytes])).toEqual([
      ["Cofold 7aa793db", 2, 2000],
      ["Job abc123", 1, null],
      ["targets", 1, 500],
      ["Top level", 1, 20],
    ]);
  });

  it("names the archive after the filter", () => {
    expect(archiveName({ scope: "all", type: "prediction", query: "" })).toBe("all-predictions");
    expect(archiveName({ scope: "chat", type: null, query: "BRD4 run" })).toBe("chat-brd4-run-files");
    expect(archiveName({ scope: "all", type: "config", query: "" })).toBe("all-run-logs");
  });
});
