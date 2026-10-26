// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { poseKey, posebustersCheck, poseSiblings, stepPose, structureKind, structureOption } from "./poses";

const run = "predictions/out_7a/boltz_results_batch_7a/predictions";
const paths = [
  `${run}/mol_0/mol_0_model_1.pdb`,
  `${run}/mol_0/mol_0_model_0.pdb`,
  `${run}/mol_1/mol_1_model_0.pdb`,
  `${run}/mol_1/mol_1_model_10.pdb`,
  `${run}/mol_1/mol_1_model_2.pdb`,
  "targets/4lr6.pdb",
];

describe("poseKey", () => {
  it("strips the sample number", () => {
    expect(poseKey(`${run}/mol_0/mol_0_model_3.pdb`)).toEqual({ key: `${run}/mol_0/mol_0`, sample: 3 });
    expect(poseKey("targets/4lr6.pdb")).toEqual({ key: "targets/4lr6", sample: null });
  });
});

describe("poseSiblings", () => {
  it("returns the samples of the same molecule in order", () => {
    expect(poseSiblings(paths, `${run}/mol_1/mol_1_model_2.pdb`)).toEqual({
      paths: [`${run}/mol_1/mol_1_model_0.pdb`, `${run}/mol_1/mol_1_model_2.pdb`, `${run}/mol_1/mol_1_model_10.pdb`],
      grouped: true,
    });
  });
  it("falls back to every structure without siblings", () => {
    expect(poseSiblings(paths, "targets/4lr6.pdb")).toEqual({ paths, grouped: false });
    expect(poseSiblings([`${run}/mol_0/mol_0_model_0.pdb`], `${run}/mol_0/mol_0_model_0.pdb`).grouped).toBe(false);
  });
});

describe("stepPose", () => {
  it("wraps around", () => {
    expect(stepPose(["a", "b", "c"], "c", 1)).toBe("a");
    expect(stepPose(["a", "b", "c"], "a", -1)).toBe("c");
    expect(stepPose([], "a", 1)).toBeNull();
  });
});

describe("structureKind", () => {
  it("infers the kind from the path", () => {
    expect(structureKind(`${run}/mol_0/mol_0_model_0.pdb`)).toBe("predicted");
    expect(structureKind("targets/4LR6.pdb")).toBe("experimental");
    expect(structureKind("rcsb/brd4.cif")).toBe("experimental");
    expect(structureKind("structures/brd4_prepared.pdb")).toBe("prepared");
    expect(structureKind("prep/brd4/complex.pdb")).toBe("prepared");
    expect(structureKind("structures/brd4.pdb")).toBeNull();
    expect(structureKind("structures/brd4.pdb", true)).toBe("predicted");
  });
  it("trusts the tool over the path", () => {
    expect(structureKind("uploads/x/entry.cif", false, "rcsb")).toBe("experimental");
  });
});

describe("structureOption", () => {
  it("names the run, and the pose after the first", () => {
    expect(structureOption("JQ1", `${run}/mol_0/mol_0_model_0.pdb`, "7aa793db")).toBe("JQ1 · 7aa793db");
    expect(structureOption("JQ1", `${run}/mol_0/mol_0_model_2.pdb`, "7aa793db")).toBe("JQ1 · 7aa793db · pose 3");
    expect(structureOption("3MXF", "rcsb/3MXF.cif", null)).toBe("3MXF");
  });
});

describe("posebustersCheck", () => {
  it("explains known checks and humanises the rest", () => {
    expect(posebustersCheck("sanitization")).toBe("Ligand chemistry is valid");
    expect(posebustersCheck("All atoms connected")).toBe("All ligand atoms are connected");
    expect(posebustersCheck("some_new_check")).toBe("Some new check");
  });
});
