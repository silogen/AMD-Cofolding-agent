// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { profileError } from "./profileError";
import { boltzOutput, chainSequences, fastaFile, inputFiles, parseBoltzInput } from "./structureFiles";

const MODEL = "predictions/out_2cc15dd06b46/boltz_results_batch_2cc15dd06b46/predictions/mol_2/mol_2_model_0.pdb";

describe("boltzOutput", () => {
  it("names the files next to a Boltz model and its input", () => {
    const out = boltzOutput(`/shared/${MODEL}`);
    expect(out?.candidates.map((file) => [file.label, file.name])).toEqual([
      ["Ligand", "mol_2_model_0_lig.sdf"],
      ["Confidence", "confidence_mol_2_model_0.json"],
      ["Affinity", "affinity_mol_2.json"],
    ]);
    expect(out?.input.url).toBe("/api/artifacts/batches/batch_2cc15dd06b46/mol_2.yaml");
  });

  it("is null for other structures", () => {
    expect(boltzOutput("uploads/kinases/3POZ_A.cif")).toBeNull();
  });
});

describe("parseBoltzInput", () => {
  const YAML = [
    "version: 1",
    "sequences:",
    "  - protein:",
    "      id: [A]",
    "      sequence: MKVLAAGIVG",
    "      msa: /shared/msa_cache/abc.a3m",
    "  - ligand:",
    "      id: B",
    "      smiles: 'CCO'",
    "templates:",
    "  - cif: /shared/uploads/kinases/4ZC9_A.cif",
    "properties:",
    "  - affinity:",
    "      binder: B",
  ].join("\n");

  it("reads proteins, MSAs and templates", () => {
    const input = parseBoltzInput(YAML);
    expect(input).toEqual({ proteins: [{ id: "A", sequence: "MKVLAAGIVG" }], msas: ["/shared/msa_cache/abc.a3m"], templates: ["/shared/uploads/kinases/4ZC9_A.cif"] });
    expect(inputFiles(input).map((file) => [file.label, file.url])).toEqual([
      ["MSA", "/api/artifacts/msa_cache/abc.a3m"],
      ["Template", "/api/artifacts/uploads/kinases/4ZC9_A.cif"],
    ]);
  });
});

describe("FASTA", () => {
  it("writes the protein chains of a structure", () => {
    const residues = [..."MKVLAAGIVGW"].map((letter, i) => ({
      chain: "A",
      resnum: i + 1,
      resname: { M: "MET", K: "LYS", V: "VAL", L: "LEU", A: "ALA", G: "GLY", I: "ILE", W: "TRP" }[letter] as string,
    }));
    const chains = chainSequences([...residues, { chain: "B", resnum: 1, resname: "ALA" }]);
    expect(chains).toEqual([{ id: "A", sequence: "MKVLAAGIVGW" }]);
    const file = fastaFile("3POZ_A", chains);
    expect(file?.name).toBe("3POZ_A.fasta");
    expect(decodeURIComponent(file?.url.split(",")[1] ?? "")).toBe(">3POZ_A|A\nMKVLAAGIVGW\n");
  });
});

describe("profileError", () => {
  it("states a missing ligand quietly", () => {
    expect(profileError("no ligand in this structure (protein chain A only)")).toEqual({ text: "No ligand in this structure", quiet: true });
    expect(profileError("ValueError: no small-molecule ligand detected")).toEqual({ text: "No ligand in this structure", quiet: true });
    expect(profileError("RuntimeError: file not found")).toEqual({ text: "file not found", quiet: false });
  });
});
