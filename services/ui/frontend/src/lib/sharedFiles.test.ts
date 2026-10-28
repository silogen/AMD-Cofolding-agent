// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { afterEach, describe, expect, it, vi } from "vitest";
import { a3mQuery, cifChains, msaMismatch, pdbChains, pdbIdFromPath, readA3mQuery, readCif, readSeedFile, structureChains, targetFiles } from "./sharedFiles";

afterEach(() => vi.unstubAllGlobals());

const atom = (serial: number, resname: string, chain: string, resnum: number, record = "ATOM  ") =>
  `${record}${String(serial).padStart(5)}  CA  ${resname.padEnd(3)} ${chain}${String(resnum).padStart(4)}    ${"0.000".padStart(8)}${"0.000".padStart(8)}${"0.000".padStart(8)}  1.00  0.00           C`;

const KINASE = ["MET", "GLY", "SER", "LYS", "ALA", "LEU", "VAL", "ILE", "GLU", "ASP", "PHE", "TRP"];

describe("structure chains", () => {
  it("reads PDB SEQRES, grouping identical chains and skipping DNA", () => {
    const text = [
      "HEADER    TRANSFERASE",
      `SEQRES   1 A   12  ${KINASE.slice(0, 6).join(" ")}`,
      `SEQRES   2 A   12  ${KINASE.slice(6).join(" ")}`,
      `SEQRES   1 B   12  ${KINASE.join(" ")}`,
      "SEQRES   1 C   12   DA  DT  DG  DC  DA  DT  DG  DC  DA  DT  DG  DC",
      atom(1, "MET", "A", 1),
      "END",
    ].join("\n");
    expect(pdbChains(text)).toEqual([{ ids: ["A", "B"], sequence: "MGSKALVIEDFW", source: "construct" }]);
  });

  it("falls back to the first model's residues, keeping modified ones and dropping ligands", () => {
    const lines = KINASE.map((res, i) => atom(i + 1, res === "MET" ? "MSE" : res, "A", i + 1, res === "MET" ? "HETATM" : "ATOM  "));
    const text = [...lines, atom(20, "HOH", "A", 200, "HETATM"), atom(21, "ATP", "A", 201, "HETATM"), "ENDMDL", atom(30, "GLY", "B", 1)].join("\n");
    expect(pdbChains(text)).toEqual([{ ids: ["A"], sequence: "MGSKALVIEDFW", source: "atoms" }]);
  });

  const CIF = `data_5UF0
#
_entry.id 5UF0
#
loop_
_entity_poly.entity_id
_entity_poly.type
_entity_poly.pdbx_seq_one_letter_code_can
_entity_poly.pdbx_strand_id
1 'polypeptide(L)'
;MGSKALVIED
FWMGSKALVI
;
A,B
2 polydeoxyribonucleotide ATGCATGCATGC C
#
_struct_asym.id A
_struct_asym.entity_id 1
`;

  it("reads mmCIF entity_poly, with quoted values and text fields", () => {
    expect(cifChains(CIF)).toEqual([{ ids: ["A", "B"], sequence: "MGSKALVIEDFWMGSKALVI", source: "construct" }]);
    expect(readCif(CIF, ["entry"]).get("entry")).toEqual([{ id: "5UF0" }]);
    expect(structureChains("x/5uf0.cif", CIF)).toHaveLength(1);
  });

  it("uses entity_poly_seq, then atom_site, when the one-letter code is missing", () => {
    const seq = `data_x
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
${KINASE.map((r, i) => `1 ${i + 1} ${r}`).join("\n")}
loop_
_struct_asym.id
_struct_asym.entity_id
A 1
`;
    expect(cifChains(seq)).toEqual([{ ids: ["A"], sequence: "MGSKALVIEDFW", source: "construct" }]);
    const atoms = `data_y
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_model_num
${KINASE.flatMap((r, i) => [`ATOM ${2 * i + 1} ${r} A ${i + 1} H ${i + 1} 1`, `ATOM ${2 * i + 2} ${r} A ${i + 1} H ${i + 1} 1`]).join("\n")}
HETATM 99 HOH B . W 1 1
ATOM 100 GLY A 1 H 1 2
`;
    expect(cifChains(atoms)).toEqual([{ ids: ["H"], sequence: "MGSKALVIEDFW", source: "atoms" }]);
  });

  it("names PDB ids from file names", () => {
    expect(pdbIdFromPath("uploads/new/5uf0.cif")).toBe("5UF0");
    expect(pdbIdFromPath("structures/1M17_protein.pdb")).toBe("1M17");
    expect(pdbIdFromPath("structures/egfr_kinase.pdb")).toBeNull();
  });
});

describe("MSAs", () => {
  it("reads the query as Boltz does", () => {
    expect(a3mQuery(">101\nMGS-KA\nLVIE\n>hit\nMGSKALVIE\n")).toBe("MGSKALVIE");
    expect(a3mQuery("")).toBeNull();
  });

  it("warns when the MSA was built for another sequence", () => {
    expect(msaMismatch("MGSKA", "MGSKA")).toBeNull();
    expect(msaMismatch(null, "MGSKA")).toBeNull();
    expect(msaMismatch("M".repeat(1210), "M".repeat(330))).toBe("The MSA is for a 1,210 aa sequence, not this 330 aa one: Boltz would fold the MSA's sequence.");
    expect(msaMismatch("MGSKA", "MGSKV")).toMatch(/differs/);
  });

  it("streams only the first record of a shared .a3m", async () => {
    let pulled = 0;
    const chunks = [">query\nMGSKA", "LVIE\n>hit1\nMGS", ...Array.from({ length: 6 }, () => "KALVIE\n>hit\nMGS".repeat(1000))];
    const body = new ReadableStream<Uint8Array>({
      pull(controller) {
        if (pulled < chunks.length) controller.enqueue(new TextEncoder().encode(chunks[pulled++]));
        else controller.close();
      },
    });
    const fetcher = vi.fn(async () => new Response(body));
    vi.stubGlobal("fetch", fetcher);
    expect(await readA3mQuery("/shared/msas/egfr.a3m")).toBe("MGSKALVIE");
    expect(fetcher).toHaveBeenCalledWith("/api/artifacts/msas/egfr.a3m");
    expect(pulled).toBeLessThan(chunks.length);
  });

  it("lists input structures and a3m files, not Boltz outputs", () => {
    const files = targetFiles({
      structures: ["uploads/new/5UF0.cif", "predictions/out_1/x.pdb"],
      cofold_outputs: ["predictions/out_1/x.pdb"],
      msas: ["msas/egfr.a3m", "msas/egfr.sto"],
      msa_details: [{ path: "msas/egfr.a3m", query_length: 330 }],
    });
    expect(files.structures.map((a) => a.path)).toEqual(["uploads/new/5UF0.cif"]);
    expect(files.msas.map((a) => [a.path, a.queryLength])).toEqual([["msas/egfr.a3m", 330]]);
  });
});

describe("seed files", () => {
  it("reads SMILES from a shared .smi or .csv file", async () => {
    const fetcher = vi.fn(async (url: string) => new Response(url.endsWith(".csv") ? "name,SMILES\na,CCO\nb,c1ccccc1\n" : "CCO ethanol\nCCN\n"));
    vi.stubGlobal("fetch", fetcher);
    expect(await readSeedFile("seeds/hits.csv")).toEqual({ smiles: ["CCO", "c1ccccc1"], failed: 0 });
    expect(await readSeedFile("/shared/seeds/hits.smi")).toEqual({ smiles: ["CCO", "CCN"], failed: 0 });
    expect(fetcher).toHaveBeenLastCalledWith("/api/artifacts/seeds/hits.smi");
  });

  it("explains files it cannot use", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("nope", { status: 404 })));
    await expect(readSeedFile("seeds/missing.smi")).rejects.toThrow(/not found/);
    await expect(readSeedFile("seeds/notes.txt")).rejects.toThrow(/\.smi, \.csv or \.sdf/);
    await expect(readSeedFile("/etc/passwd.smi")).rejects.toThrow(/not on the shared volume/);
  });
});
