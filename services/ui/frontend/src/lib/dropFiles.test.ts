// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { fileKind, parseCsv, parseSmi, splitSdf } from "./dropFiles";

const MOLBLOCK = (title: string) => `${title}
     RDKit          2D

  2  1  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.2990    0.7500    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
M  END`;

describe("fileKind", () => {
  it("sorts files into molecule lists and structures", () => {
    expect(["a.smi", "b.CSV", "c.sdf", "d.mol", "e.pdb", "f.cif", "g.mmcif", "h.txt", "noext"].map(fileKind)).toEqual([
      "smi",
      "csv",
      "sdf",
      "sdf",
      "structure",
      "structure",
      "structure",
      null,
      null,
    ]);
  });
});

describe("parseSmi", () => {
  it("takes the first token of each line and skips headers and comments", () => {
    expect(parseSmi("SMILES name\nCCO ethanol\r\n# comment\n\nc1ccccc1\tbenzene\n")).toEqual(["CCO", "c1ccccc1"]);
  });
});

describe("parseCsv", () => {
  it("reads the SMILES column by name", () => {
    expect(parseCsv('name,SMILES,score\naspirin,"CC(=O)Oc1ccccc1C(=O)O",0.5\nethanol,CCO,0.1\n')).toEqual([
      "CC(=O)Oc1ccccc1C(=O)O",
      "CCO",
    ]);
  });

  it("finds an unnamed SMILES column, with or without a header", () => {
    expect(parseCsv("id;structure\nm1;CCO\nm2;CCN\n")).toEqual(["CCO", "CCN"]);
    expect(parseCsv("m1,CCO\nm2,CCN\n")).toEqual(["CCO", "CCN"]);
  });

  it("gives nothing for a CSV without molecules", () => {
    expect(parseCsv("a,b\n1,2\n")).toEqual([]);
  });
});

describe("splitSdf", () => {
  it("splits records and keeps empty titles", () => {
    const blocks = splitSdf(`${MOLBLOCK("first")}\n> <score>\n1.5\n\n$$$$\n${MOLBLOCK("")}\n$$$$\n`);
    expect(blocks).toHaveLength(2);
    expect(blocks[0].startsWith("first\n")).toBe(true);
    expect(blocks[0].endsWith("M  END")).toBe(true);
    expect(blocks[1].startsWith("\n     RDKit")).toBe(true);
  });

  it("reads a single .mol file", () => {
    expect(splitSdf(MOLBLOCK("x"))).toHaveLength(1);
  });
});
