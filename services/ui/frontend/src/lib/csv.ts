// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { PropertiesPanel } from "../types/api";
import type { Candidate } from "./jobs";

/**
 * Tables as files: CSV (RFC 4180) for downloads, tab-separated text for the
 * clipboard (it pastes into a spreadsheet as cells), and SD files for
 * molecules.
 */

export type Cell = string | number | boolean | null | undefined;

const NEEDS_QUOTES = /[",\r\n]|^\s|\s$/;

/** One CSV field: quoted when it holds a comma, quote, line break or edge space. */
export function csvCell(value: Cell): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "";
  const text = String(value);
  return NEEDS_QUOTES.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

/** A header row and one row per record, CRLF-separated. */
export function toCsv(columns: string[], rows: Record<string, Cell>[]): string {
  const lines = [columns.map(csvCell).join(",")];
  for (const row of rows) lines.push(columns.map((column) => csvCell(row[column])).join(","));
  return `${lines.join("\r\n")}\r\n`;
}

/** Tab-separated text: tabs and line breaks inside a value become spaces. */
export function toTsv(columns: string[], rows: Record<string, Cell>[]): string {
  const cell = (value: Cell) =>
    value === null || value === undefined || (typeof value === "number" && !Number.isFinite(value)) ? "" : String(value).replace(/[\t\r\n]+/g, " ");
  return [columns.map(cell).join("\t"), ...rows.map((row) => columns.map((column) => cell(row[column])).join("\t"))].join("\n");
}

/** "brd4 top candidates" -> "brd4-top-candidates.csv". */
export function fileName(base: string, extension: string): string {
  const slug = base
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return `${slug || "table"}.${extension}`;
}

export interface Sheet {
  columns: string[];
  rows: Record<string, Cell>[];
}

/** ADMET / RetroDMTA predictions: mean and spread per endpoint, one row per molecule. */
export function propertiesSheet(panel: PropertiesPanel): Sheet {
  const columns = ["smiles", "label"];
  for (const endpoint of panel.endpoints) {
    const unit = panel.molecules.map((m) => m.values[endpoint]?.unit).find(Boolean);
    columns.push(unit ? `${endpoint} (${unit})` : endpoint, `${endpoint}_std`);
  }
  columns.push("ad_pass", "error");
  const rows = panel.molecules.map((molecule) => {
    const row: Record<string, Cell> = { smiles: molecule.smiles, label: molecule.label, ad_pass: molecule.adPass, error: molecule.error };
    panel.endpoints.forEach((endpoint, i) => {
      const value = molecule.values[endpoint];
      row[columns[2 + 2 * i]] = value?.mean;
      row[`${endpoint}_std`] = value?.std;
    });
    return row;
  });
  return { columns, rows };
}

/** Top candidates of a run: total score, then each component's transformed and raw score. */
export function candidatesSheet(candidates: Candidate[]): Sheet {
  const names: string[] = [];
  for (const c of candidates) for (const component of c.components) if (!names.includes(component.name)) names.push(component.name);
  const columns = ["rank", "smiles", "total_score", ...names.flatMap((name) => [name, `${name} (raw)`])];
  const rows = candidates.map((c, i) => {
    const row: Record<string, Cell> = { rank: i + 1, smiles: c.smiles, total_score: c.total };
    for (const component of c.components) {
      row[component.name] = component.transformed;
      row[`${component.name} (raw)`] = component.raw;
    }
    return row;
  });
  return { columns, rows };
}

/** Top candidates as SD records titled "candidate N", with the SMILES, rank and scores as data fields. */
export function candidatesSdf(candidates: Candidate[], molblock: (smiles: string) => string | null): string {
  const { columns, rows } = candidatesSheet(candidates);
  return toSdf(
    candidates.map((c, i) => ({
      molblock: molblock(c.smiles),
      title: `candidate ${i + 1}`,
      fields: Object.fromEntries(columns.map((column) => [column, rows[i][column]])),
    })),
  );
}

/**
 * An SD file from molblocks and their data fields. Records without a
 * molblock (unparsable SMILES) are skipped.
 */
export function toSdf(records: { molblock: string | null; title?: string; fields: Record<string, Cell> }[]): string {
  return records
    .filter((r): r is typeof r & { molblock: string } => Boolean(r.molblock))
    .map((record) => {
      const lines = record.molblock.replace(/\r\n/g, "\n").replace(/\n+$/, "").split("\n");
      if (record.title !== undefined) lines[0] = record.title.replace(/[\r\n]+/g, " ");
      const block = lines.join("\n");
      const data = Object.entries(record.fields)
        .filter(([, value]) => value !== null && value !== undefined && !(typeof value === "number" && !Number.isFinite(value)))
        .map(([key, value]) => `> <${key.replace(/[<>\r\n]/g, "_")}>\n${String(value).replace(/[\r\n]+/g, " ")}\n`)
        .join("\n");
      return `${block}\n${data}${data ? "\n" : ""}$$$$\n`;
    })
    .join("");
}
