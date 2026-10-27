// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { beforeEach, describe, expect, it, vi } from "vitest";

const store = new Map<string, string>();
vi.stubGlobal("localStorage", {
  getItem: (k: string) => store.get(k) ?? null,
  setItem: (k: string, v: string) => void store.set(k, v),
  removeItem: (k: string) => void store.delete(k),
});

const { clearShortlist, isStarred, removeStar, setShortlistSession, shortlistColumns, shortlistPrompt, shortlistSdf, shortlistSession, shortlistSheet, toggleStar } =
  await import("./shortlist");

describe("shortlist", () => {
  beforeEach(() => {
    clearShortlist("s1");
    clearShortlist("s2");
  });

  it("stars and unstars per conversation, in the browser's storage", () => {
    expect(toggleStar("s1", { smiles: "CCO", values: { Score: 0.8 } })).toBe(true);
    expect(isStarred("s1", "CCO")).toBe(true);
    expect(isStarred("s2", "CCO")).toBe(false);
    expect(JSON.parse(store.get("cofold.shortlist:s1")!)[0]).toMatchObject({ smiles: "CCO", values: { Score: 0.8 } });
    expect(toggleStar("s1", { smiles: "CCO" })).toBe(false);
    expect(store.has("cofold.shortlist:s1")).toBe(false);
  });

  it("removes one entry", () => {
    toggleStar("s1", { smiles: "CCO" });
    toggleStar("s1", { smiles: "CCN" });
    removeStar("s1", "CCO");
    expect(isStarred("s1", "CCO")).toBe(false);
    expect(isStarred("s1", "CCN")).toBe(true);
  });

  it("follows the current conversation", () => {
    setShortlistSession("s2");
    expect(shortlistSession()).toBe("s2");
  });

  it("exports stored and newer values, Score first", () => {
    const entries = [
      { smiles: "CCO", label: "ethanol", source: "3814ad67", values: { QED: 0.4, Score: 0.7 }, at: 1 },
      { smiles: "CCN", at: 2 },
    ];
    const known = (s: string) => (s === "CCN" ? { MW: 45.1 } : { Score: 0.9 });
    expect(shortlistColumns(entries, known)).toEqual(["Score", "QED", "MW"]);
    const sheet = shortlistSheet(entries, known);
    expect(sheet.columns).toEqual(["smiles", "label", "source", "Score", "QED", "MW"]);
    expect(sheet.rows[0]).toEqual({ smiles: "CCO", label: "ethanol", source: "3814ad67", Score: 0.9, QED: 0.4, MW: null });
    expect(sheet.rows[1]).toMatchObject({ smiles: "CCN", MW: 45.1, Score: null });
    const sdf = shortlistSdf(entries, () => "\n  RDKit\n\n  0  0  0  0  0  0  0  0  0  0999 V2000\nM  END", known);
    expect(sdf.match(/\$\$\$\$/g)).toHaveLength(2);
    expect(sdf).toContain("ethanol");
  });

  it("writes a prompt with the SMILES, one per line", () => {
    const prompt = shortlistPrompt([
      { smiles: "CCO", at: 1 },
      { smiles: "CCN", at: 2 },
    ]);
    expect(prompt.startsWith("Cofold these 2 molecules")).toBe(true);
    expect(prompt.endsWith("\nCCO\nCCN")).toBe(true);
  });
});
