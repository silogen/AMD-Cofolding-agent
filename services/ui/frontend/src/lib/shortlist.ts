// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useSyncExternalStore } from "react";
import { toSdf, type Cell, type Sheet } from "./csv";

/**
 * Molecules the user starred, per conversation, kept in the browser. An entry
 * remembers the scores it had where it was starred (a run's row, a candidate),
 * so the shortlist can show them after the source scrolls out of reach.
 */
export interface ShortlistEntry {
  smiles: string;
  label?: string;
  /** Where it was starred: a run id, "candidates", a tool. */
  source?: string;
  /** Scores and properties known when it was starred, by name. */
  values?: Record<string, number>;
  at: number;
}

const KEY = "cofold.shortlist:";
const listeners = new Set<() => void>();
const cache = new Map<string, ShortlistEntry[]>();
const EMPTY: ShortlistEntry[] = [];
let current = "new";

const notify = () => listeners.forEach((listener) => listener());

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function read(session: string): ShortlistEntry[] {
  const cached = cache.get(session);
  if (cached) return cached;
  let list: ShortlistEntry[] = EMPTY;
  try {
    const raw = localStorage.getItem(KEY + session);
    const parsed: unknown = raw ? JSON.parse(raw) : null;
    if (Array.isArray(parsed)) list = parsed.filter((e): e is ShortlistEntry => !!e && typeof e.smiles === "string");
  } catch {
    list = EMPTY;
  }
  cache.set(session, list);
  return list;
}

function write(session: string, list: ShortlistEntry[]) {
  cache.set(session, list);
  try {
    if (list.length) localStorage.setItem(KEY + session, JSON.stringify(list));
    else localStorage.removeItem(KEY + session);
  } catch {
    // Storage full or blocked: the shortlist stays for this page only.
  }
  notify();
}

/** The conversation stars go to where no session is passed ("new" before it has an id). */
export function setShortlistSession(session: string): void {
  if (session === current) return;
  current = session;
  notify();
}

export const shortlistSession = () => current;

export function isStarred(session: string, smiles: string): boolean {
  return read(session).some((e) => e.smiles === smiles);
}

/** Star a molecule, or unstar it when it is already on the list. Returns whether it is starred now. */
export function toggleStar(session: string, entry: Omit<ShortlistEntry, "at">): boolean {
  const list = read(session);
  if (list.some((e) => e.smiles === entry.smiles)) {
    write(session, list.filter((e) => e.smiles !== entry.smiles));
    return false;
  }
  write(session, [...list, { ...entry, at: Date.now() }]);
  return true;
}

export function removeStar(session: string, smiles: string): void {
  write(session, read(session).filter((e) => e.smiles !== smiles));
}

export function clearShortlist(session: string): void {
  write(session, []);
}

export function useShortlistSession(): string {
  return useSyncExternalStore(subscribe, shortlistSession);
}

export function useShortlist(session: string): ShortlistEntry[] {
  return useSyncExternalStore(subscribe, () => read(session));
}

// -- Views -------------------------------------------------------------------

/** What else is known about a molecule: a candidate's scores, RDKit descriptors. */
export type KnownValues = (smiles: string) => Record<string, number | undefined> | undefined;

/** Value names over all entries, the total score first. */
export function shortlistColumns(entries: readonly ShortlistEntry[], known?: KnownValues): string[] {
  const names = new Set<string>();
  for (const e of entries) {
    for (const [k, v] of Object.entries({ ...e.values, ...known?.(e.smiles) })) if (typeof v === "number") names.add(k);
  }
  return [...names].sort((a, b) => (a === "Score" ? -1 : b === "Score" ? 1 : 0));
}

/** Each entry's values, newer knowledge winning over what was stored when starred. */
export function shortlistValues(entry: ShortlistEntry, known?: KnownValues): Record<string, number> {
  const out: Record<string, number> = { ...entry.values };
  for (const [k, v] of Object.entries(known?.(entry.smiles) ?? {})) if (typeof v === "number" && Number.isFinite(v)) out[k] = v;
  return out;
}

export function shortlistSheet(entries: readonly ShortlistEntry[], known?: KnownValues): Sheet {
  const values = shortlistColumns(entries, known);
  return {
    columns: ["smiles", "label", "source", ...values],
    rows: entries.map((e) => {
      const v = shortlistValues(e, known);
      const row: Record<string, Cell> = { smiles: e.smiles, label: e.label ?? null, source: e.source ?? null };
      for (const name of values) row[name] = v[name] ?? null;
      return row;
    }),
  };
}

export function shortlistSdf(entries: readonly ShortlistEntry[], molblock: (smiles: string) => string | null, known?: KnownValues): string {
  const sheet = shortlistSheet(entries, known);
  return toSdf(entries.map((e, i) => ({ molblock: molblock(e.smiles), title: e.label ?? `shortlist ${i + 1}`, fields: sheet.rows[i] })));
}

/** A composer prompt for the agent; the user edits and sends it. */
export function shortlistPrompt(entries: readonly ShortlistEntry[]): string {
  const n = entries.length;
  return `Cofold ${n === 1 ? "this molecule" : `these ${n} molecules`} against the target and compare their poses and affinities:\n${entries.map((e) => e.smiles).join("\n")}`;
}
