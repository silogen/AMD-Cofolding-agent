// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useSyncExternalStore } from "react";

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

// -- Backup ------------------------------------------------------------------

const FORMAT = "cofold.shortlist";
const FORMAT_VERSION = 1;

/** What an exported shortlist file holds; the shortlist lives in the browser, so this is its only backup. */
export interface ShortlistFile {
  format: typeof FORMAT;
  version: number;
  exported: string;
  session?: string;
  entries: ShortlistEntry[];
}

export function shortlistJson(entries: readonly ShortlistEntry[], session?: string): string {
  const file: ShortlistFile = { format: FORMAT, version: FORMAT_VERSION, exported: new Date().toISOString(), session, entries: [...entries] };
  return JSON.stringify(file, null, 2);
}

function entryOf(value: unknown): ShortlistEntry | null {
  if (!value || typeof value !== "object") return null;
  const raw = value as Record<string, unknown>;
  const smiles = typeof raw.smiles === "string" ? raw.smiles.trim() : "";
  if (!smiles) return null;
  const entry: ShortlistEntry = { smiles, at: typeof raw.at === "number" && Number.isFinite(raw.at) ? raw.at : Date.now() };
  if (typeof raw.label === "string" && raw.label) entry.label = raw.label;
  if (typeof raw.source === "string" && raw.source) entry.source = raw.source;
  if (raw.values && typeof raw.values === "object") {
    const values: Record<string, number> = {};
    for (const [k, v] of Object.entries(raw.values as Record<string, unknown>)) if (typeof v === "number" && Number.isFinite(v)) values[k] = v;
    if (Object.keys(values).length) entry.values = values;
  }
  return entry;
}

/**
 * The entries of an exported shortlist, a bare array of entries also being
 * accepted. Throws a message meant for the user when the file is not one.
 */
export function parseShortlistFile(text: string): ShortlistEntry[] {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error("That file is not valid JSON");
  }
  let list: unknown;
  if (Array.isArray(parsed)) {
    list = parsed;
  } else if (parsed && typeof parsed === "object") {
    const file = parsed as Record<string, unknown>;
    if (typeof file.format === "string" && file.format !== FORMAT) throw new Error("That JSON file is not a shortlist");
    list = file.entries;
  }
  if (!Array.isArray(list)) throw new Error("That JSON file is not a shortlist");
  const seen = new Set<string>();
  const entries: ShortlistEntry[] = [];
  for (const value of list) {
    const entry = entryOf(value);
    if (entry && !seen.has(entry.smiles)) {
      seen.add(entry.smiles);
      entries.push(entry);
    }
  }
  if (entries.length === 0) throw new Error("That shortlist file holds no molecules");
  return entries;
}

/** Add the entries missing from a conversation's shortlist, molecules already starred staying as they are. */
export function mergeShortlist(session: string, incoming: readonly ShortlistEntry[]): { added: number; skipped: number } {
  const list = read(session);
  const seen = new Set(list.map((e) => e.smiles));
  const added: ShortlistEntry[] = [];
  for (const entry of incoming) {
    if (seen.has(entry.smiles)) continue;
    seen.add(entry.smiles);
    added.push(entry);
  }
  if (added.length) write(session, [...list, ...added]);
  return { added: added.length, skipped: incoming.length - added.length };
}

// -- Prompts -----------------------------------------------------------------

/** A composer prompt for the agent; the user edits and sends it. */
export function shortlistPrompt(entries: readonly ShortlistEntry[]): string {
  const n = entries.length;
  return `Cofold ${n === 1 ? "this molecule" : `these ${n} molecules`} against the target and compare their poses and affinities:\n${entries.map((e) => e.smiles).join("\n")}`;
}

export function admetPrompt(entries: readonly ShortlistEntry[]): string {
  const n = entries.length;
  return `Predict ADMET for ${n === 1 ? "this molecule" : `these ${n} molecules`}: ${entries.map((e) => e.smiles).join(", ")}`;
}

/** The starred SMILES as Mol2Mol seeds, one per line. */
export const shortlistSeeds = (entries: readonly ShortlistEntry[]) => entries.map((e) => e.smiles).join("\n");
