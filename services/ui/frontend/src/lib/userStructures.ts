// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useSyncExternalStore } from "react";
import type { StructurePanel } from "../types/api";
import { sameFile, type Artifacts, type Sourced } from "./artifacts";

/**
 * Structures the user brought in (an uploaded file, a PDB entry), as
 * opposed to ones a tool produced. Kept per conversation in the browser,
 * so they stay in the viewer's list after a reload.
 */
const KEY = "cofold.structures:";
const listeners = new Set<() => void>();
const cache = new Map<string, Sourced<StructurePanel>[]>();
const EMPTY: Sourced<StructurePanel>[] = [];

function read(session: string): Sourced<StructurePanel>[] {
  const cached = cache.get(session);
  if (cached) return cached;
  let list: Sourced<StructurePanel>[] = EMPTY;
  try {
    const raw = localStorage.getItem(KEY + session);
    if (raw) list = JSON.parse(raw) as Sourced<StructurePanel>[];
  } catch {
    list = EMPTY;
  }
  cache.set(session, list);
  return list;
}

/** Add a structure to a conversation ("new" before it has an id). Adding the same file again moves it first. */
export function addUserStructure(session: string, panel: StructurePanel, tool = panel.tool): void {
  const list = [{ item: panel, tool, at: Date.now() }, ...read(session).filter(({ item }) => !sameFile(item.path, panel.path))];
  cache.set(session, list);
  try {
    localStorage.setItem(KEY + session, JSON.stringify(list));
  } catch {
    // Storage full or blocked: the structure stays for this page only.
  }
  listeners.forEach((listener) => listener());
}

export function useUserStructures(session: string): Sourced<StructurePanel>[] {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => read(session),
  );
}

/** The conversation's structures with the user's own after them, without duplicates. */
export function withUserStructures(artifacts: Artifacts, extra: Sourced<StructurePanel>[]): Artifacts {
  if (!extra.length) return artifacts;
  const fresh = extra.filter(({ item }) => !artifacts.structures.some((s) => sameFile(s.item.path, item.path)));
  return { ...artifacts, structures: [...artifacts.structures, ...fresh] };
}
