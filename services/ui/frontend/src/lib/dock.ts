// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/** Pure helpers for the workbench dock: the open tabs (like an editor) and its width. */

export const DOCK_MIN_WIDTH = 380;
export const DOCK_MAX_FRACTION = 0.78;
export const DOCK_DEFAULT_WIDTH = "clamp(520px, 56vw, calc(100vw - 680px))";

const TABS_KEY = "cofold.dock.tabs";
const WIDTH_KEY = "cofold.dock.width";

/** The tabs the dock can open. */
export type DockTab = "structure" | "molecules" | "tables" | "jobs" | "results" | "target" | "ttp" | "assets";
/** What can be asked for: a tab, or a view inside one (e.g. "properties" is a view of the molecules tab). */
export type DockView = DockTab | "candidates" | "properties" | "shortlist" | "activity";

export const DOCK_TABS: readonly DockTab[] = ["structure", "molecules", "tables", "jobs", "results", "target", "ttp", "assets"];

/** Views that used to be tabs of their own; saved tab lists still name them. */
const MERGED: Partial<Record<DockView, DockTab>> = {
  candidates: "molecules",
  properties: "molecules",
  shortlist: "molecules",
  activity: "jobs",
  results: "jobs",
};

/** The tab that shows `view`. */
export function dockTabOf(view: DockView): DockTab {
  return MERGED[view] ?? (view as DockTab);
}

type Storage = Pick<globalThis.Storage, "getItem" | "setItem">;

function storage(): Storage | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null;
  }
}

/** Add `id` to the open tabs if it is not there yet (appended at the end). */
export function openTab<T extends string>(tabs: readonly T[], id: T): T[] {
  return tabs.includes(id) ? [...tabs] : [...tabs, id];
}

/**
 * Remove `id` from the open tabs. When it was the active one, the neighbour
 * becomes active: the tab to its right, else the one to its left, else none.
 */
export function closeTab<T extends string>(tabs: readonly T[], id: T, active: T | null): { tabs: T[]; active: T | null } {
  const index = tabs.indexOf(id);
  if (index < 0) return { tabs: [...tabs], active };
  const rest = tabs.filter((tab) => tab !== id);
  if (active !== id) return { tabs: rest, active };
  return { tabs: rest, active: rest[index] ?? rest[index - 1] ?? null };
}

function tabsKey(sessionId: string | null) {
  return `${TABS_KEY}:${sessionId ?? "new"}`;
}

/** The tabs saved for a session, keeping only ids in `allowed`; merged ids open the tab they became. */
export function loadTabs<T extends string>(sessionId: string | null, allowed: readonly T[], store: Storage | null = storage()): T[] | null {
  const raw = store?.getItem(tabsKey(sessionId));
  if (!raw) return null;
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return null;
    const seen = new Set<T>();
    for (const value of parsed) {
      const id = (MERGED[value as DockView] ?? value) as T;
      if (allowed.includes(id)) seen.add(id);
    }
    return [...seen];
  } catch {
    return null;
  }
}

export function saveTabs(sessionId: string | null, tabs: readonly string[], store: Storage | null = storage()) {
  try {
    store?.setItem(tabsKey(sessionId), JSON.stringify(tabs));
  } catch {
    /* quota or private mode: not worth surfacing */
  }
}

/** The chat beside the dock never gets narrower than this; the dock gives way first. */
export const CHAT_MIN_WIDTH = 360;
/** Below this the dock is no use; on a window too small for both, the chat gives way. */
export const DOCK_FLOOR_WIDTH = 300;

/**
 * The widest the dock can be: what is right of the chat's left edge
 * (*chatLeft*, after the session rail) less the chat's minimum, and never
 * more than 78% of the window.
 */
export function dockRoom(viewport: number, chatLeft = 0): number {
  const room = viewport - Math.max(0, chatLeft) - CHAT_MIN_WIDTH;
  return Math.round(Math.max(DOCK_FLOOR_WIDTH, Math.min(viewport * DOCK_MAX_FRACTION, room)));
}

/** A width the dock can take: at least 380px (less on a small window), at most *room*. */
export function clampWidth(width: number, viewport: number, room = dockRoom(viewport)): number {
  const max = Math.max(DOCK_FLOOR_WIDTH, Math.min(room, viewport * DOCK_MAX_FRACTION));
  const min = Math.min(DOCK_MIN_WIDTH, max);
  return Math.round(Math.min(max, Math.max(min, width)));
}

export function loadWidth(store: Storage | null = storage()): number | null {
  const value = Number(store?.getItem(WIDTH_KEY));
  return Number.isFinite(value) && value > 0 ? value : null;
}

export function saveWidth(width: number, store: Storage | null = storage()) {
  try {
    store?.setItem(WIDTH_KEY, String(Math.round(width)));
  } catch {
    /* ignore */
  }
}
