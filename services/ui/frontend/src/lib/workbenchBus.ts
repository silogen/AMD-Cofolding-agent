// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useSyncExternalStore } from "react";

/**
 * Lets anything in the page bring up a workbench tab: an "Open candidates"
 * link under an answer, "Open jobs" on a monitor alert. The app listens and
 * switches tabs.
 */
export type WorkbenchTab = "candidates" | "molecules" | "properties" | "shortlist" | "jobs" | "results" | "ttp" | "target" | "assets" | "tables" | "activity";

type Listener = (tab: WorkbenchTab) => void;

const listeners = new Set<Listener>();

export function openWorkbench(tab: WorkbenchTab): void {
  listeners.forEach((listener) => listener(tab));
}

export function onWorkbenchRequest(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/*
 * The REINVENT run picked for the workbench (see scopedJob), null for the
 * conversation's own. Shared by Summary, Results, Analysis and the run bar, so
 * a run picked in one is the run shown in the others; the app clears it when
 * another conversation opens.
 */
let focused: string | null = null;
const runListeners = new Set<() => void>();
const changeListeners = new Set<() => void>();

export function focusRun(jobId: string | null): void {
  if (jobId === focused) return;
  focused = jobId;
  runListeners.forEach((listener) => listener());
}

export const focusedRun = (): string | null => focused;

export function useFocusedRun(): string | null {
  return useSyncExternalStore(
    (listener) => {
      runListeners.add(listener);
      return () => runListeners.delete(listener);
    },
    focusedRun,
  );
}

/** Show one run in the Jobs or Results tab. */
export function openRun(jobId: string, tab: "jobs" | "results" = "jobs"): void {
  focusRun(jobId);
  openWorkbench(tab);
}

/** Say that a run was started or stopped here, so views of the job list reload. */
export function notifyRunsChanged(): void {
  changeListeners.forEach((listener) => listener());
}

export function onRunsChanged(listener: () => void): () => void {
  changeListeners.add(listener);
  return () => {
    changeListeners.delete(listener);
  };
}
