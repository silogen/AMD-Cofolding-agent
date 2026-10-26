// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * Lets anything in the page bring up a workbench tab: an "Open candidates"
 * link under an answer, "Open jobs" on a monitor alert. The app listens and
 * switches tabs.
 */
export type WorkbenchTab = "candidates" | "molecules" | "properties" | "jobs" | "ttp" | "target" | "assets" | "tables" | "activity";

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
