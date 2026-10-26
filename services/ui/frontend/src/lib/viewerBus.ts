// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { ViewerAction } from "./viewerCommands";

/**
 * Lets anything in the page drive the structure viewer: a residue link in an
 * answer, "/view pocket" in the composer, "Show on structure" on a contacts
 * card.
 *
 * Two kinds of listener: the app, which makes sure the viewer is on screen
 * and showing the right structure, and the viewer itself, which runs the
 * action. A request made while no viewer is mounted waits for one.
 */
export interface ViewerRequest {
  /** Show this structure first (job file path). */
  path?: string | null;
  /** Then run this, if anything. */
  action?: ViewerAction;
  /** What asked, for the viewer's message line. */
  source?: string;
}

type Listener = (request: ViewerRequest) => void;

const openers = new Set<Listener>();
const viewers = new Set<Listener>();
let pending: ViewerRequest | null = null;

export function sendToViewer(request: ViewerRequest): void {
  openers.forEach((listener) => listener(request));
  if (viewers.size === 0) {
    pending = request;
    return;
  }
  viewers.forEach((listener) => listener(request));
}

/** For the app: bring the viewer up when something asks for it. */
export function onOpenRequest(listener: Listener): () => void {
  openers.add(listener);
  return () => {
    openers.delete(listener);
  };
}

/** For the viewer: receive requests, including one made before it mounted. */
export function onViewerRequest(listener: Listener): () => void {
  viewers.add(listener);
  if (pending) {
    const request = pending;
    pending = null;
    queueMicrotask(() => listener(request));
  }
  return () => {
    viewers.delete(listener);
  };
}
