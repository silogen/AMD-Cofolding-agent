// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { ConversationEvent, Panel, SessionSummary } from "../types/api";

/**
 * The gateway's REST endpoints (reinvent4_agent/web/server.py).
 *
 * Requests go to the page's own origin: the dev server and nginx both proxy
 * /api to the gateway and add the access token on the way.
 */

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // Not JSON: keep the status text.
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

export const listSessions = () => request<SessionSummary[]>("/api/sessions");

export const renameSession = (id: string, title: string) =>
  request<SessionSummary>(`/api/sessions/${encodeURIComponent(id)}`, {
    method: "PATCH",
    body: JSON.stringify({ title }),
  });

export const deleteSession = (id: string) =>
  request<{ deleted: string }>(`/api/sessions/${encodeURIComponent(id)}`, { method: "DELETE" });

export const sessionEvents = (id: string) =>
  request<{ session: SessionSummary; events: ConversationEvent[] }>(
    `/api/sessions/${encodeURIComponent(id)}/events`,
  );

export interface ToolResponse<T> {
  name: string;
  result: T;
  panels: Panel[];
}

/**
 * Tools that only read. Identical calls to them that are in flight at the
 * same time share one request; every other tool always gets its own.
 */
const READ_ONLY = new Set([
  "list_jobs",
  "get_optimization_status",
  "get_job_alerts",
  "get_top_candidates",
  "list_ttps",
  "get_ttp",
  "list_shared_assets",
  "list_available_models",
  "list_scoring_components",
  "list_external_services",
]);
/**
 * Polled by several panels at once (run bar, Jobs, Results, TTP), so an
 * answer is reused for this long; each call costs about a second through a
 * tunnel. Shorter than the fastest poll (10 s) so every poll still sees news.
 */
const SHARED_MS: Record<string, number> = { list_jobs: 8_000, get_optimization_status: 8_000, get_job_alerts: 8_000 };

const inFlight = new Map<string, Promise<ToolResponse<unknown>>>();
const recent = new Map<string, { at: number; response: ToolResponse<unknown> }>();
/** Bumped by every call that may change something; answers read before it are not shared after it. */
let generation = 0;

/** Forget every shared answer, e.g. after a call that changes the jobs. */
export function clearToolCache(): void {
  generation += 1;
  inFlight.clear();
  recent.clear();
}

/**
 * Call one MCP tool directly, without the agent. Only the tools in the
 * gateway's DIRECT_TOOLS list are allowed; the others answer 403.
 *
 * Results of read-only tools may be shared between callers, so callers must
 * not modify them.
 */
export function callTool<T = unknown>(name: string, args: Record<string, unknown> = {}): Promise<ToolResponse<T>> {
  const send = () =>
    request<ToolResponse<T>>(`/api/tools/${encodeURIComponent(name)}`, {
      method: "POST",
      body: JSON.stringify({ arguments: args }),
    });
  if (!READ_ONLY.has(name)) {
    clearToolCache();
    return send().finally(clearToolCache);
  }
  const key = `${name} ${JSON.stringify(args)}`;
  const ttl = SHARED_MS[name] ?? 0;
  const hit = recent.get(key);
  if (hit && Date.now() - hit.at < ttl) return Promise.resolve(hit.response as ToolResponse<T>);
  const pending = inFlight.get(key);
  if (pending) return pending as Promise<ToolResponse<T>>;
  const started = generation;
  const call = send().finally(() => {
    if (inFlight.get(key) === call) inFlight.delete(key);
  });
  inFlight.set(key, call);
  if (ttl > 0) {
    call.then(
      (response) => {
        if (generation === started) recent.set(key, { at: Date.now(), response });
      },
      () => undefined,
    );
  }
  return call;
}

export const jobFileUrl = (jobId: string, name: string) =>
  `/api/jobs/${encodeURIComponent(jobId)}/files/${encodeURIComponent(name)}`;

/** Save text as a file in the browser's downloads. */
export function download(name: string, content: string, type = "text/plain"): void {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
}
