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
 * Call one MCP tool directly, without the agent. Only the tools in the
 * gateway's DIRECT_TOOLS list are allowed; the others answer 403.
 */
export const callTool = <T = unknown>(name: string, args: Record<string, unknown> = {}) =>
  request<ToolResponse<T>>(`/api/tools/${encodeURIComponent(name)}`, {
    method: "POST",
    body: JSON.stringify({ arguments: args }),
  });

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
