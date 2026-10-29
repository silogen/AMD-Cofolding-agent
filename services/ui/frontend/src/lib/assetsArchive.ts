// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { ApiError, callTool, request } from "./api";
import { parseSharedAssets } from "./assets";
import type { ListedFile } from "./assetsCatalog";
import { parseResult } from "./ttp";

/**
 * The shared volume in bulk (reinvent4_agent/web/archive.py): every file,
 * and many files as one ZIP. A gateway older than that module answers 404;
 * the listing then falls back to list_shared_assets (structures and MSAs only).
 */

export interface Listing {
  files: ListedFile[];
  /** Only structures and MSAs: the gateway cannot list the rest yet. */
  partial: boolean;
  truncated: boolean;
}

export async function listFiles(): Promise<Listing> {
  try {
    const body = await request<{ files: ListedFile[]; truncated?: boolean }>("/api/files");
    return { files: body.files ?? [], partial: false, truncated: Boolean(body.truncated) };
  } catch (err) {
    if (!(err instanceof ApiError) || err.status !== 404) throw err;
  }
  const response = await callTool<string>("list_shared_assets");
  const files = parseSharedAssets(parseResult(response.result)).map((a) => ({ path: a.path, queryLength: a.queryLength }));
  return { files, partial: true, truncated: false };
}

/** What went wrong with an archive, in a line; a 404 without our detail means the route is missing. */
export function archiveError(status: number, detail: string): string {
  if (status === 404 && !/^Not found or not readable/.test(detail)) return "Bulk download needs the updated gateway. Download files one by one for now.";
  return detail || `The archive failed (HTTP ${status}).`;
}

/** Download the files as `<name>.zip`; throws an Error with a short message. */
export async function downloadArchive(paths: string[], name: string): Promise<void> {
  const response = await fetch("/api/archive", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ paths, name }),
  });
  if (!response.ok) {
    let detail = "";
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // Not JSON: keep the status.
    }
    throw new Error(archiveError(response.status, detail));
  }
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement("a");
  link.href = url;
  link.download = `${name}.zip`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}
