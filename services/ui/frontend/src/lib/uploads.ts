// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { StructurePanel } from "../types/api";

/**
 * Files the user brings to a conversation, stored by the gateway under
 * <shared>/uploads/<session>/ where the agent's tools can read them.
 * *session* is the conversation id, or "new" before it has one.
 */
export interface UploadedFile {
  /** Absolute path on the shared volume, for the agent. */
  path: string;
  /** Where the browser fetches it. */
  url: string;
  name: string;
  size: number;
}

export const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;

async function answer(response: Response): Promise<UploadedFile> {
  if (!response.ok) {
    let detail = response.statusText || `HTTP ${response.status}`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // Not JSON: keep the status text.
    }
    throw new Error(detail);
  }
  return (await response.json()) as UploadedFile;
}

export async function uploadFile(session: string, file: File): Promise<UploadedFile> {
  if (file.size > MAX_UPLOAD_BYTES) throw new Error(`${file.name} is larger than 20 MB`);
  const query = new URLSearchParams({ session, name: file.name });
  return answer(await fetch(`/api/uploads?${query}`, { method: "POST", body: file, headers: { "Content-Type": "application/octet-stream" } }));
}

/** Fetch a PDB entry's mmCIF from RCSB into the conversation's uploads. */
export async function fetchFromRcsb(session: string, id: string): Promise<UploadedFile> {
  return answer(
    await fetch("/api/uploads/rcsb", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id, session }),
    }),
  );
}

/** An uploaded structure as a viewer panel; its path is relative to the shared volume, like the tools' ones. */
export function uploadedStructure(file: UploadedFile): StructurePanel {
  return {
    kind: "structure",
    tool: "upload",
    name: file.name,
    path: file.url.replace(/^\/api\/artifacts\//, ""),
    url: file.url,
    format: /\.(pdb|ent)$/i.test(file.name) ? "pdb" : "cif",
  };
}
