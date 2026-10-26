// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { StructurePanel } from "../types/api";
import { addUserStructure } from "./userStructures";
import { sendToViewer } from "./viewerBus";

/**
 * Deposited structures from the RCSB PDB, opened by id. The gateway copies
 * the entry to the shared volume so tools can use it; until it can, the
 * browser shows it straight from RCSB.
 */

export const PDB_ID = /^[0-9][A-Za-z0-9]{3}$/;

export const isPdbId = (text: string) => PDB_ID.test(text.trim());

/** The conversation in the address ("#/s/<id>"), or "new" before it has one. */
export function sessionFromLocation(): string {
  const match = /^#\/s\/(s-[0-9-]+-[0-9a-f]+)$/.exec(window.location.hash);
  return match ? match[1] : "new";
}

/** Fetch an entry into the conversation's structures. */
export async function fetchPdbEntry(id: string, session: string): Promise<StructurePanel> {
  const code = id.trim().toUpperCase();
  if (!PDB_ID.test(code)) throw new Error(`“${id}” is not a PDB id (four characters, starting with a digit).`);
  const response = await fetch("/api/uploads/rcsb", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: code, session }),
  });
  let panel: StructurePanel;
  if (response.ok) {
    const body = (await response.json()) as { path: string; url: string; name?: string };
    panel = { kind: "structure", tool: "rcsb", name: body.name ?? `${code}.cif`, path: body.path, url: body.url, format: "cif" };
  } else if (response.status === 404 || response.status === 405) {
    panel = { kind: "structure", tool: "rcsb", name: `${code}.cif`, path: `rcsb/${code}.cif`, url: `https://files.rcsb.org/download/${code}.cif`, format: "cif" };
  } else {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // Not JSON: keep the status text.
    }
    throw new Error(`Could not fetch ${code}: ${detail}`);
  }
  addUserStructure(session, panel);
  return panel;
}

/** Fetch an entry and show it in the viewer. */
export async function openPdbEntry(id: string, session: string = sessionFromLocation()): Promise<StructurePanel> {
  const panel = await fetchPdbEntry(id, session);
  // The app picks the structure from its list, which updates on the next render.
  await new Promise((resolve) => window.setTimeout(resolve, 30));
  sendToViewer({ path: panel.path, source: "rcsb" });
  return panel;
}
