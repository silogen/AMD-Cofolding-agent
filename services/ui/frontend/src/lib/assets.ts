// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { jobFileUrl } from "./api";
import { parseResult, type ToolEvent } from "./ttp";

/**
 * Files on the shared volume, from list_shared_assets
 * (reinvent4_mcp/tools/external.py), plus the files the session's tools
 * wrote or read. Shared files are served at /api/artifacts/<path>, job files
 * at /api/jobs/<id>/files/<name>.
 */

/** What list_shared_assets returns. */
export interface SharedAssetsListing {
  shared_dir?: string;
  structures?: string[];
  msas?: string[];
  msa_details?: { path: string; query_length: number | null }[];
  cofold_outputs?: string[];
  counts?: Record<string, number>;
  note?: string;
}

export type AssetKind = "structure" | "cofold" | "sequence" | "smiles" | "config" | "other";

export interface Asset {
  /** Relative to the shared volume, or as the tool reported it. */
  path: string;
  name: string;
  dir: string;
  kind: AssetKind;
  ext: string;
  /** Where the browser can fetch it, or null when the gateway does not serve it. */
  url: string | null;
  /** Length of an MSA's query sequence. */
  queryLength: number | null;
  /** Listed by list_shared_assets, or only mentioned by a tool in this session. */
  source: "shared" | "session";
  /** The tool that mentioned it, for session files. */
  tool: string | null;
}

export const ASSET_GROUPS: { kind: AssetKind; label: string; colour: string }[] = [
  { kind: "structure", label: "Structures", colour: "var(--color-structure)" },
  { kind: "cofold", label: "Cofold outputs", colour: "var(--color-structure)" },
  { kind: "sequence", label: "Sequences & MSAs", colour: "var(--color-molecule)" },
  { kind: "smiles", label: "SMILES & tables", colour: "var(--color-table)" },
  { kind: "config", label: "Configs", colour: "var(--color-accent)" },
  { kind: "other", label: "Other", colour: "var(--text-faint)" },
];

const EXT_KIND: Record<string, AssetKind> = {
  pdb: "structure",
  cif: "structure",
  mmcif: "structure",
  ent: "structure",
  a3m: "sequence",
  a2m: "sequence",
  fasta: "sequence",
  fa: "sequence",
  sto: "sequence",
  seq: "sequence",
  smi: "smiles",
  smiles: "smiles",
  csv: "smiles",
  tsv: "smiles",
  sdf: "smiles",
  toml: "config",
  json: "config",
  yaml: "config",
  yml: "config",
};

const extOf = (path: string) => /\.([a-z0-9]+)$/i.exec(path)?.[1].toLowerCase() ?? "";

export function assetKind(path: string, cofold = false): AssetKind {
  const kind = EXT_KIND[extOf(path)] ?? "other";
  if (kind === "structure" && (cofold || /(^|\/)predictions\//.test(path))) return "cofold";
  return kind;
}

const JOB_FILE = /(?:^|\/)job_([0-9a-f]{6,32})\/(reinvent\.log|config\.toml|job_metadata\.json|monitor_actions\.jsonl|[\w.-]+\.csv)$/;
const SHARED_PREFIX = /^\/?shared\//;

/** The path relative to the shared volume, or null for a path outside it. */
export function sharedRelative(path: string): string | null {
  const clean = path.replace(/\\/g, "/").trim();
  if (SHARED_PREFIX.test(clean)) return clean.replace(SHARED_PREFIX, "");
  if (clean.startsWith("/") || /^[a-z]:\//i.test(clean) || clean.split("/").includes("..")) return null;
  return clean;
}

/** Where the browser can download a file, or null when the gateway does not serve it. */
export function assetUrl(path: string): string | null {
  const clean = path.replace(/\\/g, "/");
  const job = JOB_FILE.exec(clean);
  if (job) return jobFileUrl(job[1], job[2]);
  const rel = sharedRelative(clean);
  return rel ? `/api/artifacts/${rel.split("/").map(encodeURIComponent).join("/")}` : null;
}

function makeAsset(path: string, source: Asset["source"], extra: Partial<Asset> = {}): Asset {
  const clean = path.replace(/\\/g, "/");
  const parts = clean.split("/");
  return {
    path: clean,
    name: parts[parts.length - 1],
    dir: parts.slice(0, -1).join("/"),
    kind: assetKind(clean),
    ext: extOf(clean),
    url: assetUrl(clean),
    queryLength: null,
    source,
    tool: null,
    ...extra,
  };
}

/** The listing as assets, one per file, cofold outputs separated from input structures. */
export function parseSharedAssets(payload: unknown): Asset[] {
  const listing = parseResult(payload);
  if (!listing || typeof listing !== "object" || Array.isArray(listing)) return [];
  const { structures = [], msas = [], msa_details = [], cofold_outputs = [] } = listing as SharedAssetsListing;
  const cofold = new Set(cofold_outputs);
  const lengths = new Map(msa_details.map((d) => [d.path, d.query_length ?? null]));
  const seen = new Set<string>();
  const out: Asset[] = [];
  for (const path of [...structures, ...cofold_outputs, ...msas]) {
    if (typeof path !== "string" || seen.has(path)) continue;
    seen.add(path);
    out.push(
      makeAsset(path, "shared", {
        kind: assetKind(path, cofold.has(path)),
        queryLength: lengths.get(path) ?? null,
      }),
    );
  }
  return out;
}

const FILE_TOKEN = /(?:^|[\s"'`(=:,[])((?:\/|[\w.-]+\/)?(?:[\w.-]+\/)*[\w.-]+\.(?:pdb|cif|mmcif|a3m|fasta|fa|sto|smi|csv|tsv|sdf|toml|yaml|yml))(?=$|[\s"'`),\]:;])/gi;

function collectPaths(node: unknown, out: Set<string>, depth = 0): void {
  if (depth > 6) return;
  if (typeof node === "string") {
    if (node.length > 20000) return;
    for (const match of node.matchAll(FILE_TOKEN)) out.add(match[1]);
    return;
  }
  if (Array.isArray(node)) {
    for (const item of node) collectPaths(item, out, depth + 1);
    return;
  }
  if (node && typeof node === "object") {
    for (const value of Object.values(node)) collectPaths(value, out, depth + 1);
  }
}

/**
 * A mentioned path the gateway can serve, in the form it is served under: a
 * job file as reported, a shared file relative to the volume. A bare file
 * name could be anywhere, so it is dropped.
 */
export function servedPath(raw: string): string | null {
  const clean = raw.replace(/\\/g, "/");
  if (JOB_FILE.test(clean)) return clean;
  const rel = sharedRelative(clean);
  if (!rel) return null;
  return SHARED_PREFIX.test(clean) || rel.includes("/") ? rel : null;
}

/** Files the session's tool calls and results name, newest mention last. */
export function sessionAssets(events: ToolEvent[]): Asset[] {
  const byPath = new Map<string, Asset>();
  for (const event of events) {
    const found = new Set<string>();
    if (event.type === "tool_call") collectPaths(event.arguments, found);
    else if (!event.isError) collectPaths(parseResult(event.result) ?? event.result, found);
    for (const raw of found) {
      const path = servedPath(raw);
      if (!path) continue;
      byPath.set(path, makeAsset(path, "session", { tool: event.name }));
    }
  }
  return [...byPath.values()];
}

/** Shared files first, then session files not already listed. */
export function mergeAssets(shared: Asset[], session: Asset[]): Asset[] {
  const known = new Set(shared.map((a) => sharedRelative(a.path) ?? a.path));
  return [...shared, ...session.filter((a) => !known.has(sharedRelative(a.path) ?? a.path))];
}

export function groupAssets(assets: Asset[], query = ""): { kind: AssetKind; label: string; colour: string; assets: Asset[] }[] {
  const needle = query.trim().toLowerCase();
  const shown = needle ? assets.filter((a) => a.path.toLowerCase().includes(needle)) : assets;
  return ASSET_GROUPS.map((group) => ({
    ...group,
    assets: shown.filter((a) => a.kind === group.kind).sort((a, b) => a.path.localeCompare(b.path)),
  })).filter((group) => group.assets.length > 0);
}

/** Viewer-openable: a structure the gateway can serve from the shared volume. */
export const canOpen = (asset: Asset) => (asset.kind === "structure" || asset.kind === "cofold") && sharedRelative(asset.path) !== null;
