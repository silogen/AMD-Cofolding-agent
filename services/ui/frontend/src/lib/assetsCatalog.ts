// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { assetUrl, sharedRelative, type Asset } from "./assets";

/**
 * The Assets tab's model of the data: every file on the shared volume (and
 * the job files this chat named) with a type, a run or folder it belongs
 * to, and a short label, so the tab can filter, group and download them.
 */

export type FileType = "prediction" | "structure" | "sdf" | "table" | "pdf" | "msa" | "config" | "other";

export const FILE_TYPES: { type: FileType; label: string }[] = [
  { type: "prediction", label: "Predictions" },
  { type: "structure", label: "Structures" },
  { type: "sdf", label: "SDF" },
  { type: "table", label: "CSV" },
  { type: "pdf", label: "PDF" },
  { type: "msa", label: "MSAs" },
  { type: "config", label: "Run logs" },
  { type: "other", label: "Other" },
];

/** Engine internals (npz, pkl, lock, yaml, most json): shown only behind their own chip, never downloaded with "all". */
export const HIDDEN_TYPE: FileType = "other";

const EXT_TYPE: Record<string, FileType> = {
  sdf: "sdf",
  mol: "sdf",
  mol2: "sdf",
  csv: "table",
  tsv: "table",
  smi: "table",
  pdf: "pdf",
  a3m: "msa",
  a2m: "msa",
  fasta: "msa",
  fa: "msa",
  sto: "msa",
  pdb: "structure",
  cif: "structure",
  mmcif: "structure",
  ent: "structure",
};

const STRUCTURE_EXTS = new Set(["pdb", "cif", "mmcif", "ent"]);
const extOf = (path: string) => /\.([a-z0-9]+)$/i.exec(path)?.[1].toLowerCase() ?? "";
const IN_PREDICTIONS = /(^|\/)predictions\//;
/** Boltz's per-model scores, worth keeping next to the model; its raw pae/pde/plddt arrays are not. */
const BOLTZ_SCORES = /(^|\/)(confidence|affinity)_[^/]*\.json$/i;
const JOB_DIR = /(?:^|\/)job_([0-9a-f]{6,32})\/(.+)$/;
/** The job files /api/jobs/{id}/files/{name} and /api/archive serve. */
const JOB_FILE = /(?:^|\/)job_([0-9a-f]{6,32})\/(reinvent\.log|config\.toml|job_metadata\.json|monitor_actions\.jsonl|[\w.-]+\.csv)$/;

export function fileType(path: string): FileType {
  const clean = path.replace(/\\/g, "/");
  const type = EXT_TYPE[extOf(clean)];
  if (type === "structure") return IN_PREDICTIONS.test(clean) ? "prediction" : type;
  if (type) return type;
  if (IN_PREDICTIONS.test(clean) && BOLTZ_SCORES.test(clean)) return "prediction";
  if (JOB_FILE.test(clean)) return "config";
  return "other";
}

export interface FileGroup {
  key: string;
  label: string;
  /** A run or upload folder: when this chat names one of its files, the whole folder belongs to the chat. */
  run: boolean;
}

const shortId = (id: string) => (id.length > 8 ? id.slice(0, 8) : id);

/** The run or folder a file belongs to. */
export function groupOf(path: string): FileGroup {
  const clean = path.replace(/\\/g, "/");
  const job = JOB_DIR.exec(clean);
  if (job) return { key: `job_${job[1]}`, label: `Job ${shortId(job[1])}`, run: true };
  const parts = (sharedRelative(clean) ?? clean.replace(/^\/+/, "")).split("/").filter(Boolean);
  if (parts.length < 2) return { key: "", label: "Top level", run: false };
  if (parts[0] === "predictions" && parts.length > 2) {
    const run = parts[1].replace(/^(out|boltz_results)_/, "");
    return { key: `predictions/${parts[1]}`, label: `Cofold ${shortId(run)}`, run: true };
  }
  if (parts[0] === "uploads") return { key: "uploads", label: "Uploads", run: true };
  return { key: parts[0], label: parts[0], run: false };
}

const BOLTZ_PART = /^(confidence|pae|pde|plddt|affinity)_(.+)$/;
const MODEL = /^(.+)_model_(\d+)$/;

/** A short name: "mol_0 · model 0 · confidence" for confidence_mol_0_model_0.json, long hashes cut to 8 characters. */
export function fileLabel(path: string): string {
  const name = path.replace(/\\/g, "/").split("/").pop() ?? path;
  let stem = name.replace(/\.[a-z0-9]+$/i, "") || name;
  const part = BOLTZ_PART.exec(stem);
  if (part) stem = part[2];
  const model = MODEL.exec(stem);
  if (model) stem = model[1];
  stem = stem.replace(/[0-9a-f]{16,}/gi, (hash) => `${hash.slice(0, 8)}\u2026`);
  return [stem, model && `model ${model[2]}`, part?.[1]].filter(Boolean).join(" \u00b7 ");
}

export interface ListedFile {
  path: string;
  size?: number | null;
  mtime?: number | null;
  queryLength?: number | null;
}

export interface CatalogFile {
  /** Relative to the shared volume, or as a tool reported a job file. */
  path: string;
  label: string;
  ext: string;
  type: FileType;
  group: FileGroup;
  url: string | null;
  /** How /api/archive names it, or null when it cannot go in an archive. */
  archivePath: string | null;
  size: number | null;
  mtime: number | null;
  queryLength: number | null;
  /** Named by this chat's tools, or in a run folder they named. */
  inChat: boolean;
  openable: boolean;
}

export function archivePath(path: string): string | null {
  const clean = path.replace(/\\/g, "/");
  const job = JOB_FILE.exec(clean);
  if (job) return `job_${job[1]}/${job[2]}`;
  return sharedRelative(clean);
}

const keyOf = (path: string) => archivePath(path) ?? path.replace(/\\/g, "/");

/** The listed files plus those only this chat named, each marked as in the chat or not. */
export function buildCatalog(listed: ListedFile[], session: Asset[]): CatalogFile[] {
  const named = new Set(session.map((a) => keyOf(a.path)));
  const runs = new Set(session.map((a) => groupOf(a.path)).filter((g) => g.run).map((g) => g.key));
  const byKey = new Map<string, CatalogFile>();
  const add = (file: ListedFile) => {
    const path = file.path.replace(/\\/g, "/");
    const key = keyOf(path);
    if (byKey.has(key)) return;
    const group = groupOf(path);
    const ext = extOf(path);
    const archived = archivePath(path);
    byKey.set(key, {
      path,
      label: fileLabel(path),
      ext,
      type: fileType(path),
      group,
      url: assetUrl(path),
      archivePath: archived,
      size: file.size ?? null,
      mtime: file.mtime ?? null,
      queryLength: file.queryLength ?? null,
      inChat: named.has(key) || (group.run && runs.has(group.key)),
      openable: STRUCTURE_EXTS.has(ext) && sharedRelative(path) !== null && !JOB_DIR.test(path),
    });
  };
  listed.forEach(add);
  session.forEach((a) => add({ path: a.path, queryLength: a.queryLength }));
  return [...byKey.values()];
}

export type Scope = "chat" | "all";

export interface Filter {
  scope: Scope;
  type: FileType | null;
  query: string;
}

function matches(file: CatalogFile, words: string[]): boolean {
  if (!words.length) return true;
  const text = `${file.path} ${file.label} ${file.group.label}`.toLowerCase();
  return words.every((word) => text.includes(word));
}

/** The files in scope that match the search, before the type filter, so the type chips can count them. */
export function inScope(files: CatalogFile[], filter: Omit<Filter, "type">): CatalogFile[] {
  const words = filter.query.toLowerCase().split(/\s+/).filter(Boolean);
  return files.filter((f) => (filter.scope === "all" || f.inChat) && matches(f, words));
}

/** One type, or every type but the hidden one. */
export const ofType = (files: CatalogFile[], type: FileType | null) => files.filter((f) => (type ? f.type === type : f.type !== HIDDEN_TYPE));

/** Null when no file's size is known. */
export function totalBytes(files: CatalogFile[]): number | null {
  let total: number | null = null;
  for (const file of files) if (file.size !== null) total = (total ?? 0) + file.size;
  return total;
}

export function typeCounts(files: CatalogFile[]): { type: FileType; label: string; count: number }[] {
  return FILE_TYPES.map((t) => ({ ...t, count: files.filter((f) => f.type === t.type).length })).filter((t) => t.count > 0);
}

export interface Group extends FileGroup {
  files: CatalogFile[];
  /** Null when no file's size is known. */
  bytes: number | null;
  inChat: boolean;
  newest: number;
}

const byLabel = (a: CatalogFile, b: CatalogFile) => a.label.localeCompare(b.label, "en", { numeric: true }) || a.path.localeCompare(b.path);

/** Files by run or folder: those of this chat first, then the most recently written, files by label. */
export function groupFiles(files: CatalogFile[]): Group[] {
  const groups = new Map<string, Group>();
  for (const file of files) {
    const group = groups.get(file.group.key) ?? { ...file.group, files: [], bytes: null, inChat: false, newest: 0 };
    group.files.push(file);
    group.inChat ||= file.inChat;
    group.newest = Math.max(group.newest, file.mtime ?? 0);
    groups.set(file.group.key, group);
  }
  for (const group of groups.values()) {
    group.files.sort(byLabel);
    group.bytes = totalBytes(group.files);
  }
  return [...groups.values()].sort(
    (a, b) => Number(b.inChat) - Number(a.inChat) || b.newest - a.newest || a.label.localeCompare(b.label, "en", { numeric: true }),
  );
}

/** The archive's download name: "chat-predictions", "all-sdf", "brd4-files". */
export function archiveName(filter: Filter): string {
  const type = FILE_TYPES.find((t) => t.type === filter.type)?.label.toLowerCase().replace(/\W+/g, "-") ?? "files";
  const query = filter.query.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 30);
  return [filter.scope === "chat" ? "chat" : "all", query, type].filter(Boolean).join("-");
}
