// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useState } from "react";
import { sessionAssets, sharedRelative } from "../../lib/assets";
import { downloadArchive, listFiles, type Listing } from "../../lib/assetsArchive";
import {
  archiveName,
  buildCatalog,
  groupFiles,
  HIDDEN_TYPE,
  inScope,
  ofType,
  totalBytes,
  typeCounts,
  type CatalogFile,
  type FileType,
  type Group,
  type Scope,
} from "../../lib/assetsCatalog";
import { formatBytes } from "../../lib/sessionData";
import type { ToolEvent } from "../../lib/ttp";

interface Props {
  /** The session's tool calls, from `toolEvents(artifacts.steps)`; their files are listed too. */
  events?: ToolEvent[];
  /** Show a structure (path relative to the shared volume) in the 3D viewer. */
  onOpenStructure: (path: string) => void;
  /** Put text in the composer. */
  onInsert: (text: string) => void;
}

type State = { state: "loading" } | { state: "error"; message: string } | ({ state: "ready" } & Listing);

/** Rows a group shows before "Show all", so a run of thousands of files stays responsive. */
const ROWS = 200;

const chip = (on: boolean) =>
  `cursor-pointer rounded-full border px-2 py-0.5 text-[11px] whitespace-nowrap touch:min-h-11 ${on ? "border-strong bg-inset text-body" : "border-subtle text-muted hover:text-body"}`;
const action = "text-muted hover:text-body hover:bg-inset cursor-pointer rounded-md px-1.5 py-0.5 text-[11px] touch:min-h-11";
const checkbox = "accent-[var(--color-accent)] disabled:opacity-30";

/**
 * The data of the conversation and of the shared volume: filter by type,
 * search, open, insert into the prompt, and download one file, a run, a
 * selection or everything shown as a ZIP.
 */
export function AssetsPanel({ events = [], onOpenStructure, onInsert }: Props) {
  const [listing, setListing] = useState<State>({ state: "loading" });
  const [reload, setReload] = useState(0);
  const [scopeChoice, setScope] = useState<Scope | null>(null);
  const [type, setType] = useState<FileType | null>(null);
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState<Set<string>>(() => new Set());
  const [selected, setSelected] = useState<Set<string>>(() => new Set());
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    listFiles()
      .then((result) => {
        if (!cancelled) setListing({ state: "ready", ...result });
      })
      .catch((err: unknown) => {
        if (!cancelled) setListing({ state: "error", message: err instanceof Error ? err.message : String(err) });
      });
    return () => {
      cancelled = true;
    };
  }, [reload]);

  const session = useMemo(() => sessionAssets(events), [events]);
  const catalog = useMemo(() => buildCatalog(listing.state === "ready" ? listing.files : [], session), [listing, session]);
  const hasChat = useMemo(() => catalog.some((f) => f.inChat), [catalog]);
  const scope = scopeChoice ?? (hasChat ? "chat" : "all");
  const scoped = useMemo(() => inScope(catalog, { scope, query }), [catalog, scope, query]);
  const counts = useMemo(() => typeCounts(scoped), [scoped]);
  const activeType = type && counts.some((c) => c.type === type) ? type : null;
  const shown = useMemo(() => ofType(scoped, activeType), [scoped, activeType]);
  const groups = useMemo(() => groupFiles(shown), [shown]);
  const expandAll = query.trim() !== "" || groups.length === 1;
  const visibleCount = counts.reduce((sum, c) => sum + (c.type === HIDDEN_TYPE ? 0 : c.count), 0);
  const bulkBytes = useMemo(
    () => totalBytes(selected.size ? catalog.filter((f) => f.archivePath !== null && selected.has(f.archivePath)) : shown),
    [catalog, selected, shown],
  );

  const archive = async (paths: string[], name: string) => {
    if (!paths.length || busy) return;
    setBusy(true);
    setNotice(null);
    try {
      await downloadArchive(paths, name);
    } catch (err) {
      setNotice(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const archivable = (files: CatalogFile[]) => files.flatMap((f) => (f.archivePath ? [f.archivePath] : []));
  const downloadShown = () =>
    selected.size ? archive([...selected], "selected-files") : archive(archivable(shown), archiveName({ scope, type: activeType, query }));

  const toggle = (paths: string[], on: boolean) =>
    setSelected((current) => {
      const next = new Set(current);
      for (const path of paths) {
        if (on) next.add(path);
        else next.delete(path);
      }
      return next;
    });

  const refresh = () => {
    setListing({ state: "loading" });
    setReload((n) => n + 1);
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="border-subtle flex shrink-0 flex-col gap-2 border-b px-3 py-2">
        <div className="flex items-center gap-2">
          <div className="border-subtle inline-flex shrink-0 overflow-hidden rounded-md border" role="group" aria-label="Scope">
            {(["chat", "all"] as const).map((s) => (
              <button
                key={s}
                type="button"
                aria-pressed={scope === s}
                onClick={() => setScope(s)}
                className={`cursor-pointer px-2.5 py-1 text-[11px] font-medium touch:min-h-11 ${scope === s ? "bg-inset text-body" : "text-muted hover:text-body"}`}
              >
                {s === "chat" ? "This chat" : "All"}
              </button>
            ))}
          </div>
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search files"
            aria-label="Search files"
            className="bg-sunken border-subtle focus:border-strong min-w-0 flex-1 rounded-md border px-2.5 py-1 text-xs outline-none"
          />
          <button
            type="button"
            onClick={refresh}
            disabled={listing.state === "loading"}
            title="Refresh"
            aria-label="Refresh"
            className={`${action} shrink-0 disabled:cursor-default disabled:opacity-50`}
          >
            {"\u21bb"}
          </button>
        </div>
        {counts.length > 0 && (
          <div className="flex flex-wrap gap-1" role="group" aria-label="File type">
            <button type="button" aria-pressed={activeType === null} onClick={() => setType(null)} className={chip(activeType === null)}>
              All <span className="text-faint">{visibleCount}</span>
            </button>
            {counts.map((c) => (
              <button
                key={c.type}
                type="button"
                aria-pressed={activeType === c.type}
                onClick={() => setType(activeType === c.type ? null : c.type)}
                title={c.type === HIDDEN_TYPE ? "Engine internals (npz, pkl, lock, yaml, json); not part of All" : undefined}
                className={`${chip(activeType === c.type)} ${c.type === HIDDEN_TYPE && activeType !== c.type ? "ml-auto border-transparent opacity-70" : ""}`}
              >
                {c.label} <span className="text-faint">{c.count}</span>
              </button>
            ))}
          </div>
        )}
      </div>

      {shown.length > 0 && (
        <div className="border-subtle flex shrink-0 items-center gap-2 border-b px-3 py-1.5 text-[11px]">
          <span className="text-muted min-w-0 truncate">
            {selected.size ? `${selected.size} selected` : `${shown.length} file${shown.length === 1 ? "" : "s"}`}
          </span>
          {selected.size > 0 && (
            <button type="button" onClick={() => setSelected(new Set())} className={action}>
              Clear
            </button>
          )}
          <button
            type="button"
            onClick={() => void downloadShown()}
            disabled={busy}
            title={selected.size ? "Download the selected files as one ZIP" : "Download the files shown as one ZIP"}
            className="border-subtle hover:border-strong text-body ml-auto shrink-0 cursor-pointer rounded-md border px-2 py-0.5 text-[11px] disabled:cursor-default disabled:opacity-50 touch:min-h-11"
          >
            {busy ? "Preparing\u2026" : `${selected.size ? `Download ${selected.size}` : "Download all"}${bulkBytes ? ` (${formatBytes(bulkBytes)})` : ""}`}
          </button>
        </div>
      )}

      {notice && (
        <p className="border-subtle mx-3 mt-2 flex items-start gap-2 rounded-md border border-l-2 px-3 py-1.5 text-[11px]" style={{ borderLeftColor: "var(--color-accent)" }} role="alert">
          <span className="min-w-0 flex-1">{notice}</span>
          <button type="button" onClick={() => setNotice(null)} aria-label="Dismiss" className="text-faint hover:text-body cursor-pointer">
            {"\u00d7"}
          </button>
        </p>
      )}

      <div className="min-h-0 flex-1 overflow-y-auto px-2 py-2">
        {listing.state === "error" && (
          <p className="border-subtle mx-1 rounded-md border border-l-2 px-3 py-2 text-[11px]" style={{ borderLeftColor: "var(--color-accent)" }}>
            Could not list the files: {listing.message}
          </p>
        )}
        {listing.state === "ready" && (listing.partial || listing.truncated) && (
          <p className="text-faint px-1 pb-1 text-[10px]">
            {listing.partial ? "Only structures and MSAs are listed until the gateway is updated." : `Showing the first ${listing.files.length.toLocaleString("en")} files.`}
          </p>
        )}
        {groups.map((group) => (
          <GroupSection
            key={group.key}
            group={group}
            expanded={expandAll || open.has(group.key)}
            onToggleOpen={() =>
              setOpen((current) => {
                const next = new Set(current);
                if (next.has(group.key)) next.delete(group.key);
                else next.add(group.key);
                return next;
              })
            }
            selected={selected}
            onSelect={toggle}
            onDownload={() => void archive(archivable(group.files), group.label.toLowerCase().replace(/[^a-z0-9]+/g, "-"))}
            busy={busy}
            onOpenStructure={onOpenStructure}
            onInsert={onInsert}
          />
        ))}
        {listing.state === "loading" && <p className="text-faint px-6 py-10 text-center text-xs">Loading{"\u2026"}</p>}
        {listing.state === "ready" && groups.length === 0 && (
          <p className="text-muted px-6 py-10 text-center text-xs">
            {scoped.length > 0 && !activeType
              ? "Only engine internals here, under Other."
              : query || activeType
                ? "No file matches."
                : scope === "chat"
                  ? "This chat has no files yet."
                  : "No files yet."}
            {scope === "chat" && catalog.length > 0 && (
              <button type="button" onClick={() => setScope("all")} className="text-body ml-1 cursor-pointer underline-offset-2 hover:underline">
                Show all
              </button>
            )}
          </p>
        )}
      </div>
    </div>
  );
}

interface GroupProps {
  group: Group;
  expanded: boolean;
  onToggleOpen: () => void;
  selected: Set<string>;
  onSelect: (paths: string[], on: boolean) => void;
  onDownload: () => void;
  busy: boolean;
  onOpenStructure: (path: string) => void;
  onInsert: (text: string) => void;
}

function GroupSection({ group, expanded, onToggleOpen, selected, onSelect, onDownload, busy, onOpenStructure, onInsert }: GroupProps) {
  const [all, setAll] = useState(false);
  const paths = group.files.flatMap((f) => (f.archivePath ? [f.archivePath] : []));
  const picked = paths.filter((p) => selected.has(p)).length;
  const rows = all ? group.files : group.files.slice(0, ROWS);
  return (
    <section className="mb-0.5">
      <div className="group hover:bg-sunken flex items-center gap-2 rounded-md px-1 py-1">
        <input
          type="checkbox"
          checked={paths.length > 0 && picked === paths.length}
          ref={(el) => {
            if (el) el.indeterminate = picked > 0 && picked < paths.length;
          }}
          disabled={!paths.length}
          onChange={(event) => onSelect(paths, event.target.checked)}
          aria-label={`Select ${group.label}`}
          className={checkbox}
        />
        <button type="button" onClick={onToggleOpen} aria-expanded={expanded} className="flex min-w-0 flex-1 cursor-pointer items-center gap-1.5 text-left text-xs">
          <span className="text-faint w-2.5 shrink-0 text-[10px]">{expanded ? "\u25be" : "\u25b8"}</span>
          <span className="min-w-0 truncate font-medium" title={group.key || undefined}>
            {group.label}
          </span>
          {group.inChat && <span className="size-1.5 shrink-0 rounded-full" style={{ background: "var(--color-accent)" }} title="Used in this chat" />}
          <span className="text-faint shrink-0 text-[10px]">
            {group.files.length}
            {group.bytes ? ` \u00b7 ${formatBytes(group.bytes)}` : ""}
          </span>
        </button>
        {paths.length > 0 && (
          <button type="button" onClick={onDownload} disabled={busy} title="Download this folder as one ZIP" className={`${action} opacity-0 group-hover:opacity-100 focus-visible:opacity-100 touch:opacity-100 disabled:opacity-50`}>
            Download
          </button>
        )}
      </div>
      {expanded && (
        <ul className="mb-1 ml-3">
          {rows.map((file) => (
            <FileRow
              key={file.path}
              file={file}
              checked={file.archivePath !== null && selected.has(file.archivePath)}
              onCheck={(on) => file.archivePath && onSelect([file.archivePath], on)}
              onOpenStructure={onOpenStructure}
              onInsert={onInsert}
            />
          ))}
          {rows.length < group.files.length && (
            <li>
              <button type="button" onClick={() => setAll(true)} className={`${action} ml-6`}>
                Show all {group.files.length}
              </button>
            </li>
          )}
        </ul>
      )}
    </section>
  );
}

interface RowProps {
  file: CatalogFile;
  checked: boolean;
  onCheck: (on: boolean) => void;
  onOpenStructure: (path: string) => void;
  onInsert: (text: string) => void;
}

function FileRow({ file, checked, onCheck, onOpenStructure, onInsert }: RowProps) {
  return (
    <li className="group hover:bg-sunken flex items-center gap-2 rounded-md px-1 py-0.5 text-[11px]">
      <input type="checkbox" checked={checked} disabled={!file.archivePath} onChange={(event) => onCheck(event.target.checked)} aria-label={`Select ${file.label}`} className={checkbox} />
      <span className="min-w-0 flex-1 truncate" title={file.path}>
        {file.label}
      </span>
      <span className="text-faint shrink-0 font-mono text-[10px] uppercase">{file.ext}</span>
      {file.queryLength !== null && <span className="text-faint shrink-0 font-mono text-[10px]">{file.queryLength} aa</span>}
      {file.size !== null && <span className="text-faint w-12 shrink-0 text-right text-[10px] tabular-nums">{formatBytes(file.size)}</span>}
      <span className="flex shrink-0 items-center opacity-0 group-hover:opacity-100 focus-within:opacity-100 touch:opacity-100">
        {file.openable && (
          <button type="button" onClick={() => onOpenStructure(sharedRelative(file.path) ?? file.path)} className={action}>
            Open
          </button>
        )}
        <button type="button" onClick={() => onInsert(file.path)} title="Insert the path into the prompt" className={action}>
          Insert
        </button>
        {file.url && (
          <a href={file.url} download={file.path.split("/").pop()} title="Download" aria-label={`Download ${file.label}`} className={action}>
            {"\u2193"}
          </a>
        )}
      </span>
    </li>
  );
}
