// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useState } from "react";
import { callTool } from "../../lib/api";
import { canOpen, groupAssets, mergeAssets, parseSharedAssets, sessionAssets, sharedRelative, type Asset, type SharedAssetsListing } from "../../lib/assets";
import { parseResult, type ToolEvent } from "../../lib/ttp";
import { CopyButton } from "../CopyButton/CopyButton";

interface Props {
  /** The session's tool calls, from `toolEvents(artifacts.steps)`; their files are listed too. */
  events?: ToolEvent[];
  /** Show a structure (path relative to the shared volume) in the 3D viewer. */
  onOpenStructure: (path: string) => void;
  /** Put text in the composer. */
  onInsert: (text: string) => void;
}

type Listing =
  | { state: "loading" }
  | { state: "error"; message: string }
  | { state: "ready"; assets: Asset[]; sharedDir: string | null; note: string | null; at: number };

/**
 * The files on the shared volume, grouped by what they are, plus the files
 * this session's tools wrote. Paths are the ones the tools accept, so
 * inserting one into a prompt grounds the agent instead of letting it guess.
 */
export function AssetsPanel({ events = [], onOpenStructure, onInsert }: Props) {
  const [listing, setListing] = useState<Listing>({ state: "loading" });
  const [reload, setReload] = useState(0);
  const [query, setQuery] = useState("");

  useEffect(() => {
    let cancelled = false;
    callTool<string>("list_shared_assets")
      .then((response) => {
        if (cancelled) return;
        const raw = parseResult(response.result) as SharedAssetsListing | null;
        setListing({
          state: "ready",
          assets: parseSharedAssets(raw),
          sharedDir: raw?.shared_dir ?? null,
          note: raw?.note ?? null,
          at: Date.now(),
        });
      })
      .catch((err: unknown) => {
        if (!cancelled) setListing({ state: "error", message: err instanceof Error ? err.message : String(err) });
      });
    return () => {
      cancelled = true;
    };
  }, [reload]);

  const fromSession = useMemo(() => sessionAssets(events), [events]);
  const all = useMemo(() => mergeAssets(listing.state === "ready" ? listing.assets : [], fromSession), [listing, fromSession]);
  const groups = useMemo(() => groupAssets(all, query), [all, query]);

  const refresh = () => {
    setListing({ state: "loading" });
    setReload((n) => n + 1);
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="border-subtle flex shrink-0 items-center gap-2 border-b px-3 py-2">
        <div className="min-w-0 flex-1">
          <h2 className="text-xs font-semibold">Shared assets</h2>
          <p className="text-faint truncate font-mono text-[10px]">
            {listing.state === "ready" ? (listing.sharedDir ?? "/shared") : "/shared"} · {all.length} file{all.length === 1 ? "" : "s"}
            {fromSession.length > 0 && ` · ${fromSession.length} from this session`}
          </p>
        </div>
        <button
          type="button"
          onClick={refresh}
          disabled={listing.state === "loading"}
          className="bg-inset text-muted hover:text-body shrink-0 cursor-pointer rounded-md px-2 py-1 text-[11px] font-medium disabled:cursor-default disabled:opacity-50 touch:min-h-11"
        >
          {listing.state === "loading" ? "Loading…" : "Refresh"}
        </button>
      </div>
      <div className="border-subtle shrink-0 border-b px-3 py-2">
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Filter by path"
          aria-label="Filter assets"
          className="bg-sunken border-subtle focus:border-strong w-full rounded-lg border px-2.5 py-1.5 text-xs outline-none"
        />
      </div>

      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3">
        {listing.state === "error" && (
          <p className="bg-sunken border-subtle rounded-lg border px-3 py-2 text-[11px]" style={{ color: "var(--color-accent)" }}>
            Could not list the shared volume: {listing.message}
          </p>
        )}
        {listing.state === "ready" && listing.note && <p className="text-muted bg-sunken border-subtle rounded-lg border px-3 py-2 text-[11px]">{listing.note}</p>}
        {groups.map((group) => (
          <details key={group.kind} open className="bg-raised border-subtle shadow-card overflow-hidden rounded-xl border">
            <summary className="border-subtle flex cursor-pointer list-none items-center gap-2 border-b px-3 py-2">
              <span aria-hidden="true" className="size-2 shrink-0 rounded-full" style={{ background: group.colour }} />
              <span className="text-xs font-semibold">{group.label}</span>
              <span className="text-faint ml-auto text-[11px]">{group.assets.length}</span>
            </summary>
            <ul>
              {group.assets.map((asset) => (
                <AssetRow key={asset.path} asset={asset} onOpenStructure={onOpenStructure} onInsert={onInsert} />
              ))}
            </ul>
          </details>
        ))}
        {listing.state !== "loading" && groups.length === 0 && (
          <p className="text-muted px-6 py-10 text-center text-xs">
            {query ? "No file matches the filter." : "No files yet. Structures, MSAs and cofold outputs on the shared volume appear here."}
          </p>
        )}
      </div>
    </div>
  );
}

function AssetRow({ asset, onOpenStructure, onInsert }: { asset: Asset; onOpenStructure: (path: string) => void; onInsert: (text: string) => void }) {
  const openable = canOpen(asset);
  return (
    <li className="border-subtle flex items-center gap-2 border-b px-3 py-1.5 text-[11px] last:border-b-0">
      <div className="min-w-0 flex-1">
        <p className="truncate font-mono font-semibold" title={asset.path}>
          {asset.name}
        </p>
        <p className="text-faint flex gap-2 truncate font-mono text-[10px]">
          <span className="truncate">{asset.dir || "/"}</span>
          {asset.queryLength !== null && <span className="shrink-0">query {asset.queryLength} aa</span>}
          {asset.source === "session" && <span className="shrink-0">· {asset.tool}</span>}
        </p>
      </div>
      <span className="text-faint shrink-0 text-[10px] uppercase">{asset.ext}</span>
      {openable && (
        <button
          type="button"
          onClick={() => onOpenStructure(sharedRelative(asset.path) ?? asset.path)}
          className="shrink-0 cursor-pointer rounded-md px-2 py-1 text-[11px] font-semibold text-white transition-opacity hover:opacity-90 touch:min-h-11"
          style={{ background: "var(--color-accent)" }}
        >
          View
        </button>
      )}
      <button
        type="button"
        onClick={() => onInsert(asset.path)}
        title="Insert the path into the prompt"
        className="text-muted hover:text-body hover:bg-inset shrink-0 cursor-pointer rounded-md px-2 py-1 text-[11px] font-medium touch:min-h-11"
      >
        Insert
      </button>
      {asset.url && (
        <a href={asset.url} download={asset.name} className="text-muted hover:text-body shrink-0 rounded-md px-2 py-1 text-[11px] font-medium touch:min-h-11">
          Download
        </a>
      )}
      <CopyButton value={asset.path} label="Copy" />
    </li>
  );
}
