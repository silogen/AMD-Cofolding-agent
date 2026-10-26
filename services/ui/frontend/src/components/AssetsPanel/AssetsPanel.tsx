// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useState } from "react";
import { callTool } from "../../lib/api";
import { ASSET_GROUPS, canOpen, groupByFolder, mergeAssets, parseSharedAssets, sessionAssets, sharedRelative, type Asset, type SharedAssetsListing } from "../../lib/assets";
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
  const groups = useMemo(() => groupByFolder(all, query), [all, query]);

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
            {listing.state === "ready" ? (listing.sharedDir ?? "/shared") : "/shared"} {"\u00b7"} {all.length} file{all.length === 1 ? "" : "s"}
            {fromSession.length > 0 && ` \u00b7 ${fromSession.length} from this session`}
          </p>
        </div>
        <button
          type="button"
          onClick={refresh}
          disabled={listing.state === "loading"}
          className="text-muted hover:text-body hover:bg-inset shrink-0 cursor-pointer rounded-md px-2 py-1 text-[11px] disabled:cursor-default disabled:opacity-50 touch:min-h-11"
        >
          {listing.state === "loading" ? "Loading\u2026" : "Refresh"}
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

      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-3">
        {listing.state === "error" && (
          <p className="border-subtle rounded-md border border-l-2 px-3 py-2 text-[11px]" style={{ borderLeftColor: "var(--color-accent)" }}>
            Could not list the shared volume: {listing.message}
          </p>
        )}
        {listing.state === "ready" && listing.note && <p className="text-muted px-1 text-[11px]">{listing.note}</p>}
        <p className="text-faint px-1 text-[11px] leading-snug">
          Paths are part of the service contract and cannot be fabricated by the UI. Insert one into the prompt to ground the agent.
        </p>
        {groups.map((group) => (
          <section key={group.folder}>
            <h3 className="text-muted mb-1.5 px-1 font-mono text-[11px]">
              {group.folder} <span className="text-faint">{"\u00b7"} {group.assets.length}</span>
            </h3>
            <div className="border-subtle overflow-x-auto rounded-lg border">
              <table className="w-full min-w-[34rem] table-fixed border-collapse text-[11px]">
                <colgroup>
                  <col />
                  <col className="w-48" />
                  <col className="w-28" />
                  <col className="w-40" />
                </colgroup>
                <thead>
                  <tr className="text-faint border-subtle border-b text-left text-[10px]">
                    <th className="px-3 py-1.5 font-normal">Path</th>
                    <th className="px-2 py-1.5 font-normal">Type</th>
                    <th className="px-2 py-1.5 font-normal">Producer</th>
                    <th className="px-2 py-1.5 text-right font-normal">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {group.assets.map((asset) => (
                    <AssetRow key={asset.path} asset={asset} onOpenStructure={onOpenStructure} onInsert={onInsert} />
                  ))}
                </tbody>
              </table>
            </div>
          </section>
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
  const kind = ASSET_GROUPS.find((g) => g.kind === asset.kind);
  return (
    <tr className="border-subtle group hover:bg-sunken border-b last:border-b-0">
      <td className="px-3 py-1.5">
        <div className="flex min-w-0 items-center gap-1">
          <span className="min-w-0 flex-1 truncate font-mono" title={asset.path} dir="rtl">
            &lrm;
            {asset.path}
          </span>
          <span className="opacity-0 group-hover:opacity-100 focus-within:opacity-100 touch:opacity-100">
            <CopyButton value={asset.path} label="Copy" />
          </span>
        </div>
      </td>
      <td className="whitespace-nowrap px-2 py-1.5">
        <span className="inline-flex items-center gap-1.5">
          <span className="text-muted">{kind?.label ?? asset.kind}</span>
          <span className="text-faint font-mono text-[10px] uppercase">{asset.ext}</span>
          {asset.queryLength !== null && <span className="text-faint font-mono text-[10px]">{"\u00b7"} {asset.queryLength} aa</span>}
        </span>
      </td>
      <td className="text-muted whitespace-nowrap px-2 py-1.5 font-mono text-[10px]">{asset.source === "session" ? asset.tool : "shared volume"}</td>
      <td className="whitespace-nowrap px-2 py-1 text-right">
        {openable && (
          <button
            type="button"
            onClick={() => onOpenStructure(sharedRelative(asset.path) ?? asset.path)}
            className="border-subtle hover:border-strong text-body cursor-pointer rounded-md border px-2 py-0.5 text-[11px] touch:min-h-11"
          >
            Open
          </button>
        )}
        <button
          type="button"
          onClick={() => onInsert(asset.path)}
          title="Insert the path into the prompt"
          className="text-muted hover:text-body hover:bg-inset cursor-pointer rounded-md px-2 py-1 text-[11px] touch:min-h-11"
        >
          Insert
        </button>
        {asset.url && (
          <a href={asset.url} download={asset.name} className="text-muted hover:text-body hover:bg-inset rounded-md px-2 py-1 text-[11px] touch:min-h-11">
            Download
          </a>
        )}
      </td>
    </tr>
  );
}