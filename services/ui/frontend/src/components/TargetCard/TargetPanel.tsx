// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useState, type FormEvent } from "react";
import { callTool } from "../../lib/api";
import { loadTtp } from "../../lib/scoreWhy";
import { extractTargets, orderTargets, scopedTarget, targetKey, targetName, withLookup, type TargetEntry } from "../../lib/targets";
import { emptyTargetFacts, latestTtp, mergeUniprot, payloadError, type ToolEvent, type Ttp } from "../../lib/ttp";
import { TargetCard } from "./TargetCard";

interface Props {
  /** The session's tool calls, from `toolEvents(artifacts.steps)`. */
  events: ToolEvent[];
  /** Put a prompt in the composer. */
  onAsk?: (text: string) => void;
}

type Lookup = { state: "idle" } | { state: "busy" } | { state: "error"; message: string };

/**
 * The Target tab: the proteins this conversation worked on, one shown at a
 * time. A lookup here uses the same fetch_target_info call as the agent, so
 * it brings the same names, function and PDB cross-references.
 */
export function TargetPanel({ events, onAsk }: Props) {
  const session = useMemo(() => extractTargets(events), [events]);
  const [found, setFound] = useState<unknown[]>([]);
  const [picked, setPicked] = useState<string | null>(null);
  const [lookup, setLookup] = useState<Lookup>({ state: "idle" });

  // A run that only named its profile: the TTP store knows the target it holds.
  const ttpId = useMemo(() => latestTtp(events).id, [events]);
  const [saved, setSaved] = useState<{ id: string; ttp: Ttp | null } | null>(null);
  const savedId = session.length === 0 ? ttpId : null;
  useEffect(() => {
    if (!savedId) return;
    let cancelled = false;
    void loadTtp(savedId).then((ttp) => {
      if (!cancelled) setSaved({ id: savedId, ttp });
    });
    return () => {
      cancelled = true;
    };
  }, [savedId]);

  const entries = useMemo(() => {
    const stored = savedId && saved?.id === savedId && saved.ttp ? saved.ttp : null;
    const base = stored ? extractTargets([{ type: "tool_result", name: "get_ttp", result: stored }]) : session;
    return found.reduce<TargetEntry[]>((list, payload) => withLookup(list, payload), base);
  }, [session, found, saved, savedId]);

  const current = scopedTarget(entries, picked, ttpId);
  const ordered = orderTargets(entries);

  const runLookup = async (query: string) => {
    setLookup({ state: "busy" });
    try {
      const { result } = await callTool("fetch_target_info", { query });
      const error = payloadError(result);
      if (error) {
        setLookup({ state: "error", message: error });
        return;
      }
      setFound((list) => [...list, result]);
      setPicked(targetKey(mergeUniprot(emptyTargetFacts(), result, "lookup")));
      setLookup({ state: "idle" });
    } catch (err) {
      setLookup({ state: "error", message: err instanceof Error ? err.message : String(err) });
    }
  };

  return (
    <div className="space-y-2">
      {ordered.length > 1 && (
        <div role="tablist" aria-label="Targets" className="flex gap-1 overflow-x-auto pb-0.5">
          {ordered.map((target) => {
            const selected = target.key === current?.key;
            return (
              <button
                key={target.key}
                type="button"
                role="tab"
                aria-selected={selected}
                onClick={() => setPicked(target.key)}
                title={[target.facts.name ?? targetName(target.facts), target.facts.organism, target.inSession ? "this chat" : "looked up here"].filter(Boolean).join(" · ")}
                className={`shrink-0 cursor-pointer rounded-full border px-2 py-0.5 text-[11px] whitespace-nowrap ${selected ? "bg-sunken" : "border-subtle text-muted hover:text-body"}`}
                style={selected ? { borderColor: "var(--color-structure)", color: "var(--color-structure)" } : undefined}
              >
                {target.inSession && <span aria-hidden="true" className="mr-1 inline-block size-1 rounded-full align-middle" style={{ background: "var(--color-accent)" }} />}
                {targetName(target.facts)}
              </button>
            );
          })}
        </div>
      )}

      <TargetCard facts={current?.facts ?? emptyTargetFacts()} onAsk={onAsk}>
        <LookupField busy={lookup.state === "busy"} known={current?.facts.uniprot ?? null} onRun={(query) => void runLookup(query)} />
      </TargetCard>

      {lookup.state === "error" && (
        <p className="px-1 text-[11px]" style={{ color: "var(--color-accent)" }}>
          {lookup.message}
        </p>
      )}
    </div>
  );
}

/** "Look up…", which opens one field for an accession or a gene and organism. */
function LookupField({ busy, known, onRun }: { busy: boolean; known: string | null; onRun: (query: string) => void }) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");

  if (!open) {
    return (
      <button type="button" onClick={() => setOpen(true)} className="text-muted hover:text-body hover:bg-inset cursor-pointer rounded-md px-2 py-1 text-[11px] font-medium touch:min-h-11">
        {known ? "Another target…" : "Look up…"}
      </button>
    );
  }

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (query.trim()) onRun(query.trim());
  };

  return (
    <form onSubmit={submit} className="flex min-w-0 flex-1 items-center gap-1">
      <input
        autoFocus
        type="search"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        placeholder="UniProt accession, or gene and organism"
        aria-label="Look a target up on UniProt"
        className="bg-sunken border-subtle focus:border-strong min-w-0 flex-1 rounded-md border px-2 py-1 text-[11px] outline-none"
      />
      <button
        type="submit"
        disabled={busy || !query.trim()}
        className="cursor-pointer rounded-md px-2 py-1 text-[11px] font-semibold text-white transition-opacity hover:opacity-90 disabled:cursor-default disabled:opacity-50 touch:min-h-11"
        style={{ background: "var(--color-accent)" }}
      >
        {busy ? "Looking…" : "Find"}
      </button>
    </form>
  );
}
