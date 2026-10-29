// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useState, type FormEvent, type ReactNode } from "react";
import { callTool } from "../../lib/api";
import { loadTtp } from "../../lib/scoreWhy";
import { sendToViewer } from "../../lib/viewerBus";
import { openPdbEntry } from "../../lib/rcsb";
import { hasTargetFacts, mergeTtpTarget, mergeUniprot, payloadError, pocketBySequence, type PocketResidue, type TargetFacts, type Ttp } from "../../lib/ttp";
import { CopyButton } from "../CopyButton/CopyButton";

interface Props {
  /** What the session established, from `extractTargetFacts(toolEvents(artifacts.steps))`. */
  facts: TargetFacts;
  /** Put a prompt in the composer. */
  onAsk?: (text: string) => void;
}

type Lookup = { state: "idle" } | { state: "busy" } | { state: "error"; message: string } | { state: "done"; payload: unknown };

const SEQUENCE_PREVIEW = 120;
const POCKET_PREVIEW = 24;

/**
 * The protein target: UniProt identity, sequence, structures and the pocket
 * the profiled ligands contact. Facts come from the session's tool calls; a
 * UniProt lookup made here fills in what the session did not fetch.
 */
export function TargetCard({ facts, onAsk }: Props) {
  const [lookup, setLookup] = useState<Lookup>({ state: "idle" });
  const [query, setQuery] = useState("");
  const [showSequence, setShowSequence] = useState(false);
  const [showPocket, setShowPocket] = useState(false);

  const [saved, setSaved] = useState<{ id: string; ttp: Ttp | null } | null>(null);
  const savedId = !hasTargetFacts(facts) ? facts.ttpId : null;
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
  const base = savedId && saved?.id === savedId && saved.ttp ? mergeTtpTarget(facts, saved.ttp, "get_ttp") : facts;
  const shown = lookup.state === "done" ? mergeUniprot(base, lookup.payload, "lookup") : base;
  const known = hasTargetFacts(shown);
  const label = shown.gene ?? shown.name ?? shown.uniprot ?? "the target";

  const runLookup = async (event: FormEvent) => {
    event.preventDefault();
    const text = query.trim() || shown.uniprot || shown.gene || "";
    if (!text) return;
    setLookup({ state: "busy" });
    try {
      const response = await callTool("fetch_target_info", { query: text });
      const error = payloadError(response.result);
      setLookup(error ? { state: "error", message: error } : { state: "done", payload: response.result });
    } catch (err) {
      setLookup({ state: "error", message: err instanceof Error ? err.message : String(err) });
    }
  };

  const focus = (residues: PocketResidue[]) =>
    sendToViewer({
      action: {
        kind: "residues",
        residues: residues.filter((r) => r.resnum !== null).map((r) => ({ chain: r.chain, resnum: r.resnum as number, resname: r.resname })),
        mode: "focus",
      },
      source: "target",
    });

  const [opening, setOpening] = useState<{ id: string; error?: string } | null>(null);
  const openEntry = async (id: string) => {
    setOpening({ id });
    try {
      await openPdbEntry(id);
      setOpening(null);
    } catch (err) {
      setOpening({ id, error: err instanceof Error ? err.message : String(err) });
    }
  };

  const pocket = showPocket ? shown.pocket : shown.pocket.slice(0, POCKET_PREVIEW);
  const sequence = shown.sequence ?? "";

  return (
    <section className="bg-raised border-subtle shadow-card overflow-hidden rounded-xl border">
      <header className="border-subtle flex items-center gap-2 border-b px-3 py-2">
        <span aria-hidden="true" className="size-2 shrink-0 rounded-full" style={{ background: "var(--color-structure)" }} />
        <span className="text-faint shrink-0 text-[10px] font-semibold uppercase">Target</span>
        <span className="min-w-0 truncate text-xs font-semibold" title={shown.name ?? undefined}>
          {known ? (shown.name ?? shown.gene ?? shown.uniprot ?? "Protein target") : "No target yet"}
        </span>
        {shown.uniprot && (
          <a
            href={`https://www.uniprot.org/uniprotkb/${encodeURIComponent(shown.uniprot)}/entry`}
            target="_blank"
            rel="noreferrer"
            className="ml-auto shrink-0 rounded-full border px-1.5 py-px font-mono text-[10px] font-semibold hover:opacity-80"
            style={{ color: "var(--color-structure)", borderColor: "var(--color-structure)" }}
            title="Open in UniProt"
          >
            {shown.uniprot}
          </a>
        )}
        {shown.reviewed !== null && (
          <span className={`text-faint shrink-0 text-[10px] ${shown.uniprot ? "" : "ml-auto"}`} title={shown.reviewed ? "Swiss-Prot (reviewed)" : "TrEMBL (unreviewed)"}>
            {shown.reviewed ? "reviewed" : "unreviewed"}
          </span>
        )}
      </header>

      {known ? (
        <div className="space-y-3 px-3 py-2">
          <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-[11px] sm:grid-cols-3">
            <Fact label="Gene">{shown.gene}</Fact>
            <Fact label="Organism">{shown.organism}</Fact>
            <Fact label="Entry">{shown.entryName}</Fact>
            <Fact label="Length">{shown.sequenceLength !== null ? `${shown.sequenceLength} aa` : null}</Fact>
            <Fact label="Chains">{shown.chains.length ? shown.chains.join(", ") : null}</Fact>
            <Fact label="TTP">{shown.ttpId}</Fact>
            <Fact label="MSA" title={shown.msaPath ?? undefined}>
              {shown.msaPath?.split("/").pop()}
            </Fact>
            <Fact label="Receptor file" title={shown.proteinPdbPath ?? undefined}>
              {shown.proteinPdbPath?.split("/").pop()}
            </Fact>
          </dl>

          {shown.pdbIds.length > 0 && (
            <div>
              <h3 className="text-faint mb-1 text-[10px] font-semibold uppercase">
                PDB entries{shown.pdbCount !== null && shown.pdbCount > shown.pdbIds.length ? ` · ${shown.pdbIds.length} of ${shown.pdbCount}` : ""}
              </h3>
              <div className="flex flex-wrap gap-1">
                {shown.pdbIds.map((id) => (
                  <span key={id} className="bg-sunken group inline-flex items-center rounded-md font-mono text-[10px]">
                    <button
                      type="button"
                      onClick={() => void openEntry(id)}
                      disabled={opening?.id === id && !opening.error}
                      title={`Open ${id} in the 3D viewer`}
                      className="hover:bg-inset cursor-pointer rounded-l-md py-px pr-1 pl-1.5 disabled:cursor-wait disabled:opacity-60"
                      style={{ color: "var(--color-structure)" }}
                    >
                      {id}
                    </button>
                    <a
                      href={`https://www.rcsb.org/structure/${encodeURIComponent(id)}`}
                      target="_blank"
                      rel="noreferrer"
                      title={`${id} on rcsb.org`}
                      aria-label={`${id} on rcsb.org`}
                      className="text-faint hover:text-body rounded-r-md py-px pr-1.5 pl-0.5"
                    >
                      ↗
                    </a>
                  </span>
                ))}
              </div>
              {opening?.error && (
                <p className="mt-1 text-[10px]" style={{ color: "var(--color-warn)" }}>
                  {opening.error}
                </p>
              )}
            </div>
          )}

          {sequence && (
            <div>
              <div className="mb-1 flex items-center gap-2">
                <h3 className="text-faint text-[10px] font-semibold uppercase">Sequence</h3>
                {sequence.length > SEQUENCE_PREVIEW && (
                  <button type="button" onClick={() => setShowSequence((v) => !v)} className="text-muted hover:text-body ml-auto cursor-pointer text-[11px] font-medium">
                    {showSequence ? "Less" : "All"}
                  </button>
                )}
                <span className={sequence.length > SEQUENCE_PREVIEW ? "" : "ml-auto"}>
                  <CopyButton value={sequence} label="Copy" />
                </span>
              </div>
              <p className="bg-sunken border-subtle max-h-40 overflow-y-auto rounded-lg border px-2 py-1.5 font-mono text-[10px] leading-relaxed break-all">
                {showSequence || sequence.length <= SEQUENCE_PREVIEW ? sequence : `${sequence.slice(0, SEQUENCE_PREVIEW)}…`}
              </p>
            </div>
          )}

          {shown.pocket.length > 0 && (
            <div>
              <div className="mb-1 flex items-center gap-2">
                <h3 className="text-faint text-[10px] font-semibold uppercase">
                  Pocket · {shown.pocket.length} residues
                  <span className="font-normal normal-case">
                    {" "}
                    from {shown.profiledComplexes} profiled complex{shown.profiledComplexes === 1 ? "" : "es"}
                  </span>
                </h3>
                <button
                  type="button"
                  onClick={() => focus(shown.pocket)}
                  className="text-muted hover:text-body ml-auto cursor-pointer text-[11px] font-medium"
                  title="Focus these residues in the 3D viewer"
                >
                  Show in viewer
                </button>
              </div>
              <div className="flex flex-wrap gap-1">
                {pocket.map((residue) => (
                  <button
                    key={`${residue.chain}:${residue.label}`}
                    type="button"
                    onClick={() => focus([residue])}
                    title={`${residue.chain ? `chain ${residue.chain} · ` : ""}${residue.contacts} contact${residue.contacts === 1 ? "" : "s"}: ${residue.types.join(", ")}`}
                    className="bg-sunken border-subtle hover:border-strong cursor-pointer rounded-md border px-1.5 py-px font-mono text-[10px]"
                    style={{ color: "var(--color-structure)" }}
                  >
                    {residue.chain && shown.chains.length > 1 ? `${residue.chain}:` : ""}
                    {residue.label}
                    <span className="text-faint ml-1">{residue.contacts}</span>
                  </button>
                ))}
                {shown.pocket.length > POCKET_PREVIEW && (
                  <button type="button" onClick={() => setShowPocket((v) => !v)} className="text-muted hover:text-body cursor-pointer px-1 text-[11px] font-medium">
                    {showPocket ? "Fewer" : `+${shown.pocket.length - POCKET_PREVIEW} more`}
                  </button>
                )}
              </div>
              <p className="text-faint mt-1 font-mono text-[10px]">
                {pocketBySequence(shown.pocket)
                  .map((r) => r.label)
                  .join(" ")}
              </p>
            </div>
          )}

          {shown.function && <p className="text-muted line-clamp-4 text-[11px] leading-relaxed">{shown.function}</p>}
        </div>
      ) : (
        <p className="text-muted px-3 py-3 text-[11px]">
          The session has not named a protein yet. Look one up on UniProt, or ask the agent to fetch the target.
        </p>
      )}

      <form onSubmit={runLookup} className="border-subtle flex items-center gap-1 border-t px-3 py-1.5">
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder={shown.uniprot ? `Refresh ${shown.uniprot} or look up another (e.g. BRD4 human)` : "UniProt accession or gene + organism"}
          aria-label="UniProt query"
          className="bg-sunken border-subtle focus:border-strong min-w-0 flex-1 rounded-md border px-2 py-1 text-[11px] outline-none"
        />
        <button
          type="submit"
          disabled={lookup.state === "busy" || !(query.trim() || shown.uniprot || shown.gene)}
          className="cursor-pointer rounded-md px-2 py-1 text-[11px] font-semibold text-white transition-opacity hover:opacity-90 disabled:cursor-default disabled:opacity-50 touch:min-h-11"
          style={{ background: "var(--color-accent)" }}
        >
          {lookup.state === "busy" ? "Looking up…" : "UniProt"}
        </button>
        {onAsk && known && (
          <button
            type="button"
            onClick={() =>
              onAsk(
                `Use ${label}${shown.uniprot ? ` (UniProt ${shown.uniprot})` : ""} as the target: summarise what is known about its binding pocket and suggest a TTP for lead optimisation.`,
              )
            }
            className="text-muted hover:text-body hover:bg-inset cursor-pointer rounded-md px-2 py-1 text-[11px] font-medium touch:min-h-11"
          >
            Ask…
          </button>
        )}
      </form>
      {lookup.state === "error" && (
        <p className="border-subtle border-t px-3 py-1.5 text-[11px]" style={{ color: "var(--color-accent)" }}>
          {lookup.message}
        </p>
      )}
    </section>
  );
}

function Fact({ label, title, children }: { label: string; title?: string; children: ReactNode }) {
  if (children === null || children === undefined || children === "") return null;
  return (
    <div className="min-w-0">
      <dt className="text-faint truncate">{label}</dt>
      <dd className="truncate font-mono" title={title ?? (typeof children === "string" ? children : undefined)}>
        {children}
      </dd>
    </div>
  );
}
