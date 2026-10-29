// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useState, type ReactNode } from "react";
import { sendToViewer } from "../../lib/viewerBus";
import { openPdbEntry } from "../../lib/rcsb";
import { hasTargetFacts, structureHint, targetLigands, type PocketResidue, type TargetFacts } from "../../lib/ttp";
import { CopyButton } from "../CopyButton/CopyButton";

interface Props {
  /** One protein, from `extractTargets` or `extractTargetFacts`. */
  facts: TargetFacts;
  /** Put a prompt in the composer. */
  onAsk?: (text: string) => void;
  /** Shown in the footer, before "Ask…": the target lookup. */
  children?: ReactNode;
}

const POCKET_PREVIEW = 24;
const LIGAND_PREVIEW = 8;

const count = (n: number) => n.toLocaleString("en");

/**
 * One protein target: who it is, the facts the session established, the PDB
 * entries UniProt cross-references and the pocket the profiled ligands
 * contacted. The sequence stays folded away.
 */
export function TargetCard({ facts, onAsk, children }: Props) {
  const [showSequence, setShowSequence] = useState(false);
  const [showPocket, setShowPocket] = useState(false);
  const [opening, setOpening] = useState<{ id: string; error?: string } | null>(null);

  const known = hasTargetFacts(facts);
  const title = facts.gene ?? facts.name ?? facts.uniprot ?? "Protein target";
  const subtitle = facts.gene && facts.name !== facts.gene ? facts.name : null;
  const sequence = facts.sequence ?? "";
  const ligands = targetLigands(facts);
  const pocket = showPocket ? facts.pocket : facts.pocket.slice(0, POCKET_PREVIEW);

  const focus = (residues: PocketResidue[]) =>
    sendToViewer({
      action: {
        kind: "residues",
        residues: residues.filter((r) => r.resnum !== null).map((r) => ({ chain: r.chain, resnum: r.resnum as number, resname: r.resname })),
        mode: "focus",
      },
      source: "target",
    });

  const openEntry = async (id: string) => {
    setOpening({ id });
    try {
      await openPdbEntry(id);
      setOpening(null);
    } catch (err) {
      setOpening({ id, error: err instanceof Error ? err.message : String(err) });
    }
  };

  return (
    <section className="bg-raised border-subtle shadow-card overflow-hidden rounded-xl border">
      {known ? (
        <div className="space-y-3 px-3 py-2.5">
          <div className="flex items-start gap-2">
            <span aria-hidden="true" className="mt-1.5 size-2 shrink-0 rounded-full" style={{ background: "var(--color-structure)" }} />
            <div className="min-w-0 flex-1">
              <h3 className="truncate text-sm font-semibold" title={facts.name ?? title}>
                {title}
              </h3>
              {subtitle && <p className="text-muted truncate text-[11px]">{subtitle}</p>}
              <p className="text-faint truncate text-[10px]">
                {[facts.organism, facts.entryName, facts.reviewed === null ? null : facts.reviewed ? "reviewed" : "unreviewed"].filter(Boolean).join(" · ") || "target of this session"}
              </p>
            </div>
            {facts.uniprot && (
              <a
                href={`https://www.uniprot.org/uniprotkb/${encodeURIComponent(facts.uniprot)}/entry`}
                target="_blank"
                rel="noreferrer"
                title={`${facts.uniprot} on uniprot.org`}
                className="shrink-0 rounded-full border px-1.5 py-px font-mono text-[10px] font-semibold hover:opacity-80"
                style={{ color: "var(--color-structure)", borderColor: "var(--color-structure)" }}
              >
                {facts.uniprot}
              </a>
            )}
          </div>

          {facts.function && (
            <p className="text-muted line-clamp-2 text-[11px] leading-relaxed" title={facts.function}>
              {facts.function}
            </p>
          )}

          <dl className="flex flex-wrap gap-x-5 gap-y-1.5">
            <Fact
              label="Length"
              title={facts.uniprotSequence && facts.uniprotSequence.length !== facts.sequenceLength ? `a construct; the UniProt sequence has ${count(facts.uniprotSequence.length)} aa` : undefined}
            >
              {facts.sequenceLength !== null ? `${count(facts.sequenceLength)} aa` : null}
            </Fact>
            <Fact label="Chains">{facts.chains.length ? facts.chains.join(", ") : null}</Fact>
            <Fact label="Profile">{facts.ttpId}</Fact>
            <Fact label="MSA" title={facts.msaPath ?? undefined}>
              {facts.msaPath?.split("/").pop()}
            </Fact>
            <Fact label="Receptor" title={facts.proteinPdbPath ?? undefined}>
              {facts.proteinPdbPath?.split("/").pop()}
            </Fact>
          </dl>

          {facts.pdbIds.length > 0 && (
            <Group label="PDB entries" hint={facts.pdbCount !== null && facts.pdbCount > facts.pdbIds.length ? `${facts.pdbIds.length} of ${count(facts.pdbCount)}` : undefined}>
              {facts.pdbIds.map((id) => (
                <span key={id} className="bg-sunken inline-flex items-center rounded-md font-mono text-[10px]">
                  <a
                    href={`https://www.rcsb.org/structure/${encodeURIComponent(id)}`}
                    target="_blank"
                    rel="noreferrer"
                    title={[`${id} on rcsb.org`, structureHint(facts, id)].filter(Boolean).join(" · ")}
                    className="hover:bg-inset rounded-l-md py-px pr-1 pl-1.5"
                    style={{ color: "var(--color-structure)" }}
                  >
                    {id}
                  </a>
                  <button
                    type="button"
                    onClick={() => void openEntry(id)}
                    disabled={opening?.id === id && !opening.error}
                    title={`Open ${id} in the 3D viewer`}
                    aria-label={`Open ${id} in the 3D viewer`}
                    className="text-faint hover:text-body cursor-pointer rounded-r-md py-px pr-1.5 pl-0.5 disabled:cursor-wait disabled:opacity-60"
                  >
                    3D
                  </button>
                </span>
              ))}
              {opening?.error && (
                <p className="w-full text-[10px]" style={{ color: "var(--color-warn)" }}>
                  {opening.error}
                </p>
              )}
            </Group>
          )}

          {ligands.length > 0 && (
            <Group label="Bound ligands" hint="in those entries">
              {ligands.slice(0, LIGAND_PREVIEW).map((ligand) => (
                <a
                  key={ligand.code}
                  href={`https://www.rcsb.org/ligand/${encodeURIComponent(ligand.code)}`}
                  target="_blank"
                  rel="noreferrer"
                  title={[ligand.name, ligand.pdbIds.join(", ")].filter(Boolean).join(" · ")}
                  className="bg-sunken border-subtle hover:border-strong rounded-md border px-1.5 py-px font-mono text-[10px]"
                >
                  {ligand.code}
                </a>
              ))}
            </Group>
          )}

          {facts.pocket.length > 0 && (
            <Group
              label="Pocket"
              hint={`${facts.pocket.length} residues · ${facts.profiledComplexes} profiled complex${facts.profiledComplexes === 1 ? "" : "es"}`}
              action={
                <button type="button" onClick={() => focus(facts.pocket)} title="Focus these residues in the 3D viewer" className="text-muted hover:text-body cursor-pointer text-[11px] font-medium">
                  Show in viewer
                </button>
              }
            >
              {pocket.map((residue) => (
                <button
                  key={`${residue.chain}:${residue.label}`}
                  type="button"
                  onClick={() => focus([residue])}
                  title={`${residue.chain ? `chain ${residue.chain} · ` : ""}${residue.contacts} contact${residue.contacts === 1 ? "" : "s"}: ${residue.types.join(", ")}`}
                  className="bg-sunken border-subtle hover:border-strong cursor-pointer rounded-md border px-1.5 py-px font-mono text-[10px]"
                  style={{ color: "var(--color-structure)" }}
                >
                  {residue.chain && facts.chains.length > 1 ? `${residue.chain}:` : ""}
                  {residue.label}
                  <span className="text-faint ml-1">{residue.contacts}</span>
                </button>
              ))}
              {facts.pocket.length > POCKET_PREVIEW && (
                <button type="button" onClick={() => setShowPocket((v) => !v)} className="text-muted hover:text-body cursor-pointer px-1 text-[11px] font-medium">
                  {showPocket ? "Fewer" : `+${facts.pocket.length - POCKET_PREVIEW} more`}
                </button>
              )}
            </Group>
          )}

          {sequence && (
            <div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setShowSequence((v) => !v)}
                  aria-expanded={showSequence}
                  className="text-muted hover:text-body cursor-pointer text-[11px] font-medium"
                >
                  {showSequence ? "▾" : "▸"} Sequence <span className="text-faint font-normal">{count(sequence.length)} aa</span>
                </button>
                <span className="ml-auto">
                  <CopyButton value={sequence} label="Copy" />
                </span>
              </div>
              {showSequence && (
                <p className="bg-sunken border-subtle mt-1 max-h-40 overflow-y-auto rounded-lg border px-2 py-1.5 font-mono text-[10px] leading-relaxed break-all">{sequence}</p>
              )}
            </div>
          )}
        </div>
      ) : (
        <p className="text-muted px-3 py-2.5 text-[11px]">No target yet. Ask the agent to fetch one, or look one up.</p>
      )}

      {(children || (onAsk && known)) && (
        <div className="border-subtle flex items-center gap-1 border-t px-3 py-1.5">
          {children}
          {onAsk && known && (
            <button
              type="button"
              onClick={() =>
                onAsk(
                  `Use ${facts.gene ?? facts.name ?? facts.uniprot ?? "the target"}${facts.uniprot ? ` (UniProt ${facts.uniprot})` : ""} as the target: summarise what is known about its binding pocket and suggest a TTP for lead optimisation.`,
                )
              }
              className="text-muted hover:text-body hover:bg-inset ml-auto shrink-0 cursor-pointer rounded-md px-2 py-1 text-[11px] font-medium touch:min-h-11"
            >
              Ask…
            </button>
          )}
        </div>
      )}
    </section>
  );
}

function Fact({ label, title, children }: { label: string; title?: string; children: ReactNode }) {
  if (children === null || children === undefined || children === "") return null;
  return (
    <div className="min-w-0">
      <dt className="text-faint truncate text-[10px]">{label}</dt>
      <dd className="truncate font-mono text-[11px]" title={title ?? (typeof children === "string" ? children : undefined)}>
        {children}
      </dd>
    </div>
  );
}

function Group({ label, hint, action, children }: { label: string; hint?: string; action?: ReactNode; children: ReactNode }) {
  return (
    <div>
      <div className="mb-1 flex items-center gap-2">
        <h4 className="text-faint text-[10px] font-semibold uppercase">
          {label}
          {hint && <span className="ml-1 font-normal normal-case">{hint}</span>}
        </h4>
        {action && <span className="ml-auto">{action}</span>}
      </div>
      <div className="flex flex-wrap items-center gap-1">{children}</div>
    </div>
  );
}
