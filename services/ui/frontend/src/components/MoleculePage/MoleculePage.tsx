// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useState } from "react";
import type { Theme } from "../../hooks/useTheme";
import { useRDKit } from "../../hooks/useRDKit";
import { sameFile, type Artifacts } from "../../lib/artifacts";
import { admetPrompt, candidatesOf, cofoldPrompt } from "../../lib/candidates";
import { loadJobStructures, openRunStructure, runStructurePanel, type JobStructure } from "../../lib/jobStructures";
import { loadRunTable, moleculeView, runRowContext, type MoleculeContext } from "../../lib/moleculePage";
import { canonical, descriptors } from "../../lib/rdkit";
import type { RunTable } from "../../lib/runResults";
import { sendToViewer } from "../../lib/viewerBus";
import { CopyButton } from "../CopyButton/CopyButton";
import { MoleculeFacts } from "../MoleculeFacts/MoleculeFacts";
import { MolDepiction } from "../MolDepiction/MolDepiction";
import { ScoreWhy } from "../RunResults/ScoreWhy";
import { StarButton } from "../Shortlist/StarButton";

interface Props {
  smiles: string;
  context?: MoleculeContext;
  artifacts: Artifacts;
  theme: Theme;
  sessionId: string | null;
  onBack: () => void;
  onAsk: (prompt: string) => void;
}

const action = "text-muted hover:text-body hover:bg-inset cursor-pointer rounded-md px-2 py-1 text-[11px]";

/** One molecule: its drawing, where it came from, and every value the conversation has for it. */
export function MoleculePage({ smiles, context, artifacts, theme, sessionId, onBack, onAsk }: Props) {
  const { rdkit } = useRDKit();
  const canonicalOf = useMemo(() => {
    if (!rdkit) return undefined;
    const seen = new Map<string, string>();
    return (value: string) => {
      let out = seen.get(value);
      if (out === undefined) seen.set(value, (out = canonical(rdkit, value) ?? value));
      return out;
    };
  }, [rdkit]);
  const candidate = useMemo(() => {
    const all = candidatesOf(artifacts);
    const wanted = canonicalOf?.(smiles);
    return all.find((c) => c.smiles === smiles) ?? (wanted ? all.find((c) => canonicalOf?.(c.smiles) === wanted) : undefined);
  }, [artifacts, smiles, canonicalOf]);
  const run = context?.run ?? null;
  const [fetched, setFetched] = useState<{ key: string; entry: JobStructure | null } | null>(null);
  const [table, setTable] = useState<{ run: string; table: RunTable | null } | null>(null);
  const key = `${run} ${smiles}`;
  const needsRun = run !== null && !context?.structure;
  const needsRow = run !== null && !context?.components;
  useEffect(() => {
    if (!needsRun || !run) return;
    let cancelled = false;
    loadJobStructures(run)
      .then((map) => !cancelled && setFetched({ key, entry: map.get(smiles) ?? null }))
      .catch(() => !cancelled && setFetched({ key, entry: null }));
    return () => {
      cancelled = true;
    };
  }, [needsRun, run, smiles, key]);
  useEffect(() => {
    if (!needsRow || !run) return;
    let cancelled = false;
    loadRunTable(run)
      .then((loaded) => !cancelled && setTable({ run, table: loaded }))
      .catch(() => !cancelled && setTable({ run, table: null }));
    return () => {
      cancelled = true;
    };
  }, [needsRow, run]);
  const runTable = run && table?.run === run ? table.table : null;
  const merged = useMemo(() => {
    const row = runTable && run ? runRowContext(runTable, smiles, run, canonicalOf) : null;
    if (!row) return context;
    const given = Object.fromEntries(Object.entries(context ?? {}).filter(([, value]) => value !== undefined && value !== null)) as MoleculeContext;
    return { ...row, ...given };
  }, [runTable, run, smiles, canonicalOf, context]);

  const view = useMemo(
    () =>
      moleculeView({
        smiles,
        context: merged,
        candidate,
        fetched: fetched?.key === key ? fetched.entry : null,
        descriptors: rdkit ? descriptors(rdkit, smiles) : null,
        predictions: artifacts.properties,
        key: canonicalOf,
      }),
    [smiles, merged, candidate, fetched, key, rdkit, artifacts.properties, canonicalOf],
  );
  const [notice, setNotice] = useState<string | null>(null);
  const session = sessionId ?? "new";

  function openComplex(path: string) {
    if (artifacts.structures.some(({ item }) => sameFile(item.path, path))) {
      sendToViewer({ path, action: { kind: "ligand" }, source: "molecule" });
      return;
    }
    const panel = run ? runStructurePanel(path, smiles, run, merged?.step ?? fetched?.entry?.step ?? null, view.cofold ?? undefined) : null;
    if (!panel) {
      setNotice("The complex is not on the shared volume.");
      return;
    }
    openRunStructure(session, panel);
  }

  const why = merged?.ttpId && merged.components?.length ? Object.fromEntries(merged.components.flatMap((c) => [[c.name, c.value], [`${c.name} (raw)`, c.raw ?? null]])) : null;
  const scores = () => ({
    ...Object.fromEntries((merged?.components ?? []).flatMap((c) => (c.value === null ? [] : [[c.name, c.value]]))),
    ...(view.score !== null ? { Score: view.score } : {}),
  });

  return (
    <div className="flex min-h-0 flex-1 flex-col" aria-label="Molecule">
      <header className="border-subtle flex shrink-0 items-center gap-2 border-b px-2 py-1.5">
        <button type="button" onClick={onBack} aria-label="Back to the list" title="Back" className="text-muted hover:text-body hover:bg-inset grid size-7 cursor-pointer place-items-center rounded-md text-sm">
          ←
        </button>
        <h2 className="min-w-0 truncate text-xs font-semibold" title={view.name ?? smiles}>
          {view.name ?? "Molecule"}
        </h2>
        <span className="text-faint min-w-0 truncate text-[11px]" title={view.tools.join(" · ") || undefined}>
          {view.origin}
        </span>
        <StarButton smiles={smiles} label={view.name} source={run ?? candidate?.origin} session={session} values={scores} className="ml-auto" />
      </header>

      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3">
        <div className="flex flex-wrap gap-3">
          <div className="bg-sunken border-subtle w-56 shrink-0 rounded-lg border p-1">
            <MolDepiction smiles={smiles} theme={theme} width={320} height={220} label={view.name ?? undefined} className="h-40 w-full" />
          </div>
          <div className="flex min-w-48 flex-1 flex-col gap-2">
            <div className="flex items-start gap-1">
              <code className="text-muted min-w-0 flex-1 font-mono text-[11px] break-all">{smiles}</code>
              <CopyButton value={smiles} label="Copy" />
            </div>
            <div className="-ml-2 flex flex-wrap">
              {!view.structurePath && (
                <button type="button" className={action} onClick={() => onAsk(cofoldPrompt([{ smiles, name: view.name ?? undefined }]))}>
                  Cofold…
                </button>
              )}
              {!view.predictions.some((group) => group.title === "ADMET") && (
                <button type="button" className={action} onClick={() => onAsk(admetPrompt([{ smiles }]))}>
                  ADMET…
                </button>
              )}
              <button type="button" className={action} onClick={() => onAsk(`Generate analogues of ${smiles} with Mol2Mol and score them against {TARGET}.`)}>
                Analogues…
              </button>
            </div>
          </div>
        </div>

        {notice && <p className="text-faint text-[11px]">{notice}</p>}

        <div className="@container">
          <MoleculeFacts
            score={view.score}
            components={view.components}
            cofold={view.cofold}
            onOpenComplex={view.structurePath ? () => openComplex(view.structurePath as string) : undefined}
            properties={view.properties}
            predictions={view.predictions}
          />
        </div>

        {why && (
          <details className="text-[11px]">
            <summary className="text-muted hover:text-body cursor-pointer select-none">Why this score</summary>
            <div className="mt-1.5">
              <ScoreWhy values={why} ttpId={merged?.ttpId} ttpVersion={merged?.ttpVersion} />
            </div>
          </details>
        )}
      </div>
    </div>
  );
}
