// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { StructurePanel } from "../../types/api";
import type { Theme } from "../../hooks/useTheme";
import { gate } from "../../lib/gate";
import { runOf, structureTitle } from "../../lib/artifacts";
import { STRUCTURE_KINDS, structureKind } from "../../lib/poses";
import { sendToViewer } from "../../lib/viewerBus";
import { MolDepiction } from "../MolDepiction/MolDepiction";
import { structureFacts, verdictLabel } from "./structureFacts";

interface Props {
  panel: StructurePanel;
  theme: Theme;
  isOpen: boolean;
}

/**
 * A protein-ligand complex, inline in the conversation.
 *
 * No 3D here: a WebGL viewer per card made long conversations slow to load
 * and some cards never finished. The card carries the verdict (valid or
 * not, affinity, confidence, PoseBusters) and the ligand in 2D; the buttons drive the one
 * viewer in the workbench.
 */
export function StructureCard({ panel, theme, isOpen }: Props) {
  const verdict = gate(panel.metrics);
  const badge = verdictLabel(verdict.status);
  const facts = structureFacts(panel.metrics);

  const show = () => sendToViewer({ path: panel.path, source: "chat" });

  if (facts.length === 0 && !panel.smiles) {
    const kind = structureKind(panel.path, false, panel.tool);
    return (
      <figure className="bg-raised border-subtle m-0 flex items-center gap-2 rounded-xl border py-1.5 pr-1.5 pl-3">
        <button
          type="button"
          onClick={show}
          title={isOpen ? "In the 3D viewer" : "Open in the 3D viewer"}
          className="flex min-w-0 flex-1 cursor-pointer items-baseline gap-2 text-left"
        >
          <span className="min-w-0 truncate text-xs font-semibold">{structureTitle(panel)}</span>
          {kind && <span className="text-faint shrink-0 text-[11px]">{STRUCTURE_KINDS[kind].label}</span>}
        </button>
        <DownloadLink panel={panel} />
      </figure>
    );
  }

  return (
    <figure className="bg-raised border-subtle m-0 overflow-hidden rounded-xl border">
      <figcaption className="border-subtle flex items-center gap-2 border-b px-3 py-2">
        <span aria-hidden="true" className="size-2 shrink-0 rounded-full" style={{ background: "var(--color-structure)" }} />
        <span className="min-w-0 truncate text-xs font-semibold" title={panel.path}>
          {structureTitle(panel)}
        </span>
        {runOf(panel.path) && <span className="text-faint shrink-0 font-mono text-[10px]">{runOf(panel.path)}</span>}
        {badge && (
          <span
            className="ml-auto shrink-0 text-[10px]"
            style={{ color: verdict.status === "pass" ? "var(--color-ok)" : "var(--color-warn)" }}
            title={verdict.reasons.join(" · ") || "Confidence > 0.6, ligand ipTM > 0.5, PoseBusters all pass"}
          >
            {badge}
          </span>
        )}
      </figcaption>

      <button
        type="button"
        onClick={() => show()}
        className="hover:bg-sunken flex w-full cursor-pointer items-center gap-3 px-3 py-2 text-left"
        title="Open in the 3D viewer"
      >
        {panel.smiles && (
          <MolDepiction smiles={panel.smiles} theme={theme} width={150} height={120} label={panel.label ?? undefined} className="h-20 w-24 shrink-0" />
        )}
        {facts.length > 0 && (
          <dl className="min-w-0 flex-1 space-y-0.5 text-[11px]">
            {facts.map((fact) => (
              <div key={fact.label} className="flex items-baseline justify-between gap-2" title={fact.title}>
                <dt className="text-faint truncate">{fact.label}</dt>
                <dd className="shrink-0 font-mono tabular-nums" style={{ color: fact.tone === "warn" ? "var(--color-warn)" : undefined }}>
                  {fact.value}
                  {fact.unit && <span className="text-faint ml-0.5 font-sans">{fact.unit}</span>}
                </dd>
              </div>
            ))}
          </dl>
        )}
      </button>

      <div className="border-subtle flex items-center gap-1 border-t py-1 pr-1.5 pl-3">
        <button
          type="button"
          onClick={show}
          className="border-subtle text-body hover:bg-inset cursor-pointer rounded-md border px-2 py-1 text-[11px] font-medium whitespace-nowrap touch:min-h-11 touch:px-3"
        >
          {isOpen ? "Showing in viewer" : "Open viewer"}
        </button>
        <span className="ml-auto" />
        <DownloadLink panel={panel} />
      </div>
    </figure>
  );
}

function DownloadLink({ panel }: { panel: StructurePanel }) {
  return (
    <a
      href={panel.url}
      download={panel.name}
      title={`Download ${panel.name}`}
      aria-label={`Download ${panel.name}`}
      className="text-muted hover:text-body hover:bg-inset inline-flex size-7 shrink-0 items-center justify-center rounded-md touch:size-11"
    >
      <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M8 2.5v8M4.5 7 8 10.5 11.5 7M3 13.5h10" />
      </svg>
    </a>
  );
}
