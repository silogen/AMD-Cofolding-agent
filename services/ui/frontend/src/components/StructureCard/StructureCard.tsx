// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { StructurePanel } from "../../types/api";
import type { Theme } from "../../hooks/useTheme";
import { affinityText, gate } from "../../lib/gate";
import { runOf, structureTitle } from "../../lib/artifacts";
import { STRUCTURE_KINDS, structureKind } from "../../lib/poses";
import { sendToViewer } from "../../lib/viewerBus";
import { MolDepiction } from "../MolDepiction/MolDepiction";

interface Props {
  panel: StructurePanel;
  theme: Theme;
  isOpen: boolean;
}

/**
 * A protein-ligand complex, inline in the conversation.
 *
 * No 3D here: a WebGL viewer per card made long conversations slow to load
 * and some cards never finished. The card carries the verdict (gate,
 * confidence, affinity) and the ligand in 2D; the buttons drive the one
 * viewer in the workbench.
 */
export function StructureCard({ panel, theme, isOpen }: Props) {
  const verdict = gate(panel.metrics);
  const m = panel.metrics ?? {};
  const affinity = affinityText(m);
  const facts: [string, string, boolean | null][] = verdict.checks
    .filter((check) => check.value !== "—")
    .map((check) => [check.label, check.value, check.ok]);
  if (affinity) facts.push(["Affinity", affinity, null]);
  if (m.binderProbability !== undefined) facts.push(["Binder p", m.binderProbability.toFixed(2), null]);

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
          <span className="text-faint shrink-0 text-[11px]">{kind ? STRUCTURE_KINDS[kind].label : `${panel.format.toUpperCase()} file`}</span>
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
        {verdict.status !== "unknown" && (
          <span
            className="ml-auto shrink-0 text-[10px]"
            style={{ color: verdict.status === "pass" ? "var(--color-ok)" : "var(--color-warn)" }}
            title={verdict.reasons.join(" · ") || "Confidence > 0.6, ligand ipTM > 0.5, PoseBusters all pass"}
          >
            {verdict.status === "pass" ? "✓ quality gate" : "below quality gate"}
          </span>
        )}
        <span className={`text-faint shrink-0 text-[10px] uppercase ${verdict.status === "unknown" ? "ml-auto" : ""}`}>{panel.format}</span>
      </figcaption>

      <button
        type="button"
        onClick={() => show()}
        className="hover:bg-sunken flex w-full cursor-pointer gap-3 px-3 py-2 text-left"
        title="Open in the 3D viewer"
      >
        {panel.smiles ? (
          <MolDepiction smiles={panel.smiles} theme={theme} width={180} height={120} label={panel.label ?? undefined} className="h-20 w-28 shrink-0" />
        ) : (
          <span className="bg-sunken text-faint grid h-20 w-28 shrink-0 place-items-center rounded-lg text-[10px]">{panel.format.toUpperCase()} file</span>
        )}
        {facts.length > 0 && (
          <div className="min-w-0 flex-1 self-center">
            <dl className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[11px]">
              {facts.map(([label, value, ok]) => (
                <div key={label} className="min-w-0">
                  <dt className="text-faint truncate">{label}</dt>
                  <dd className="truncate font-mono tabular-nums" style={{ color: ok === false ? "var(--color-warn)" : undefined }}>
                    {value}
                  </dd>
                </div>
              ))}
            </dl>
            {verdict.status === "fail" && affinity && (
              <p className="text-faint mt-1 text-[10px]">Pose below the quality gate; read the affinity with caution.</p>
            )}
          </div>
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
