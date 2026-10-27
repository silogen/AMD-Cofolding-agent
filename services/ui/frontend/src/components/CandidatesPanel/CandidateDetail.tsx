// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useMemo } from "react";
import type { Theme } from "../../hooks/useTheme";
import { type Candidate, candidateNumbers, cofoldPrompt, METRIC_LABELS, type MolProps } from "../../lib/candidates";
import type { Ttp } from "../../lib/ttp";
import { ScoreWhy } from "../RunResults/ScoreWhy";
import { affinityConcentration, affinityText, pIC50 } from "../../lib/gate";
import type { StructureMetrics } from "../../types/api";
import { MolDepiction } from "../MolDepiction/MolDepiction";
import { GateBadge } from "./GateBadge";
import { ScoreBar } from "./ScoreBar";

interface Props {
  candidate: Candidate;
  /** RDKit descriptors, once computed. */
  props?: MolProps;
  theme: Theme;
  onClose: () => void;
  onOpenMolecule: (smiles: string) => void;
  onOpenStructure: (path: string) => void;
  onAsk: (prompt: string) => void;
  /** The conversation's TTP, to explain the score with. */
  ttp?: Ttp | null;
  ttpId?: string | null;
}

const button =
  "border-subtle hover:bg-sunken inline-flex cursor-pointer items-center gap-1 rounded-md border px-2 py-1 text-[11px] font-medium disabled:cursor-not-allowed disabled:opacity-50";

/** Everything known about one candidate: drawing, scores, gate and where it came from. */
export function CandidateDetail({ candidate: c, props: p, theme, onClose, onOpenMolecule, onOpenStructure, onAsk, ttp, ttpId }: Props) {
  const values = useMemo(() => candidateNumbers(c), [c]);
  const descriptorRows: [string, string][] = p
    ? [
        ["MW", `${p.mw.toFixed(1)} Da`],
        ["cLogP", p.clogp.toFixed(2)],
        ["TPSA", `${p.tpsa.toFixed(1)} Å²`],
        ["HBD / HBA", `${p.hbd} / ${p.hba}`],
        ["Rotatable bonds", String(p.rotb)],
        ["Lipinski", `${p.ro5} violation${p.ro5 === 1 ? "" : "s"}`],
      ]
    : [];
  const components = Object.entries(c.components);
  const metrics = (Object.keys(METRIC_LABELS) as (keyof StructureMetrics)[])
    .filter((key) => key !== "affinity" && typeof c.metrics?.[key] === "number")
    .map((key) => [METRIC_LABELS[key] as string, c.metrics?.[key] as number] as const);
  const affinity = c.metrics ? affinityText(c.metrics) : null;

  return (
    <aside className="flex min-h-0 flex-col gap-3 overflow-y-auto p-3 text-xs" aria-label="Candidate details">
      <header className="flex items-start gap-2">
        <div className="min-w-0 flex-1">
          <h3 className="truncate text-sm font-semibold" title={c.name ?? c.smiles}>
            {c.name ?? "Unnamed candidate"}
          </h3>
          <p className="text-faint truncate text-[11px]">{c.origins.join(" · ")}</p>
        </div>
        <GateBadge gate={c.gate} />
        <button type="button" onClick={onClose} className="text-muted hover:text-body cursor-pointer px-1" aria-label="Close details">
          ×
        </button>
      </header>

      <div className="grid gap-4 sm:grid-cols-[minmax(0,14rem)_minmax(0,1fr)] lg:grid-cols-[minmax(0,14rem)_minmax(0,12rem)_minmax(0,1fr)]">
      <div className="flex min-w-0 flex-col gap-2">
      <button
        type="button"
        onClick={() => onOpenMolecule(c.smiles)}
        className="bg-sunken border-subtle cursor-pointer rounded-lg border p-1"
        title="Open the 2D drawing"
      >
        <MolDepiction smiles={c.smiles} theme={theme} width={260} height={170} label={c.name} className="h-32 w-full" />
      </button>
      <p className="text-muted font-mono text-[10px] break-all">{c.smiles}</p>

      <div className="flex flex-wrap gap-1.5">
        <button type="button" className={button} onClick={() => onOpenMolecule(c.smiles)}>
          2D
        </button>
        <button
          type="button"
          className={button}
          disabled={!c.structurePath}
          title={c.structurePath ? "Open the predicted complex" : "Not cofolded yet"}
          onClick={() => c.structurePath && onOpenStructure(c.structurePath)}
        >
          3D
        </button>
        <button type="button" className={button} onClick={() => onAsk(cofoldPrompt([c]))}>
          Ask to cofold
        </button>
        <button type="button" className={button} onClick={() => void navigator.clipboard?.writeText(c.smiles)}>
          Copy SMILES
        </button>
      </div>
      </div>

      <section className="flex flex-col gap-1.5">
        <h4 className="text-muted text-[11px] font-semibold tracking-wide uppercase">Descriptors</h4>
        {descriptorRows.length === 0 && <p className="text-faint">Computing with RDKit…</p>}
        <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
          {descriptorRows.map(([label, value]) => (
            <div key={label} className="contents">
              <dt className="text-muted">{label}</dt>
              <dd
                className="text-right font-mono tabular-nums"
                style={label === "Lipinski" && p && p.ro5 > 1 ? { color: "var(--color-warn)" } : undefined}
              >
                {value}
              </dd>
            </div>
          ))}
        </dl>
      </section>

      <div className="flex min-w-0 flex-col gap-3 sm:col-span-2 lg:col-span-1">
      {(ttp || ttpId) && <ScoreWhy title="Why this score" values={values} ttp={ttp} ttpId={ttpId} />}

      <section className="flex flex-col gap-1.5">
        <h4 className="text-muted text-[11px] font-semibold tracking-wide uppercase">Score components</h4>
        {c.score === undefined && components.length === 0 && <p className="text-faint">No scores reported.</p>}
        {c.score !== undefined && (
          <div className="flex items-center justify-between gap-2 font-semibold">
            <span>Total score</span>
            <ScoreBar value={c.score} className="w-24" />
          </div>
        )}
        {components.map(([key, value]) => (
          <div key={key} className="flex items-center justify-between gap-2">
            <span className="text-muted truncate" title={key}>
              {key}
            </span>
            <ScoreBar value={value} className="w-24" />
          </div>
        ))}
      </section>

      {c.metrics && (
        <section className="flex flex-col gap-1.5">
          <h4 className="text-muted text-[11px] font-semibold tracking-wide uppercase">Predicted complex</h4>
          {metrics.map(([label, value]) => (
            <div key={label} className="flex items-center justify-between gap-2">
              <span className="text-muted">{label}</span>
              <ScoreBar value={value} className="w-24" />
            </div>
          ))}
          {c.gate.checks.map((check) => (
            <div key={check.label} className="flex items-center justify-between gap-2">
              <span className="text-muted">
                {check.label} <span className="text-faint">({check.rule})</span>
              </span>
              <span
                className="font-mono"
                style={{ color: check.ok === null ? "var(--text-faint)" : check.ok ? "var(--color-ok)" : "var(--color-warn)" }}
              >
                {check.value} {check.ok === null ? "" : check.ok ? "✓" : "!"}
              </span>
            </div>
          ))}
          {affinity && c.metrics.affinity !== undefined && (
            <div className="bg-sunken rounded-md px-2 py-1.5">
              <p>
                Boltz affinity <span className="font-mono font-semibold">{affinity}</span>{" "}
                <span className="text-muted">
                  (pIC50 {pIC50(c.metrics.affinity).toFixed(2)}, {affinityConcentration(c.metrics.affinity)})
                </span>
              </p>
              {c.gate.status === "fail" && (
                <p className="mt-0.5 text-[10px]" style={{ color: "var(--color-warn)" }} title={c.gate.reasons.join("; ")}>
                  Pose below the quality gate; read the affinity with caution.
                </p>
              )}
            </div>
          )}
        </section>
      )}

      {c.structurePath && (
        <section className="flex flex-col gap-1">
          <h4 className="text-muted text-[11px] font-semibold tracking-wide uppercase">Provenance</h4>
          <span className="text-faint font-mono text-[10px] break-all">{c.structurePath}</span>
        </section>
      )}
      </div>
      </div>
    </aside>
  );
}
