// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useState, type ReactNode } from "react";
import { callTool } from "../../lib/api";
import { asTtpList, type TtpSummary } from "../../lib/ttp";
import {
  MAX_BATCH,
  MAX_STEPS,
  loadRunSettings,
  mol2molChoices,
  parseMol2MolModels,
  rememberRun,
  runCost,
  runProblems,
  runRequest,
  withOfferedModel,
  type PriorModel,
  type RunSettings,
} from "../../lib/ttpRun";

const int = (n: number) => n.toLocaleString("en");

interface FieldsProps {
  run: RunSettings;
  /** The Mol2Mol priors on offer, from `mol2molChoices`. */
  models: PriorModel[];
  cofold?: boolean;
  /** Mol2Mol only: the seeds are the point. */
  mol2molOnly?: boolean;
  onChange: (patch: Partial<RunSettings>) => void;
}

/** Steps, batch size, prior and seeds of a REINVENT run. */
export function RunFields({ run, models, cofold = false, mol2molOnly = false, onChange }: FieldsProps) {
  const effective = withOfferedModel(run, models);
  const cost = runCost(run, cofold);
  return (
    <>
      <div className="flex flex-wrap items-end gap-3">
        <Labelled label="Steps">
          <input
            type="number"
            min={1}
            max={MAX_STEPS}
            step={50}
            value={run.maxSteps}
            onChange={(event) => onChange({ maxSteps: Math.round(Number(event.target.value)) })}
            className={`${INPUT} w-20 font-mono tabular-nums`}
          />
        </Labelled>
        <Labelled label="Batch size">
          <input
            type="number"
            min={1}
            max={MAX_BATCH}
            step={8}
            value={run.batchSize}
            onChange={(event) => onChange({ batchSize: Math.round(Number(event.target.value)) })}
            className={`${INPUT} w-20 font-mono tabular-nums`}
          />
        </Labelled>
        <p className="text-faint pb-1 text-[11px]">
          = {int(cost.molecules)} molecules{cofold ? `, each cofolded` : ""}
        </p>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-muted text-[11px]">Prior</span>
        <div role="radiogroup" aria-label="Prior" className="bg-sunken border-subtle inline-flex rounded-md border p-0.5">
          {(
            [
              ["reinvent", "Reinvent · de novo"],
              ["mol2mol", "Mol2Mol · from seeds"],
            ] as const
          ).map(([value, label]) => (
            <button
              key={value}
              type="button"
              role="radio"
              aria-checked={run.prior === value}
              disabled={mol2molOnly && value !== "mol2mol"}
              onClick={() => onChange({ prior: value })}
              className={`cursor-pointer rounded px-2 py-0.5 text-[11px] disabled:cursor-default disabled:opacity-40 ${run.prior === value ? "bg-raised text-body shadow-card font-medium" : "text-muted hover:text-body"}`}
            >
              {label}
            </button>
          ))}
        </div>
        {run.prior === "mol2mol" && models.length > 1 && (
          <select value={effective.model} onChange={(event) => onChange({ model: event.target.value })} aria-label="Mol2Mol prior" className={INPUT}>
            {models.map((m) => (
              <option key={m.key} value={m.key}>
                {m.label}
              </option>
            ))}
          </select>
        )}
      </div>
      {run.prior === "mol2mol" && (
        <textarea
          value={run.seeds}
          onChange={(event) => onChange({ seeds: event.target.value })}
          rows={2}
          spellCheck={false}
          aria-label="Seed SMILES"
          placeholder="Seed SMILES, one per line: the molecules Mol2Mol starts from"
          className={`${INPUT} w-full resize-y font-mono text-[10px]`}
        />
      )}
    </>
  );
}

interface CardProps {
  /** The TTP to run; without it the card lists the stored ones to pick from. */
  ttp?: { id: string; name: string };
  /** The TTP to pick first when listing, e.g. the conversation's. */
  preferTtpId?: string | null;
  /** Start Mol2Mol from these SMILES, one per line. */
  seeds?: string;
  cofold?: boolean;
  /** Called with the run request for the composer. */
  onAsk: (request: string) => void;
  onClose: () => void;
}

/** A small card to put a REINVENT run request in the composer; the user sends it. */
export function RunCard({ ttp, preferTtpId, seeds, cofold, onAsk, onClose }: CardProps) {
  const [run, setRun] = useState<RunSettings>(() => ({ ...loadRunSettings(), ...(seeds !== undefined ? { prior: "mol2mol" as const, seeds } : {}) }));
  const [models, setModels] = useState<PriorModel[]>([]);
  const [ttps, setTtps] = useState<TtpSummary[] | null>(null);
  const [pick, setPick] = useState<string | null>(null);
  const [attempted, setAttempted] = useState(false);

  useEffect(() => {
    let cancelled = false;
    callTool("list_available_models")
      .then((response) => !cancelled && setModels(parseMol2MolModels(response.result)))
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  const listing = !ttp;
  useEffect(() => {
    if (!listing) return;
    let cancelled = false;
    callTool("list_ttps")
      .then((response) => !cancelled && setTtps(asTtpList(response.result)))
      .catch(() => !cancelled && setTtps([]));
    return () => {
      cancelled = true;
    };
  }, [listing]);

  const choices = mol2molChoices(models);
  const effective = withOfferedModel(run, choices);
  const chosen = pick ?? preferTtpId ?? null;
  const target = ttp ?? ttps?.find((t) => t.id === chosen) ?? ttps?.[0] ?? null;
  const problems = runProblems(effective);
  const update = (patch: Partial<RunSettings>) => {
    setRun({ ...run, ...patch });
    rememberRun(patch);
  };

  const put = () => {
    setAttempted(true);
    if (!target || problems.length) return;
    onAsk(runRequest(target, effective));
    onClose();
  };

  return (
    <div className="border-subtle bg-sunken space-y-2.5 rounded-lg border px-3 py-2.5">
      <div className="flex items-baseline gap-2">
        <p className="min-w-0 flex-1 text-[11px] font-semibold">Run REINVENT</p>
        <button type="button" onClick={onClose} aria-label="Close" className="text-faint hover:text-body shrink-0 cursor-pointer px-1 leading-none">
          ×
        </button>
      </div>
      {listing && (
        <Labelled label="TTP">
          <select
            value={target?.id ?? ""}
            onChange={(event) => setPick(event.target.value)}
            disabled={!ttps?.length}
            aria-label="TTP to run"
            className={`${INPUT} w-full`}
          >
            {!target && <option value="">{ttps === null ? "Loading…" : "No TTP stored yet"}</option>}
            {ttps?.map((t) => (
              <option key={t.id} value={t.id}>
                {t.name} · v{t.version}
              </option>
            ))}
          </select>
        </Labelled>
      )}
      <RunFields run={run} models={choices} cofold={cofold} mol2molOnly={seeds !== undefined} onChange={update} />
      <div className="flex flex-wrap items-center gap-2">
        <p className="text-faint min-w-0 flex-1 text-[11px]">
          {attempted && (problems[0] ?? (target ? null : "Pick a TTP to run.")) ? (
            <span style={{ color: "var(--color-accent)" }}>{problems[0] ?? "Pick a TTP to run."}</span>
          ) : (
            "The chat gets the request; you send it."
          )}
        </p>
        <button type="button" onClick={put} title="Put the run request in the message box" className={ACTION} style={{ borderColor: "var(--color-accent)", color: "var(--color-accent)" }}>
          Put in chat
        </button>
      </div>
    </div>
  );
}

const INPUT = "bg-sunken border-subtle focus:border-strong rounded-md border px-2 py-1 text-xs outline-none";
const ACTION = "border-subtle text-body hover:bg-inset cursor-pointer rounded-md border px-2 py-0.5 text-[11px] font-medium touch:min-h-11";

function Labelled({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="block space-y-0.5">
      <span className="text-muted flex gap-2 text-[11px]">
        {label}
        {hint && <span className="text-faint font-mono">{hint}</span>}
      </span>
      {children}
    </label>
  );
}
