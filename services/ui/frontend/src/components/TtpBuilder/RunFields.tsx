// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { callTool } from "../../lib/api";
import { moleculesFromFile } from "../../lib/dropFiles";
import { readSeedFile } from "../../lib/sharedFiles";
import { useShortlist, useShortlistSession } from "../../lib/shortlist";
import { asTtpList, type TtpSummary } from "../../lib/ttp";
import {
  MAX_BATCH,
  MAX_STEPS,
  addSeeds,
  loadRunSettings,
  mol2molChoices,
  parseMol2MolModels,
  rememberRun,
  removeSeed,
  runCost,
  runProblems,
  runRequest,
  seedChoices,
  seedList,
  withOfferedModel,
  type Prior,
  type PriorModel,
  type RunSettings,
  type SeedMolecule,
} from "../../lib/ttpRun";

const int = (n: number) => n.toLocaleString("en");

const PRIORS: [Prior, string][] = [
  ["reinvent", "From scratch"],
  ["mol2mol", "Analogues of my molecules"],
];

interface FieldsProps {
  run: RunSettings;
  /** The Mol2Mol priors on offer, from `mol2molChoices`. */
  models: PriorModel[];
  cofold?: boolean;
  /** Mol2Mol only: the seeds are the point. */
  mol2molOnly?: boolean;
  /** Molecules of the conversation to offer as seeds, besides its shortlist. */
  molecules?: SeedMolecule[];
  onChange: (patch: Partial<RunSettings>) => void;
}

/** Steps, batch size, prior and seeds of a REINVENT run. */
export function RunFields({ run, models, cofold = false, mol2molOnly = false, molecules, onChange }: FieldsProps) {
  const name = useId();
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
      <div role="radiogroup" aria-label="How to generate" className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px]">
        {PRIORS.map(([value, label]) => (
          <label key={value} className={`flex items-center gap-1.5 ${mol2molOnly && value !== "mol2mol" ? "opacity-40" : "cursor-pointer"}`}>
            <input
              type="radio"
              name={name}
              value={value}
              checked={run.prior === value}
              disabled={mol2molOnly && value !== "mol2mol"}
              onChange={() => onChange({ prior: value })}
              className="accent-[var(--color-accent)]"
            />
            {label}
          </label>
        ))}
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
      {run.prior === "mol2mol" && <Seeds seeds={run.seeds} molecules={molecules} onChange={(seeds) => onChange({ seeds })} />}
    </>
  );
}

type Reading = { state: "idle" } | { state: "busy" } | { state: "done"; text: string } | { state: "error"; text: string };

/** Seed SMILES from the conversation's molecules, a file, or typed. */
function Seeds({ seeds, molecules, onChange }: { seeds: string; molecules?: SeedMolecule[]; onChange: (seeds: string) => void }) {
  const shortlist = useShortlist(useShortlistSession());
  const chat = seedChoices(molecules, shortlist);
  const chosen = new Set(seedList(seeds));
  const [path, setPath] = useState("");
  const [reading, setReading] = useState<Reading>({ state: "idle" });
  const picker = useRef<HTMLInputElement>(null);

  const took = (found: { smiles: string[]; failed: number }, from: string) => {
    onChange(addSeeds(seeds, found.smiles));
    const skipped = found.failed ? `, ${found.failed} unreadable` : "";
    setReading({ state: found.smiles.length ? "done" : "error", text: found.smiles.length ? `${found.smiles.length} from ${from}${skipped}` : `No SMILES in ${from}${skipped}` });
  };

  const read = async (task: () => Promise<{ smiles: string[]; failed: number }>, from: string) => {
    setReading({ state: "busy" });
    try {
      took(await task(), from);
    } catch (err) {
      setReading({ state: "error", text: err instanceof Error ? err.message : String(err) });
    }
  };

  const fromShared = () => {
    const file = path.trim();
    if (file) void read(() => readSeedFile(file), file.split("/").pop() ?? file);
  };

  return (
    <div className="space-y-1.5">
      {chat.length > 0 && (
        <div className="flex flex-wrap items-center gap-1">
          <span className="text-muted mr-1 text-[11px]">In this chat</span>
          {chat.map((m) => {
            const on = chosen.has(m.smiles);
            return (
              <button
                key={m.smiles}
                type="button"
                aria-pressed={on}
                title={m.smiles}
                onClick={() => onChange(on ? removeSeed(seeds, m.smiles) : addSeeds(seeds, [m.smiles]))}
                className={`max-w-40 cursor-pointer truncate rounded-full border px-2 py-px text-[10px] ${on ? "border-strong bg-sunken text-body font-medium" : "border-subtle text-muted hover:text-body"} ${m.label ? "" : "font-mono"}`}
              >
                {m.label ?? m.smiles}
              </button>
            );
          })}
          {chat.some((m) => !chosen.has(m.smiles)) && chat.length > 1 && (
            <button type="button" onClick={() => onChange(addSeeds(seeds, chat.map((m) => m.smiles)))} className="text-faint hover:text-body cursor-pointer px-1 text-[10px]">
              all
            </button>
          )}
        </div>
      )}
      <div className="flex flex-wrap items-center gap-1.5">
        <input
          value={path}
          onChange={(event) => setPath(event.target.value)}
          onKeyDown={(event) => {
            if (event.key !== "Enter") return;
            event.preventDefault();
            fromShared();
          }}
          placeholder="Shared file: .smi, .csv or .sdf"
          aria-label="Seed file on the shared volume"
          spellCheck={false}
          className={`${INPUT} min-w-40 flex-1 font-mono text-[10px]`}
        />
        <button type="button" onClick={fromShared} disabled={!path.trim() || reading.state === "busy"} className={ACTION}>
          Read
        </button>
        <button type="button" onClick={() => picker.current?.click()} disabled={reading.state === "busy"} className={ACTION}>
          From computer
        </button>
        <input
          ref={picker}
          type="file"
          accept=".smi,.smiles,.csv,.sdf,.mol"
          hidden
          onChange={(event) => {
            const file = event.target.files?.[0];
            event.target.value = "";
            if (file) void read(() => moleculesFromFile(file), file.name);
          }}
        />
      </div>
      <textarea
        value={seeds}
        onChange={(event) => onChange(event.target.value)}
        rows={2}
        spellCheck={false}
        aria-label="Seed SMILES"
        placeholder="Seed SMILES, one per line"
        className={`${INPUT} w-full resize-y font-mono text-[10px]`}
      />
      {(reading.state !== "idle" || chosen.size > 0) && (
        <p className="text-faint text-[10px]" aria-live="polite">
          {chosen.size > 0 && `${chosen.size} seed${chosen.size === 1 ? "" : "s"}`}
          {reading.state === "busy" && " · reading…"}
          {reading.state === "done" && ` · added ${reading.text}`}
          {reading.state === "error" && <span style={{ color: "var(--color-accent)" }}>{chosen.size > 0 ? " · " : ""}{reading.text}</span>}
        </p>
      )}
    </div>
  );
}

interface CardProps {
  /** The TTP to run; without it the card lists the stored ones to pick from. */
  ttp?: { id: string; name: string };
  /** The TTP to pick first when listing, e.g. the conversation's. */
  preferTtpId?: string | null;
  /** Start Mol2Mol from these SMILES, one per line. */
  seeds?: string;
  /** Molecules of the conversation to offer as seeds, besides its shortlist. */
  molecules?: SeedMolecule[];
  cofold?: boolean;
  /** Called with the run request for the composer. */
  onAsk: (request: string) => void;
  onClose: () => void;
}

/** A small card to put a REINVENT run request in the composer; the user sends it. */
export function RunCard({ ttp, preferTtpId, seeds, molecules, cofold, onAsk, onClose }: CardProps) {
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
  const shown = attempted ? (problems[0] ?? (target ? null : "Pick a TTP to run.")) : null;
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
      <RunFields run={run} models={choices} cofold={cofold} mol2molOnly={seeds !== undefined} molecules={molecules} onChange={update} />
      <div className="flex flex-wrap items-center gap-2">
        <p className="min-w-0 flex-1 text-[11px]" style={{ color: "var(--color-accent)" }}>
          {shown}
        </p>
        <button type="button" onClick={put} title="Put the run request in the message box" className={ACTION} style={{ borderColor: "var(--color-accent)", color: "var(--color-accent)" }}>
          Put in chat
        </button>
      </div>
    </div>
  );
}

const INPUT = "bg-sunken border-subtle focus:border-strong rounded-md border px-2 py-1 text-xs outline-none";
const ACTION = "border-subtle text-body hover:bg-inset cursor-pointer rounded-md border px-2 py-0.5 text-[11px] font-medium disabled:cursor-default disabled:opacity-50 touch:min-h-11";

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
