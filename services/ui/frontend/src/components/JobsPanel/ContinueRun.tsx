// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useState } from "react";
import { callTool, jobFileUrl } from "../../lib/api";
import {
  continueCost,
  continueSteps,
  isMol2Mol,
  jobTitle,
  parseRunConfig,
  resultsFile,
  resumeArguments,
  resumePrompt,
  startedJobId,
  toolError,
  type JobStatus,
  type JobSummary,
  type RunOverrides,
} from "../../lib/jobs";
import { parseRunResults, runSeeds } from "../../lib/runResults";
import { loadTtp } from "../../lib/scoreWhy";
import type { Ttp } from "../../lib/ttp";
import { focusRun, notifyRunsChanged } from "../../lib/workbenchBus";

interface Props {
  job: JobSummary | JobStatus;
  /** The run's Mol2Mol seeds when the caller already has its results. */
  seeds?: string[];
  /** A run that starting this one would stop. */
  running?: JobSummary | null;
  onClose: () => void;
  onAsk?: (prompt: string) => void;
}

interface Prepared {
  config: RunOverrides;
  ttp: Ttp | null;
  seeds: string[];
  /** Why it cannot continue, e.g. a Mol2Mol run whose seeds are unknown. */
  blocker: string | null;
}

const MAX_STEPS = 5000;

async function readText(url: string): Promise<string> {
  try {
    const response = await fetch(url);
    return response.ok ? await response.text() : "";
  } catch {
    return "";
  }
}

type Source = Pick<JobSummary, "id" | "ttpId" | "ttpVersion" | "model" | "csvPath">;

async function prepare(job: Source, known: string[]): Promise<Prepared> {
  const [toml, ttp] = await Promise.all([readText(jobFileUrl(job.id, "config.toml")), job.ttpId ? loadTtp(job.ttpId, job.ttpVersion) : null]);
  const config = parseRunConfig(toml);
  let seeds = known;
  if (isMol2Mol(job, config) && !seeds.length) seeds = runSeeds(parseRunResults(await readText(jobFileUrl(job.id, resultsFile(job)))));
  const blocker = isMol2Mol(job, config) && !seeds.length ? "The Mol2Mol seeds of this run could not be read." : null;
  return { config, ttp, seeds, blocker };
}

/**
 * Continue a stopped or finished run from its checkpoint: the same TTP
 * version, model and learning settings, for as many steps as the user picks.
 * Asks first with what it will cost; nothing runs until "Continue run".
 */
export function ContinueRun({ job, seeds, running, onClose, onAsk }: Props) {
  const [steps, setSteps] = useState(() => continueSteps(job));
  const [prepared, setPrepared] = useState<Prepared | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { id, ttpId, ttpVersion, model, csvPath } = job;
  const seedList = seeds?.join("\n") ?? "";
  useEffect(() => {
    let cancelled = false;
    prepare({ id, ttpId, ttpVersion, model, csvPath }, seedList ? seedList.split("\n") : []).then((p) => {
      if (!cancelled) setPrepared(p);
    });
    return () => {
      cancelled = true;
    };
  }, [id, ttpId, ttpVersion, model, csvPath, seedList]);

  const valid = Number.isInteger(steps) && steps >= 1 && steps <= MAX_STEPS;

  async function start() {
    if (!prepared) return;
    setBusy(true);
    setError(null);
    try {
      const { result } = await callTool("run_optimization", resumeArguments(job, { steps, config: prepared.config, ttp: prepared.ttp ?? undefined, seeds: prepared.seeds }));
      const failure = toolError(result);
      const started = startedJobId(result);
      if (failure || !started) {
        setError(failure ?? "The run did not start.");
        return;
      }
      focusRun(started);
      notifyRunsChanged();
      onClose();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  }

  const details = [valid ? continueCost(job, steps, prepared?.config.batchSize ?? null) : "", running ? `stops ${jobTitle(running)}` : ""].filter(Boolean);

  return (
    <div className="bg-sunken border-subtle mt-2 rounded-lg border px-3 py-2 text-[11px]" role="group" aria-label="Continue the run">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1.5">
        <label className="flex items-center gap-1.5">
          <span>Continue {jobTitle(job)} for</span>
          <input
            type="number"
            min={1}
            max={MAX_STEPS}
            step={10}
            value={Number.isFinite(steps) ? steps : ""}
            onChange={(event) => setSteps(Math.round(Number(event.target.value)))}
            aria-label="Extra steps"
            className="bg-raised border-subtle w-16 rounded border px-1.5 py-0.5 font-mono tabular-nums"
          />
          <span>steps</span>
        </label>
        <span className="text-faint min-w-0 flex-1">{prepared ? (prepared.blocker ?? details.join(" · ")) : "Reading the run's settings…"}</span>
        <button type="button" onClick={onClose} className="text-muted hover:text-body hover:bg-inset cursor-pointer rounded px-2 py-0.5 font-medium">
          Cancel
        </button>
        <button
          type="button"
          disabled={!prepared || !!prepared.blocker || !valid || busy}
          onClick={() => void start()}
          className="bg-accent hover:bg-accent-strong cursor-pointer rounded px-2 py-0.5 font-medium text-white disabled:cursor-not-allowed disabled:opacity-50"
        >
          {busy ? "Starting…" : "Continue run"}
        </button>
      </div>
      {(error || prepared?.blocker) && (
        <p className="mt-1" style={{ color: error ? "var(--color-accent)" : undefined }}>
          {error}
          {onAsk && (
            <button type="button" onClick={() => onAsk(resumePrompt(job))} className="ml-2 cursor-pointer font-semibold underline underline-offset-2">
              Ask the agent instead
            </button>
          )}
        </p>
      )}
    </div>
  );
}
