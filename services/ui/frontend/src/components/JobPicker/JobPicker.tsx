// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useState } from "react";
import { isActive, jobTitle, stateTone, toneColor, type JobSummary } from "../../lib/jobs";
import { focusRun } from "../../lib/workbenchBus";
import { jobOptions, useJobScope, type JobGroup } from "./useJobScope";

/** The job the Jobs tab is about, and "Other jobs" to pick another. */
export function JobPicker({ sessionId }: { sessionId: string | null }) {
  const { runs, own, seen, job, alerts, error } = useJobScope(sessionId);
  return <JobPickerView runs={runs} own={own} seen={seen} job={job} alerts={alerts} error={error} onPick={focusRun} />;
}

interface ViewProps {
  runs: readonly JobSummary[] | null;
  own: readonly JobSummary[];
  /** Jobs the conversation referenced without launching. */
  seen?: readonly JobSummary[];
  job: JobSummary | null;
  /** Distinct alerts per job id. */
  alerts?: ReadonlyMap<string, number>;
  error: string | null;
  onPick: (id: string) => void;
  /** Start with the list shown. */
  initialOpen?: boolean;
}

const HINT: Partial<Record<JobGroup, string>> = { own: "this chat", seen: "seen here" };

export function JobPickerView({ runs, own, seen = [], job, alerts, error, onPick, initialOpen = false }: ViewProps) {
  const [open, setOpen] = useState(initialOpen);
  const options = runs ? jobOptions(runs, own, seen) : [];
  const empty = runs === null ? (error ? `Could not load the jobs: ${error}` : "Loading jobs…") : "No job in this conversation yet.";

  return (
    <div
      className="relative flex min-w-0 shrink-0 items-center gap-2 px-3 pt-2 pb-0.5 text-xs"
      onKeyDown={(event) => {
        if (event.key !== "Escape" || !open) return;
        event.stopPropagation();
        setOpen(false);
      }}
    >
      {job ? (
        <span className="flex min-w-0 items-center gap-2" title={`${jobTitle(job)} · ${job.id} · ${job.state}`}>
          <StateDot job={job} />
          <span className="truncate">
            <span className="text-body">{jobTitle(job)}</span>
            <span className="text-faint">
              {" "}
              · {job.id} · {job.state}
            </span>
          </span>
          <AlertCount count={alerts?.get(job.id) ?? 0} />
        </span>
      ) : (
        <span className="text-faint truncate">{empty}</span>
      )}
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-haspopup="menu"
        aria-expanded={open}
        className="text-muted hover:text-body hover:bg-inset ml-auto shrink-0 cursor-pointer rounded-md px-2 py-0.5 touch:min-h-11"
      >
        Other jobs
      </button>
      {open && (
        <>
          <button type="button" aria-label="Close the job list" tabIndex={-1} className="fixed inset-0 z-30 cursor-default" onClick={() => setOpen(false)} />
          <ul role="menu" aria-label="Jobs" className="bg-raised border-subtle shadow-card absolute top-full right-2 z-40 mt-1 max-h-72 w-72 max-w-[calc(100%-1rem)] overflow-y-auto rounded-lg border py-1">
            {options.length === 0 && <li className="text-faint px-3 py-2">No REINVENT jobs yet.</li>}
            {options.map(({ job: option, group }, index) => {
              const selected = option.id === job?.id;
              const hint = HINT[group];
              const newGroup = index > 0 && options[index - 1].group !== group;
              return (
                <li key={option.id} role="none" className={newGroup ? "border-subtle mt-1 border-t pt-1" : undefined}>
                  <button
                    type="button"
                    role="menuitemradio"
                    aria-checked={selected}
                    onClick={() => {
                      setOpen(false);
                      onPick(option.id);
                    }}
                    className={`hover:bg-sunken flex w-full cursor-pointer items-center gap-2 px-3 py-1.5 text-left ${selected ? "bg-sunken" : ""}`}
                  >
                    <StateDot job={option} />
                    <span className="min-w-0 flex-1 truncate">
                      <span className="text-body">{jobTitle(option)}</span>
                      <span className="text-faint">
                        {" "}
                        · {option.id} · {option.state}
                      </span>
                    </span>
                    <AlertCount count={alerts?.get(option.id) ?? 0} />
                    {hint && (
                      <span className="text-faint flex shrink-0 items-center gap-1 text-[10px]">
                        {group === "own" && <span aria-hidden="true" className="size-1 rounded-full" style={{ background: "var(--color-accent)" }} />}
                        {hint}
                      </span>
                    )}
                  </button>
                </li>
              );
            })}
          </ul>
        </>
      )}
    </div>
  );
}

function StateDot({ job }: { job: JobSummary }) {
  return (
    <span
      aria-hidden="true"
      className={`size-1.5 shrink-0 rounded-full ${isActive(job.state) ? "animate-pulse" : ""}`}
      style={{ background: toneColor(stateTone(job.state)) }}
    />
  );
}

function AlertCount({ count }: { count: number }) {
  if (count <= 0) return null;
  const label = `${count} alert${count === 1 ? "" : "s"}`;
  return (
    <span className="text-muted flex shrink-0 items-center gap-0.5 text-[10px] tabular-nums" title={label} aria-label={label}>
      <span aria-hidden="true" className="size-1 rounded-full" style={{ background: toneColor("warn") }} />
      {count}
    </span>
  );
}
