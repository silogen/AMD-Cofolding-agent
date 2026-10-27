// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useState } from "react";
import type { Artifacts } from "../../lib/artifacts";
import { callTool, jobFileUrl } from "../../lib/api";
import { isActive, jobTitle, parseAlerts, parseJobs, parseStatus, toolError, type JobState, type JobStatus, type JobSummary } from "../../lib/jobs";
import {
  LOG_TAIL_BYTES,
  conversationJobIds,
  countDown,
  etaLabel,
  parseStepTimes,
  runEventText,
  runProgress,
  runsToShow,
  RunWatch,
  secondsPerStep,
  shortDuration,
  type RunEvent,
} from "../../lib/runProgress";
import { notifyRunsChanged, onRunsChanged, openRun, openWorkbench } from "../../lib/workbenchBus";
import { ScoreTrend } from "./ScoreTrend";

interface Props {
  sessionId: string | null;
  /** The conversation's tool calls: the runs it started are followed first. */
  steps: Artifacts["steps"];
  /** A short in-app message when a run ends or raises an alert, in any conversation. */
  onNotice?: (message: string) => void;
}

/** get_optimization_status reads the whole results CSV, so not more often than this. */
const POLL_MS = 20_000;
/** list_jobs for runs of other conversations, also while the page is hidden. */
const WATCH_MS = 60_000;
/** How long a finished run stays in the bar. */
const DONE_MS = 8_000;
const ASKED_KEY = "cofold.notify.asked";

const watch = new RunWatch();

/** Asks once ever, the first time a run is seen; the browser remembers the answer. */
function askNotifyOnce() {
  try {
    if (typeof Notification === "undefined" || Notification.permission !== "default" || localStorage.getItem(ASKED_KEY)) return;
    localStorage.setItem(ASKED_KEY, "1");
    void Notification.requestPermission().catch(() => undefined);
  } catch {
    // Blocked storage or no Notification API: in-app messages only.
  }
}

function announce(events: RunEvent[], jobs: readonly JobSummary[], onNotice?: (message: string) => void) {
  for (const event of events) {
    const job = jobs.find((j) => j.id === event.jobId);
    const text = runEventText(event, job ? jobTitle(job) : event.jobId);
    onNotice?.(event.kind === "completed" ? text.title : `${text.title}: ${text.body}`);
    if (document.hidden && typeof Notification !== "undefined" && Notification.permission === "granted") {
      try {
        const note = new Notification(text.title, { body: text.body, tag: event.key });
        note.onclick = () => {
          window.focus();
          openRun(event.jobId, event.kind === "completed" || event.kind === "stopped" ? "results" : "jobs");
          note.close();
        };
      } catch {
        // Some browsers only allow notifications from a service worker.
      }
    }
  }
}

interface Live {
  status: JobStatus;
  secondsPerStep: number | null;
  logStep: number | null;
  /** When the status was read, to count the estimate down from. */
  at: number;
}

const errorText = (error: unknown) => (error instanceof Error ? error.message : String(error));

async function readLogTail(jobId: string): Promise<string> {
  try {
    const response = await fetch(jobFileUrl(jobId, "reinvent.log"), { headers: { Range: `bytes=-${LOG_TAIL_BYTES}` } });
    return response.ok ? await response.text() : "";
  } catch {
    return "";
  }
}

async function readLive(jobId: string, withLog: boolean): Promise<Live | null> {
  const [{ result }, log] = await Promise.all([callTool("get_optimization_status", { job_id: jobId }), withLog ? readLogTail(jobId) : Promise.resolve("")]);
  const status = parseStatus(result);
  if (!status) return null;
  const marks = parseStepTimes(log);
  return { status, secondsPerStep: secondsPerStep(marks), logStep: marks.length ? marks[marks.length - 1].step : null, at: Date.now() };
}

/**
 * "Design is running": a slim bar above the composer while a REINVENT run
 * goes on, with its progress, elapsed time and an estimate of the time left.
 * A click opens the run in the workbench; Stop asks first, then calls
 * stop_optimization. A finished run stays a few seconds with its outcome.
 */
export function RunBar({ sessionId, steps, onNotice }: Props) {
  const conversation = useMemo(() => conversationJobIds(steps), [steps]);
  const conversationKey = conversation.join(",");
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [listTick, setListTick] = useState(0);
  const [live, setLive] = useState<Record<string, Live>>({});
  const [pollTick, setPollTick] = useState(0);
  const [finished, setFinished] = useState<{ job: JobSummary | JobStatus; at: number } | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const [confirm, setConfirm] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const stateOf = (job: JobSummary): JobState => live[job.id]?.status.state ?? job.state;
  const shown = runsToShow(
    jobs.map((job) => ({ ...job, state: stateOf(job) })),
    conversation,
  );
  const shownKey = shown.map((job) => job.id).join(",");
  const first = shown[0] ?? null;

  useEffect(() => {
    let cancelled = false;
    callTool("list_jobs")
      .then(async ({ result }) => {
        if (cancelled || toolError(result)) return;
        const list = parseJobs(result);
        setJobs(list);
        announce(watch.jobs(list), list, onNotice);
        if (!list.some((job) => isActive(job.state))) return;
        const alerts = await callTool("get_job_alerts").catch(() => null);
        if (!cancelled && alerts) announce(watch.alertsOf(parseAlerts(alerts.result)), list, onNotice);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [sessionId, conversationKey, listTick, onNotice]);

  useEffect(() => onRunsChanged(() => setListTick((t) => t + 1)), []);

  useEffect(() => {
    const timer = window.setInterval(() => setListTick((t) => t + 1), WATCH_MS);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!shownKey) return;
    let cancelled = false;
    const ids = shownKey.split(",");
    Promise.all(ids.map((id, i) => readLive(id, i === 0).catch(() => null))).then((results) => {
      if (cancelled) return;
      const next: Record<string, Live> = {};
      results.forEach((entry, i) => {
        if (entry) next[ids[i]] = entry;
      });
      setLive((previous) => ({ ...previous, ...next }));
      const statuses = results.filter((entry): entry is Live => entry !== null).map((entry) => entry.status);
      announce(watch.jobs(statuses), statuses, onNotice);
      const done = results.find((entry) => entry && entry.status.state !== "running" && entry.status.state !== "queued");
      if (done) setFinished({ job: done.status, at: Date.now() });
    });
    return () => {
      cancelled = true;
    };
  }, [shownKey, pollTick, onNotice]);

  const active = shown.length > 0;
  useEffect(() => {
    if (active) askNotifyOnce();
  }, [active]);
  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(() => {
      if (!document.hidden) setPollTick((t) => t + 1);
    }, POLL_MS);
    return () => window.clearInterval(timer);
  }, [active]);

  const ticking = active || finished !== null;
  useEffect(() => {
    if (!ticking) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [ticking]);

  useEffect(() => {
    if (!finished) return;
    const timer = window.setTimeout(() => setFinished(null), Math.max(0, finished.at + DONE_MS - Date.now()));
    return () => window.clearTimeout(timer);
  }, [finished]);

  async function stop(job: JobSummary) {
    setConfirm(false);
    setStopping(true);
    setError(null);
    try {
      const { result } = await callTool("stop_optimization", { job_id: job.id });
      const failure = toolError(result);
      if (failure) setError(failure);
    } catch (reason) {
      setError(errorText(reason));
    } finally {
      setStopping(false);
      setPollTick((t) => t + 1);
      notifyRunsChanged();
    }
  }

  if (first) {
    const entry = live[first.id];
    const input = {
      state: first.state,
      currentStep: entry?.status.currentStep ?? null,
      logStep: entry?.logStep ?? null,
      maxSteps: entry?.status.maxSteps ?? first.maxSteps,
      startedAt: first.startedAt,
      secondsPerStep: entry?.secondsPerStep ?? null,
    };
    const progress = runProgress({ ...input, now });
    const estimate = entry ? runProgress({ ...input, now: entry.at }) : null;
    const remaining = estimate ? countDown(estimate.remainingMs, now - (entry?.at ?? now)) : null;
    const others = shown.slice(1);
    const title = jobTitle(first);

    if (confirm) {
      return (
        <Shell>
          <span className="min-w-0 flex-1 leading-tight">
            Stop <b className="font-semibold">{title}</b> at step {progress.done}? REINVENT saves a checkpoint, so it can be resumed.
          </span>
          <BarButton onClick={() => setConfirm(false)}>Cancel</BarButton>
          <BarButton tone="bad" onClick={() => void stop(first)}>
            Stop run
          </BarButton>
        </Shell>
      );
    }

    return (
      <Shell>
        <button
          type="button"
          onClick={() => openRun(first.id, "jobs")}
          title={`Open ${title} (${first.id}) in the workbench`}
          className="flex min-w-0 flex-1 cursor-pointer items-center gap-2 overflow-hidden text-left"
        >
          <span aria-hidden="true" className="size-1.5 shrink-0 animate-pulse rounded-full" style={{ background: "var(--color-accent)" }} />
          <span className="shrink-0 font-medium">{first.state === "queued" ? "Design is starting" : stopping ? "Stopping design…" : "Design is running"}</span>
          <span className="text-faint hidden min-w-0 truncate @lg:inline">{title}</span>
          <span className="bg-inset h-1 w-10 shrink-0 overflow-hidden rounded-full @md:w-20" aria-hidden="true">
            <span
              className="block h-full rounded-full transition-[width] duration-700"
              style={{ width: `${Math.round((progress.fraction ?? 0) * 100)}%`, background: "var(--color-structure)" }}
            />
          </span>
          <span className="text-muted shrink-0 font-mono tabular-nums" title="Steps done / maximum steps">
            {progress.done}/{progress.max ?? "—"}
          </span>
          <ScoreTrend status={entry?.status} width={56} className="hidden @lg:block" />
          {progress.elapsedMs !== null && (
            <span className="text-faint shrink-0 tabular-nums" title="Elapsed">
              {shortDuration(progress.elapsedMs)}
            </span>
          )}
          <span className="text-faint shrink-0 tabular-nums" title={estimate?.remainingMs != null ? "Estimated from the recent step rate" : "Waiting for a few steps"}>
            {first.state === "queued" ? "" : etaLabel(remaining)}
          </span>
        </button>
        {error && (
          <span className="max-w-48 shrink truncate" style={{ color: "var(--color-accent)" }} title={error}>
            {error}
          </span>
        )}
        {others.length > 0 && (
          <BarButton onClick={() => openWorkbench("jobs")} title={`Also running: ${others.map((job) => `${jobTitle(job)} (${job.id})`).join(", ")}`}>
            +{others.length}
          </BarButton>
        )}
        <BarButton onClick={() => openRun(first.id, "results")} title="Every molecule generated so far">
          Results
        </BarButton>
        {first.type === "optimization" && first.state === "running" && (
          <BarButton tone="bad" disabled={stopping} onClick={() => setConfirm(true)} title="Stop the run (asks first)">
            <StopIcon /> Stop
          </BarButton>
        )}
      </Shell>
    );
  }

  if (finished) {
    const job = finished.job;
    const status = job as Partial<JobStatus>;
    const state = job.state;
    const progress = runProgress({ state, currentStep: status.currentStep ?? null, maxSteps: job.maxSteps, startedAt: job.startedAt, finishedAt: job.finishedAt, now });
    const color = state === "completed" ? "var(--color-ok)" : state === "failed" ? "var(--color-accent)" : "var(--color-warn)";
    const label = state === "completed" ? "Design finished" : state === "failed" ? "Design failed" : "Design stopped";
    return (
      <Shell>
        <button type="button" onClick={() => openRun(job.id, "jobs")} className="flex min-w-0 flex-1 cursor-pointer items-center gap-2 overflow-hidden text-left">
          <span aria-hidden="true" className="size-1.5 shrink-0 rounded-full" style={{ background: color }} />
          <span className="shrink-0 font-medium" style={{ color }}>
            {label}
          </span>
          <span className="text-faint min-w-0 truncate">{jobTitle(job)}</span>
          <span className="text-muted shrink-0 font-mono tabular-nums">
            {progress.done}/{progress.max ?? "—"}
          </span>
          {progress.elapsedMs !== null && <span className="text-faint shrink-0">ran {shortDuration(progress.elapsedMs)}</span>}
        </button>
        {state !== "failed" && <BarButton onClick={() => openRun(job.id, "results")}>Results</BarButton>}
        <BarButton onClick={() => setFinished(null)} title="Dismiss">
          ×
        </BarButton>
      </Shell>
    );
  }

  return null;
}

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <div className="@container mx-auto w-full max-w-3xl px-3 sm:px-4" role="status" aria-live="polite">
      <div className="bg-sunken border-subtle flex min-h-7 items-center gap-1.5 rounded-t-lg border border-b-0 py-0.5 pr-1 pl-2.5 text-[11px]">{children}</div>
    </div>
  );
}

function BarButton({
  children,
  onClick,
  title,
  disabled,
  tone,
}: {
  children: React.ReactNode;
  onClick: () => void;
  title?: string;
  disabled?: boolean;
  tone?: "bad";
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      disabled={disabled}
      className={`inline-flex shrink-0 cursor-pointer items-center gap-1 rounded px-1.5 py-0.5 font-medium disabled:cursor-default disabled:opacity-40 ${
        tone === "bad" ? "text-accent hover:bg-accent-soft" : "text-muted hover:text-body hover:bg-inset"
      }`}
    >
      {children}
    </button>
  );
}

function StopIcon() {
  return (
    <svg width={8} height={8} viewBox="0 0 10 10" aria-hidden="true">
      <rect x={1} y={1} width={8} height={8} rx={1.5} fill="currentColor" />
    </svg>
  );
}
