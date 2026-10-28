// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { memo, useEffect, useMemo, useState } from "react";
import type { Artifacts } from "../../lib/artifacts";
import { callTool, jobFileUrl } from "../../lib/api";
import { isActive, jobTitle, parseAlerts, parseJobs, parseStatus, toolError, type JobState, type JobStatus, type JobSummary } from "../../lib/jobs";
import {
  LOG_TAIL_BYTES,
  countDown,
  ownJobIds,
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
import { cofoldShown, inFlightCofold, runUsesBoltz, type CofoldStatus } from "../../lib/cofold";
import { refreshSessionList, useSessionRuns } from "../../hooks/useSessions";
import { Bar, CofoldRow, Eta } from "./CofoldRow";
import { ScoreTrend } from "./ScoreTrend";
import { useCofoldStatus } from "./useCofoldStatus";

interface Props {
  sessionId: string | null;
  /** The conversation's tool calls: the runs it started are followed first. */
  steps: Artifacts["steps"];
  /** When the agent turn in progress started; null when none is. Its open calls may be cofolding. */
  turnAt?: number | null;
  /** A short in-app message when a run ends or raises an alert, in any conversation. */
  onNotice?: (message: string) => void;
}

/** get_optimization_status reads the whole results CSV, so not more often than this. */
const POLL_MS = 20_000;
/**
 * list_jobs while a run is going on, for runs of other conversations. Not
 * while the page is hidden: the list is read again as soon as it is shown,
 * and RunWatch then reports whatever ended meanwhile.
 */
const WATCH_MS = 60_000;
/** How long a finished run stays in the bar. */
const DONE_MS = 8_000;
/** How long a run's finished cofold batch stays while the run prepares the next. */
const HOLD_MS = 180_000;
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
 * "Designing": a slim bar above the composer while a REINVENT run of this
 * conversation goes on, with its progress, elapsed time and an estimate of
 * the time left, and under it what Boltz cofolds for the conversation.
 * A click opens the run in the workbench; Stop asks first, then calls
 * stop_optimization. A finished run stays a few seconds with its outcome.
 */
export const RunBar = memo(function RunBar({ sessionId, steps, turnAt = null, onNotice }: Props) {
  const runs = useSessionRuns(sessionId);
  const conversation = useMemo(() => ownJobIds(steps, runs), [steps, runs]);
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

  useEffect(
    () =>
      onRunsChanged(() => {
        setListTick((t) => t + 1);
        refreshSessionList();
      }),
    [],
  );

  const anyRunning = jobs.some((job) => isActive(job.state));
  useEffect(() => {
    if (!anyRunning) return;
    const timer = window.setInterval(() => {
      if (!document.hidden) setListTick((t) => t + 1);
    }, WATCH_MS);
    return () => window.clearInterval(timer);
  }, [anyRunning]);

  useEffect(() => {
    const onVisible = () => {
      if (!document.hidden) setListTick((t) => t + 1);
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => document.removeEventListener("visibilitychange", onVisible);
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

  const cofoldCalls = useMemo(() => inFlightCofold(steps, turnAt), [steps, turnAt]);
  const designing = shown.filter((job) => job.type === "optimization" && job.state === "running");
  const boltzUse = designing.map((job) => runUsesBoltz(steps, job.ttpId, live[job.id]?.status.components));
  const boltzRun = boltzUse.includes(true) ? true : boltzUse.includes(null) ? null : false;
  const cofold = useCofoldStatus(cofoldCalls, boltzRun, designing.length > 0);
  const ownActive = cofold.status?.available === true && cofold.status.active;
  const [held, setHeld] = useState<{ session: string | null; status: CofoldStatus; at: number } | null>(null);
  if (ownActive && cofold.status && (held?.at !== cofold.at || held.session !== sessionId)) setHeld({ session: sessionId, status: cofold.status, at: cofold.at });
  const holding =
    !ownActive && held !== null && held.session === sessionId && designing.length > 0 && boltzRun !== false && cofold.at - held.at < HOLD_MS;
  const cofoldRow = cofoldShown(cofold.status, cofoldCalls) ? (
    <CofoldRow status={cofold.status} at={cofold.at} calls={cofoldCalls} memory={cofold.memory} />
  ) : holding ? (
    <CofoldRow status={held.status} at={held.at} calls={[]} finished />
  ) : null;

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
        <Shell extra={cofoldRow}>
          <span className="col-span-8 min-w-0 leading-tight whitespace-normal">
            Stop <b className="font-semibold">{title}</b> at step {progress.done}? REINVENT saves a checkpoint, so it can be resumed.
          </span>
          <Actions>
            <BarButton onClick={() => setConfirm(false)}>Cancel</BarButton>
            <BarButton tone="bad" onClick={() => void stop(first)}>
              Stop run
            </BarButton>
          </Actions>
        </Shell>
      );
    }

    return (
      <Shell extra={cofoldRow}>
        <button
          type="button"
          onClick={() => openRun(first.id, "jobs")}
          title={`Open ${title} (${first.id}) in the workbench`}
          className="col-span-8 grid min-w-0 cursor-pointer grid-cols-subgrid items-center text-left"
        >
          <span aria-hidden="true" className="size-1.5 animate-pulse rounded-full" style={{ background: "var(--color-accent)" }} />
          <span className="font-medium">{first.state === "queued" ? "Starting" : stopping ? "Stopping" : "Designing"}</span>
          <span className="text-faint min-w-0 truncate">{error ? <span style={{ color: "var(--color-accent)" }}>{error}</span> : title}</span>
          <Bar fraction={progress.fraction ?? 0} />
          <span className="text-muted text-right font-mono tabular-nums" title="Steps done / maximum steps">
            {progress.done}/{progress.max ?? "—"}
          </span>
          <span>
            <ScoreTrend status={entry?.status} width={56} className="hidden @lg:block" />
          </span>
          <span className="text-faint text-right tabular-nums" title="Elapsed">
            {progress.elapsedMs !== null ? shortDuration(progress.elapsedMs) : ""}
          </span>
          <span className="text-faint tabular-nums" title={estimate?.remainingMs != null ? "Estimated from the recent step rate" : "Waiting for a few steps"}>
            {first.state === "queued" ? "" : <Eta remainingMs={remaining} />}
          </span>
        </button>
        <Actions>
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
        </Actions>
      </Shell>
    );
  }

  if (finished) {
    const job = finished.job;
    const status = job as Partial<JobStatus>;
    const state = job.state;
    const progress = runProgress({ state, currentStep: status.currentStep ?? null, maxSteps: job.maxSteps, startedAt: job.startedAt, finishedAt: job.finishedAt, now });
    const color = state === "completed" ? "var(--color-ok)" : state === "failed" ? "var(--color-accent)" : "var(--color-warn)";
    const label = state === "completed" ? "Finished" : state === "failed" ? "Failed" : "Stopped";
    return (
      <Shell extra={cofoldRow}>
        <button type="button" onClick={() => openRun(job.id, "jobs")} className="col-span-8 grid min-w-0 cursor-pointer grid-cols-subgrid items-center text-left">
          <span aria-hidden="true" className="size-1.5 rounded-full" style={{ background: color }} />
          <span className="font-medium" style={{ color }}>
            {label}
          </span>
          <span className="text-faint min-w-0 truncate">{jobTitle(job)}</span>
          <Bar fraction={progress.fraction} colour={color} />
          <span className="text-muted text-right font-mono tabular-nums">
            {progress.done}/{progress.max ?? "—"}
          </span>
          <span />
          <span className="text-faint text-right tabular-nums" title="Ran for">
            {progress.elapsedMs !== null ? shortDuration(progress.elapsedMs) : ""}
          </span>
          <span />
        </button>
        <Actions>
          {state !== "failed" && <BarButton onClick={() => openRun(job.id, "results")}>Results</BarButton>}
          <BarButton onClick={() => setFinished(null)} title="Dismiss">
            ×
          </BarButton>
        </Actions>
      </Shell>
    );
  }

  return cofoldRow ? <Shell>{cofoldRow}</Shell> : null;
});

/** Every line shares the bar's columns: dot, label, detail, progress, count, extra, elapsed, time left, actions. */
const ROW = "col-span-full grid min-h-7 grid-cols-subgrid items-center py-0.5 pr-1 pl-2.5";

/** The bar; *extra* is a second line under the run, e.g. what Boltz is cofolding. */
function Shell({ children, extra }: { children: React.ReactNode; extra?: React.ReactNode }) {
  return (
    <div className="@container mx-auto w-full max-w-3xl px-3 sm:px-4" role="status" aria-live="polite">
      <div className="bg-sunken border-subtle grid grid-cols-[auto_auto_minmax(0,1fr)_auto_auto_auto_auto_auto_auto] gap-x-1.5 rounded-t-lg @lg:gap-x-2 border border-b-0 text-[11px] whitespace-nowrap">
        <div className={ROW}>{children}</div>
        {extra && <div className={`${ROW} border-subtle border-t`}>{extra}</div>}
      </div>
    </div>
  );
}

function Actions({ children }: { children: React.ReactNode }) {
  return <span className="col-start-9 -my-0.5 flex items-center justify-end gap-0.5">{children}</span>;
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
