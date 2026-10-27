// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { parseResult } from "./ttp";
import { errorSummary, startedJobId, type JobState } from "./jobs";

/**
 * Live progress of a REINVENT run for the "Design is running" bar: steps
 * done, elapsed time and an estimate of the time left.
 *
 * The step rate comes from REINVENT's log, which stamps every step
 * ("09:41:47 <INFO> Score: 0.96 ... Step: 215"), else from the start time.
 * The log clock is the container's local time without a date, so only the
 * differences between its stamps are used.
 */

/** Fewer steps than this and the rate is too noisy to promise a finish time. */
export const MIN_STEPS = 3;
/** The rate is taken over this many most recent steps, so it follows a run that speeds up or slows down. */
export const RATE_WINDOW = 20;
/** How much of reinvent.log to read: its tail holds the last few dozen steps. */
export const LOG_TAIL_BYTES = 65_536;

export interface StepMark {
  step: number;
  /** Seconds since the first stamp's midnight; grows past 86 400 across midnight. */
  seconds: number;
}

const STEP_LINE = /^(\d{1,2}):(\d{2}):(\d{2})\s+<\w+>.*\bStep:\s*(\d+)/;

/** The time of each step REINVENT logged, in order; a later line for the same step wins. */
export function parseStepTimes(log: string): StepMark[] {
  const marks: StepMark[] = [];
  let day = 0;
  let previous = -1;
  for (const line of log.split(/\r?\n/)) {
    const match = STEP_LINE.exec(line);
    if (!match) continue;
    let seconds = Number(match[1]) * 3600 + Number(match[2]) * 60 + Number(match[3]);
    if (seconds + day * 86_400 < previous) day += 1;
    seconds += day * 86_400;
    previous = seconds;
    const step = Number(match[4]);
    const last = marks[marks.length - 1];
    if (last && last.step === step) marks[marks.length - 1] = { step, seconds };
    else marks.push({ step, seconds });
  }
  return marks;
}

/** Seconds per step over the last `window` logged steps, or null with too few of them. */
export function secondsPerStep(marks: StepMark[], window = RATE_WINDOW): number | null {
  if (marks.length < MIN_STEPS) return null;
  const recent = marks.slice(-window);
  const first = recent[0];
  const last = recent[recent.length - 1];
  const steps = last.step - first.step;
  const seconds = last.seconds - first.seconds;
  return steps > 0 && seconds > 0 ? seconds / steps : null;
}

export interface ProgressInput {
  state: JobState;
  /** get_optimization_status' current_step: the last step in the results CSV plus one. */
  currentStep: number | null;
  /** The last step REINVENT logged, when the log was read. */
  logStep?: number | null;
  maxSteps: number | null;
  startedAt: string | null;
  finishedAt?: string | null;
  /** From the log; without it the rate is the average since the start. */
  secondsPerStep?: number | null;
  now: number;
}

export interface RunProgress {
  done: number;
  max: number | null;
  /** 0..1, or null without a step count to divide by. */
  fraction: number | null;
  elapsedMs: number | null;
  /** Null while estimating: too few steps, or no way to tell the rate. */
  remainingMs: number | null;
}

const time = (iso: string | null | undefined): number | null => {
  if (!iso) return null;
  const ms = new Date(iso).getTime();
  return Number.isFinite(ms) ? ms : null;
};

export function runProgress(input: ProgressInput): RunProgress {
  const max = input.maxSteps && input.maxSteps > 0 ? input.maxSteps : null;
  const fromStatus = input.currentStep != null ? Math.max(0, input.currentStep - 1) : 0;
  let done = Math.max(fromStatus, input.logStep ?? 0);
  if (input.state === "completed" && max) done = max;
  if (max) done = Math.min(done, max);

  const start = time(input.startedAt);
  const end = time(input.finishedAt) ?? input.now;
  const elapsedMs = start !== null ? Math.max(0, end - start) : null;

  let remainingMs: number | null = null;
  if (max && input.state === "running" && done >= MIN_STEPS) {
    const rate = input.secondsPerStep ?? (elapsedMs !== null && done > 0 ? elapsedMs / 1000 / done : null);
    if (rate !== null && rate > 0) remainingMs = (max - done) * rate * 1000;
  }
  if (max && done >= max) remainingMs = 0;

  return { done, max, fraction: max ? done / max : null, elapsedMs, remainingMs };
}

/** An estimate made `ageMs` ago, counted down; never below zero. */
export function countDown(remainingMs: number | null, ageMs: number): number | null {
  return remainingMs === null ? null : Math.max(0, remainingMs - Math.max(0, ageMs));
}

/** "45 s", "12 min", "1 h 05 min": coarse, for a bar that updates every second. */
export function shortDuration(ms: number): string {
  const seconds = Math.max(0, Math.round(ms / 1000));
  if (seconds < 60) return `${seconds} s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min`;
  return `${Math.floor(minutes / 60)} h ${String(minutes % 60).padStart(2, "0")} min`;
}

/** "~7 min left", "estimating", "finishing". */
export function etaLabel(remainingMs: number | null): string {
  if (remainingMs === null) return "estimating";
  if (remainingMs < 15_000) return "finishing";
  return `~${shortDuration(remainingMs)} left`;
}

/** The jobs this conversation started, from the results of its run_optimization calls, oldest first. */
export function conversationJobIds(steps: readonly { name: string; at: number; result?: { result: string; isError: boolean } }[]): string[] {
  const ids: string[] = [];
  for (const step of [...steps].sort((a, b) => a.at - b.at)) {
    if (step.name !== "run_optimization" || !step.result || step.result.isError) continue;
    const id = startedJobId(parseResult(step.result.result));
    if (id && !ids.includes(id)) ids.push(id);
  }
  return ids;
}

/** The runs the bar follows: the conversation's own running jobs, else every running job. */
export function runsToShow<T extends { id: string; state: JobState }>(jobs: readonly T[], conversation: readonly string[]): T[] {
  const live = jobs.filter((job) => job.state === "running" || job.state === "queued");
  const own = live.filter((job) => conversation.includes(job.id));
  return own.length ? own : live;
}

// -- Notifications -------------------------------------------------------------

export interface RunEvent {
  key: string;
  jobId: string;
  kind: "completed" | "failed" | "stopped" | "alert";
  message: string | null;
}

/**
 * What changed since the last look: runs that ended after being seen running,
 * and alerts not seen before. The first look at a job or an alert only
 * records it, so a reload does not announce old news.
 */
export class RunWatch {
  private states = new Map<string, JobState>();
  private alerts = new Set<string>();
  private told = new Set<string>();
  private primed = false;

  jobs(jobs: readonly { id: string; state: JobState; error?: string | null }[]): RunEvent[] {
    const events: RunEvent[] = [];
    for (const job of jobs) {
      const before = this.states.get(job.id);
      this.states.set(job.id, job.state);
      const ended = job.state === "completed" || job.state === "failed" || job.state === "stopped";
      if (ended && (before === "running" || before === "queued")) {
        const event: RunEvent = { key: `${job.id}:${job.state}`, jobId: job.id, kind: job.state as RunEvent["kind"], message: job.error ?? null };
        if (!this.told.has(event.key)) {
          this.told.add(event.key);
          events.push(event);
        }
      }
    }
    return events;
  }

  alertsOf(alerts: readonly { jobId: string; message: string }[]): RunEvent[] {
    const events: RunEvent[] = [];
    for (const alert of alerts) {
      const key = `${alert.jobId}:alert:${alert.message.trim().replace(/\s+/g, " ").toLowerCase()}`;
      if (this.alerts.has(key)) continue;
      this.alerts.add(key);
      if (this.primed) events.push({ key, jobId: alert.jobId, kind: "alert", message: alert.message.trim() });
    }
    this.primed = true;
    return events;
  }
}

/** "Design finished · solubility v1" and a second line, for a toast or a browser notification. */
export function runEventText(event: RunEvent, title: string): { title: string; body: string } {
  switch (event.kind) {
    case "completed":
      return { title: `Design finished · ${title}`, body: "Results are ready." };
    case "stopped":
      return { title: `Design stopped · ${title}`, body: "It can be continued from its checkpoint." };
    case "failed":
      return { title: `Design failed · ${title}`, body: event.message ? errorSummary(event.message) : "See the Jobs tab." };
    default:
      return { title: `Alert · ${title}`, body: event.message ?? "" };
  }
}
