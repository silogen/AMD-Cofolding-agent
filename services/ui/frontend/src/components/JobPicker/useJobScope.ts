// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useState } from "react";
import { useJobOwners, useRunsKnown, useSessionRuns } from "../../hooks/useSessions";
import { callTool } from "../../lib/api";
import { conversationJobs, parseAlerts, parseJobs, toolError, uniqueAlerts, type JobAlert, type JobSummary } from "../../lib/jobs";
import { scopedJob } from "../../lib/jobScope";
import { onRunsChanged, useFocusedRun } from "../../lib/workbenchBus";

const REFRESH_MS = 15_000;

export interface JobScope {
  /** Every optimization job, null until the list and who launched what are known. */
  runs: JobSummary[] | null;
  /** The jobs this conversation launched; empty when the gateway does not say. */
  own: JobSummary[];
  /** The jobs it only referenced. */
  seen: JobSummary[];
  /** The job Summary, Results and Analysis are about. */
  job: JobSummary | null;
  /** Distinct monitor alerts per job id; jobs without any are absent. */
  alerts: ReadonlyMap<string, number>;
  error: string | null;
}

/** The jobs and the one the Jobs tab is about (scopedJob), read again when runs change and every 15 s. */
export function useJobScope(sessionId: string | null): JobScope {
  const [jobs, setJobs] = useState<JobSummary[] | null>(null);
  const [alertList, setAlertList] = useState<JobAlert[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);
  const focused = useFocusedRun();
  const sessionRuns = useSessionRuns(sessionId);
  const owners = useJobOwners();
  const known = useRunsKnown();

  useEffect(() => onRunsChanged(() => setTick((t) => t + 1)), []);
  useEffect(() => {
    const timer = window.setInterval(() => {
      if (!document.hidden) setTick((t) => t + 1);
    }, REFRESH_MS);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    let cancelled = false;
    Promise.all([callTool("list_jobs", {}), callTool("get_job_alerts", {}).catch(() => null)])
      .then(([list, alerted]) => {
        if (cancelled) return;
        const failure = toolError(list.result);
        setError(failure);
        if (!failure) setJobs(parseJobs(list.result));
        if (alerted) setAlertList(parseAlerts(alerted.result));
      })
      .catch((reason: unknown) => {
        if (!cancelled) setError(reason instanceof Error ? reason.message : String(reason));
      });
    return () => {
      cancelled = true;
    };
  }, [tick]);

  const scope = useMemo(() => {
    if (!jobs || known === null) return null;
    // Without per-conversation runs from the gateway, every job counts as the conversation's, as in Summary, Results and Analysis.
    const mine = known ? conversationJobs(jobs, sessionRuns, focused, owners, sessionId) : { own: [], seen: [] };
    const optimization = (j: JobSummary) => j.type === "optimization";
    return {
      runs: jobs.filter(optimization),
      own: mine.own.filter(optimization),
      seen: mine.seen.filter(optimization),
      job: scopedJob(jobs, known ? mine.own : jobs, focused),
    };
  }, [jobs, known, sessionRuns, focused, owners, sessionId]);

  const alerts = useMemo(() => alertCounts(alertList), [alertList]);
  return { runs: scope?.runs ?? null, own: scope?.own ?? [], seen: scope?.seen ?? [], job: scope?.job ?? null, alerts, error };
}

/** Distinct alerts per job, as the job's own alert list counts them. */
export function alertCounts(alerts: readonly JobAlert[]): Map<string, number> {
  const byJob = new Map<string, JobAlert[]>();
  for (const alert of alerts) byJob.set(alert.jobId, [...(byJob.get(alert.jobId) ?? []), alert]);
  return new Map([...byJob].map(([id, list]) => [id, uniqueAlerts(list).length]));
}

export type JobGroup = "own" | "seen" | "other";

/** The picker's list: this conversation's jobs, then those it referenced, then the rest, each in the gateway's order. */
export function jobOptions<T extends { id: string }>(runs: readonly T[], own: readonly T[], seen: readonly T[] = []): { job: T; group: JobGroup }[] {
  const mine = new Set(own.map((j) => j.id));
  const referenced = new Set(seen.map((j) => j.id).filter((id) => !mine.has(id)));
  const groupOf = (j: T): JobGroup => (mine.has(j.id) ? "own" : referenced.has(j.id) ? "seen" : "other");
  const order: JobGroup[] = ["own", "seen", "other"];
  return order.flatMap((group) => runs.filter((j) => groupOf(j) === group).map((job) => ({ job, group })));
}
