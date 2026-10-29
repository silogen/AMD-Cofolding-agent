// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { refreshSessionList, useJobOwners, useRunsKnown, useSessionRuns } from "../../hooks/useSessions";
import type { Theme } from "../../hooks/useTheme";
import { callTool, download, jobFileUrl } from "../../lib/api";
import { candidatesSdf, candidatesSheet, fileName } from "../../lib/csv";
import { loadRDKit, withMol } from "../../lib/rdkit";
import {
  bestScore,
  canResume,
  conversationJobs,
  convergenceTone,
  errorSummary,
  fmt,
  healthTone,
  int,
  isActive,
  jobTitle,
  parseAlerts,
  parseCandidates,
  parseJobs,
  parseStatus,
  pct,
  progress,
  resultsFile,
  resumePrompt,
  reviewPrompt,
  scoringComponents,
  severityTone,
  stateTone,
  stepsDone,
  toneColor,
  toolError,
  uniqueAlerts,
  type Candidate,
  type JobAlert,
  type JobStatus,
  type JobSummary,
  type Tone,
} from "../../lib/jobs";
import { scopedJob } from "../../lib/jobScope";
import { loadJobStructures, openRunStructure, runStructurePanel, type JobStructure } from "../../lib/jobStructures";
import type { MoleculeContext } from "../../lib/moleculePage";
import { focusRun, notifyRunsChanged, onRunsChanged, openRun, openWorkbench, useFocusedRun } from "../../lib/workbenchBus";
import { DownloadIcon, ExportButtons, TextButton } from "../DataTable/ExportButtons";
import { ScoreTrend } from "../RunBar/ScoreTrend";
import { Badge } from "./Badge";
import { ContinueRun } from "./ContinueRun";
import { TopCandidates } from "./TopCandidates";

interface Props {
  /** The conversation on screen: its job is shown unless another is picked. */
  sessionId?: string;
  theme: Theme;
  /** Hand a request to the chat agent, for what the panel cannot do itself. */
  onAsk: (prompt: string) => void;
  onOpenMolecule?: (smiles: string, context?: MoleculeContext) => void;
}

/** How often to refresh while a job runs. Nothing is polled once all jobs are done. */
const POLL_MS = 10_000;
const TOP_N = 10;

interface Loaded<T> {
  id: string;
  value: T;
  error: string | null;
}

interface Notice {
  tone: Tone;
  text: string;
  ask?: string;
}

const errorText = (error: unknown) => (error instanceof Error ? error.message : String(error));

/**
 * The Jobs tab's job (see scopedJob): its progress, score trajectory,
 * diagnostics, alerts and best molecules. Stop and resume call the tools directly behind a
 * confirmation; anything that needs judgement (adjusting the TTP) goes to the
 * agent through onAsk.
 */
export function JobsPanel({ sessionId, theme, onAsk, onOpenMolecule }: Props) {
  const [allJobs, setJobs] = useState<JobSummary[] | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [alerts, setAlerts] = useState<JobAlert[]>([]);
  const focused = useFocusedRun();
  const runs = useSessionRuns(sessionId);
  const owners = useJobOwners();
  const known = useRunsKnown();
  const summary = useMemo(() => {
    if (!allJobs || known === null) return undefined;
    // A gateway that does not say who launched what: every job is the conversation's.
    return scopedJob(allJobs, known ? conversationJobs(allJobs, runs, focused, owners, sessionId).own : allJobs, focused);
  }, [allJobs, known, runs, focused, owners, sessionId]);
  const [tick, setTick] = useState(0);
  const [status, setStatus] = useState<Loaded<JobStatus | null> | null>(null);
  const [candidates, setCandidates] = useState<Loaded<Candidate[]> | null>(null);
  const [confirm, setConfirm] = useState<"stop" | "continue" | null>(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<Notice | null>(null);

  const activeId = summary?.id ?? null;
  const current = status && status.id === activeId ? status : null;
  const job: JobSummary | JobStatus | null = current?.value ?? summary ?? null;
  const live = summary ? isActive(summary.state) : false;
  const refresh = () => setTick((t) => t + 1);

  useEffect(() => onRunsChanged(() => setTick((t) => t + 1)), []);

  useEffect(() => {
    let cancelled = false;
    refreshSessionList();
    Promise.all([callTool("list_jobs"), callTool("get_job_alerts").catch(() => null)])
      .then(([list, alertList]) => {
        if (cancelled) return;
        const error = toolError(list.result);
        setListError(error);
        if (!error) setJobs(parseJobs(list.result));
        if (alertList) setAlerts(parseAlerts(alertList.result));
      })
      .catch((error: unknown) => {
        if (!cancelled) setListError(errorText(error));
      });
    return () => {
      cancelled = true;
    };
  }, [tick, sessionId]);

  useEffect(() => {
    if (!activeId) return;
    let cancelled = false;
    callTool("get_optimization_status", { job_id: activeId })
      .then(({ result }) => {
        // A failed job's status carries the job's own error; that is not a failure of the call.
        const value = parseStatus(result);
        if (!cancelled) setStatus({ id: activeId, value, error: value ? null : toolError(result) });
      })
      .catch((error: unknown) => {
        if (!cancelled) setStatus({ id: activeId, value: null, error: errorText(error) });
      });
    return () => {
      cancelled = true;
    };
  }, [activeId, tick]);

  const hasResults = summary?.type === "optimization" && ["running", "completed", "stopped"].includes(summary.state);
  useEffect(() => {
    if (!activeId || !hasResults) return;
    let cancelled = false;
    callTool("get_top_candidates", { job_id: activeId, top_n: TOP_N })
      .then(({ result }) => {
        if (!cancelled) setCandidates({ id: activeId, value: parseCandidates(result), error: toolError(result) });
      })
      .catch((error: unknown) => {
        if (!cancelled) setCandidates({ id: activeId, value: [], error: errorText(error) });
      });
    return () => {
      cancelled = true;
    };
  }, [activeId, hasResults, tick]);

  const [structures, setStructures] = useState<Loaded<Map<string, JobStructure>> | null>(null);
  const growing = summary ? isActive(summary.state) : false;
  useEffect(() => {
    if (!activeId || !hasResults) return;
    let cancelled = false;
    loadJobStructures(activeId, growing)
      .then((value) => {
        if (!cancelled) setStructures((previous) => (previous?.id === activeId && previous.value === value ? previous : { id: activeId, value, error: null }));
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [activeId, hasResults, growing, tick]);

  useEffect(() => {
    if (!live) return;
    const timer = window.setInterval(() => {
      if (!document.hidden) setTick((t) => t + 1);
    }, POLL_MS);
    return () => window.clearInterval(timer);
  }, [live]);

  const [shownId, setShownId] = useState(activeId);
  if (shownId !== activeId) {
    setShownId(activeId);
    setConfirm(null);
    setNotice(null);
  }

  async function saveFile(jobId: string, name: string, mime: string) {
    try {
      const response = await fetch(jobFileUrl(jobId, name));
      if (!response.ok) {
        setNotice({ tone: "warn", text: response.status === 404 ? `${name} does not exist for job ${jobId} (yet).` : `Could not download ${name}: HTTP ${response.status}.` });
        return;
      }
      download(`${jobId}-${name}`, await response.text(), mime);
    } catch (error) {
      setNotice({ tone: "bad", text: `Could not download ${name}: ${errorText(error)}` });
    }
  }

  async function saveSdf(jobId: string, list: Candidate[]) {
    try {
      const rdkit = await loadRDKit();
      const sdf = candidatesSdf(list, (smiles) => withMol(rdkit, smiles, (mol) => mol.get_molblock()));
      download(fileName(`${jobId} top candidates`, "sdf"), sdf, "chemical/x-mdl-sdfile");
    } catch (error) {
      setNotice({ tone: "bad", text: `Could not write the SD file: ${errorText(error)}` });
    }
  }

  async function stop(target: JobSummary) {
    setConfirm(null);
    setBusy(true);
    try {
      const { result } = await callTool<Record<string, unknown>>("stop_optimization", { job_id: target.id });
      const error = toolError(result);
      const message = typeof result?.message === "string" ? result.message : "Job stopped";
      setNotice(error ? { tone: "bad", text: error } : { tone: "warn", text: message });
    } catch (error) {
      setNotice({ tone: "bad", text: errorText(error) });
    } finally {
      setBusy(false);
      refresh();
      notifyRunsChanged();
    }
  }

  if (summary === undefined) {
    return (
      <Centered>
        {listError ? (
          <>
            <p style={{ color: "var(--color-accent)" }}>Could not load the jobs: {listError}</p>
            <ActionButton onClick={refresh}>Retry</ActionButton>
          </>
        ) : (
          <p className="text-faint">Loading jobs…</p>
        )}
      </Centered>
    );
  }

  if (!job) {
    return (
      <Centered>
        <p className="text-muted">No job in this conversation yet.</p>
        <ActionButton onClick={() => onAsk("List the stored TTPs and help me start a REINVENT optimization run.")}>
          Ask the agent to start one
        </ActionButton>
      </Centered>
    );
  }

  const s = current?.value ?? null;
  const share = progress(job);
  const diagnostics = s?.diagnostics ?? null;
  const jobAlerts = uniqueAlerts([...(diagnostics?.warnings ?? []), ...alerts.filter((a) => a.jobId === job.id)]);
  const top = candidates && candidates.id === job.id ? candidates : null;
  const best = bestScore(s, top?.value ?? []);
  const complexes = structures && structures.id === job.id ? structures.value : null;

  function showComplex(entry: JobStructure) {
    const panel = job && runStructurePanel(entry.path, entry.smiles, job.id, entry.step, entry.metrics);
    if (panel) openRunStructure(sessionId ?? "new", panel);
  }

  return (
    <div className="@container flex h-full min-h-0 text-xs">
      <div className="min-w-0 flex-1 overflow-y-auto p-3">
        <header className="flex flex-wrap items-start gap-2">
          <div className="min-w-0 flex-1">
            <h2 className="flex items-center gap-2 text-sm font-semibold">
              <span className="truncate">{jobTitle(job)}</span>
              <Badge tone={stateTone(job.state)}>{job.state}</Badge>
              {diagnostics?.convergence && <Badge tone={convergenceTone(diagnostics.convergence)}>{diagnostics.convergence}</Badge>}
              <ScoreTrend status={s} width={72} className="hidden @md:block" />
              <button type="button" onClick={refresh} title="Refresh" aria-label="Refresh" className="text-faint hover:text-body cursor-pointer p-0.5">
                <RefreshIcon />
              </button>
            </h2>
            <div className="-ml-1.5 mt-1 flex flex-wrap items-center gap-0.5">
              {[
                { name: resultsFile(job), label: "Results CSV", mime: "text/csv", hint: "Every molecule REINVENT scored, step by step" },
                { name: "config.toml", label: "config.toml", mime: "text/plain", hint: "The REINVENT configuration the run used" },
                { name: "reinvent.log", label: "reinvent.log", mime: "text/plain", hint: "REINVENT's log" },
              ].map((file) => (
                <TextButton key={file.name} title={`${file.hint} (${file.name})`} onClick={() => void saveFile(job.id, file.name, file.mime)}>
                  <DownloadIcon /> {file.label}
                </TextButton>
              ))}
            </div>
          </div>
          <div className="flex shrink-0 flex-wrap items-center gap-1.5">
            {job.type === "optimization" && (
              <>
                <ActionButton onClick={() => openRun(job.id, "results")}>Results</ActionButton>
                <ActionButton
                  onClick={() => {
                    focusRun(job.id);
                    openWorkbench("activity");
                  }}
                >
                  Analysis
                </ActionButton>
              </>
            )}
            {job.state === "running" && job.type === "optimization" && (
              <ActionButton tone="bad" disabled={busy} onClick={() => setConfirm("stop")}>
                <StopIcon /> Stop
              </ActionButton>
            )}
            {canResume(job) && (
              <ActionButton disabled={busy} onClick={() => setConfirm(confirm === "continue" ? null : "continue")}>
                <PlayIcon /> Continue
              </ActionButton>
            )}
            <ActionButton disabled={busy} onClick={() => onAsk(job.state === "stopped" ? resumePrompt(job) : reviewPrompt(job))}>
              Ask agent
            </ActionButton>
          </div>
        </header>

        {confirm === "stop" && (
          <div className="bg-sunken border-subtle mt-2 flex flex-wrap items-center gap-2 rounded-lg border px-3 py-2">
            <span className="min-w-0 flex-1">Stop {job.id}? REINVENT saves a checkpoint, so the run can be continued later.</span>
            <ActionButton onClick={() => setConfirm(null)}>Cancel</ActionButton>
            <ActionButton tone="bad" onClick={() => void stop(job)}>
              Stop run
            </ActionButton>
          </div>
        )}
        {confirm === "continue" && canResume(job) && (
          <ContinueRun key={job.id} job={job} running={allJobs?.find((j) => isActive(j.state)) ?? null} onClose={() => setConfirm(null)} onAsk={onAsk} />
        )}
        {listError && <Note tone="bad">Could not refresh the jobs: {listError}</Note>}
        {busy && <Note tone="info">Working… stopping a run waits for REINVENT to write its checkpoint.</Note>}
        {notice && (
          <Note tone={notice.tone}>
            {notice.text}
            {notice.ask && (
              <button type="button" onClick={() => onAsk(notice.ask as string)} className="ml-2 cursor-pointer font-semibold underline underline-offset-2">
                Ask the agent instead
              </button>
            )}
          </Note>
        )}

        <div className={`border-subtle mt-3 grid grid-cols-2 border-y ${best === null ? "@lg:grid-cols-4" : "@lg:grid-cols-5"}`}>
          <Stat label="Progress" value={share === null ? "—" : `${Math.round(share * 100)}%`} />
          <Stat
            label={job.type === "transfer_learning" ? "Epochs" : "Step"}
            value={
              job.type === "transfer_learning"
                ? int(job.numEpochs)
                : `${stepsDone(job)} / ${job.maxSteps ?? "—"}`
            }
          />
          <Stat label="Molecules" value={int(s?.molecules)} />
          <Stat label="Valid" value={pct(s?.validity)} />
          {/* Nothing reports a best score on some servers; an em dash in its place only looks broken. */}
          {best !== null && <Stat label="Best" value={fmt(best)} />}
        </div>
        <div className="bg-inset mt-2 h-1.5 overflow-hidden rounded-full">
          <div
            className="h-full rounded-full transition-[width] duration-500"
            style={{ width: `${Math.round((share ?? 0) * 100)}%`, background: "var(--color-structure)" }}
          />
        </div>

        {current?.error && <Note tone="bad">{current.error}</Note>}
        {job.error && (
          <Note tone="bad">
            {errorSummary(job.error) === job.error.trim() ? (
              job.error
            ) : (
              <details>
                <summary className="cursor-pointer">{errorSummary(job.error)}</summary>
                <pre className="text-faint mt-1 max-h-48 overflow-auto font-mono text-[10px] whitespace-pre-wrap">{job.error}</pre>
              </details>
            )}
          </Note>
        )}

        <Alerts alerts={jobAlerts} />

        {diagnostics && (
          <div className="mt-3 grid gap-3 @xl:grid-cols-2">
            <Card title="Diagnostics" inGrid>
              <dl className="grid grid-cols-[max-content_1fr] gap-x-3.5 gap-y-1">
                <Field label="Convergence">{diagnostics.convergence ?? "—"}</Field>
                <Field label="Trend">
                  {diagnostics.trend
                    ? `${fmt(diagnostics.trend.early)} → ${fmt(diagnostics.trend.recent)} (${diagnostics.trend.overall !== null && diagnostics.trend.overall >= 0 ? "+" : ""}${fmt(diagnostics.trend.overall)})`
                    : "—"}
                </Field>
                <Field label="Improvement">{diagnostics.improvementRate === null ? "—" : `${diagnostics.improvementRate.toFixed(4)} / step`}</Field>
                <Field label="Plateau">{diagnostics.plateau ? `yes, since step ${diagnostics.plateauSince ?? "?"}` : "no"}</Field>
                <Field label="Diverging">{diagnostics.diverging ? "yes" : "no"}</Field>
                <Field label="Recent validity">{pct(diagnostics.recentValidity)}</Field>
                <Field label="Latest mean">{fmt(s?.latestMean)}</Field>
              </dl>
              {diagnostics.message && <p className="text-faint mt-2">{diagnostics.message}</p>}
            </Card>
            <Card title="Component health" inGrid>
              {diagnostics.health.length ? (
                diagnostics.health.map((h) => {
                  const stats = scoringComponents(s).find((c) => c.name === h.name);
                  return (
                    <div key={h.name} className="grid grid-cols-[minmax(0,1fr)_auto_auto] items-center gap-2 py-0.5">
                      <span className="truncate" title={h.name}>{h.name}</span>
                      <span className="text-muted font-mono tabular-nums" title="mean / max">
                        {fmt(stats?.mean ?? h.mean, 2)}
                        {stats?.max != null && <span className="text-faint"> / {fmt(stats.max, 2)}</span>}
                      </span>
                      <Badge tone={healthTone(h.status)} title={h.nanRate !== null ? `${pct(h.nanRate)} NaN` : undefined}>
                        {h.status}
                      </Badge>
                    </div>
                  );
                })
              ) : (
                <p className="text-faint">No component data yet.</p>
              )}
            </Card>
          </div>
        )}

        {diagnostics && diagnostics.recommendations.length > 0 && (
          <Card title="Recommendations">
            <ul className="list-disc space-y-1 pl-4">
              {diagnostics.recommendations.map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ul>
            <div className="mt-2">
              <ActionButton onClick={() => onAsk(reviewPrompt(job))}>Ask the agent to act on these</ActionButton>
            </div>
          </Card>
        )}

        {hasResults && (
          <Card
            title={`Top candidates${top?.value.length ? ` · ${top.value.length}` : ""}`}
            actions={
              top && top.value.length > 0 ? (
                <>
                  <ExportButtons name={`${job.id} top candidates`} sheet={() => candidatesSheet(top.value)} scope={`the top ${top.value.length} candidates`} />
                  <TextButton title="Download the top candidates as an SD file with their scores" onClick={() => void saveSdf(job.id, top.value)}>
                    <DownloadIcon /> SDF
                  </TextButton>
                </>
              ) : null
            }
          >
            {top?.error ? (
              <p className="text-faint">{top.error}</p>
            ) : !top ? (
              <p className="text-faint">Loading candidates…</p>
            ) : !top.value.length ? (
              <p className="text-faint">No valid molecules yet.</p>
            ) : (
              <TopCandidates
                candidates={top.value}
                complexes={complexes}
                jobId={job.id}
                session={sessionId ?? "new"}
                theme={theme}
                onOpen={onOpenMolecule && ((smiles) => onOpenMolecule(smiles, { run: job.id }))}
                onOpenStructure={showComplex}
              />
            )}
          </Card>
        )}
      </div>
    </div>
  );
}

const SEVERITY_ORDER = ["critical", "error", "high", "warning", "medium", "low", "info"];

/** The run's alerts, folded under a count until asked for. */
function Alerts({ alerts }: { alerts: { severity: string; message: string }[] }) {
  if (!alerts.length) return null;
  const worst = [...alerts].sort((a, b) => SEVERITY_ORDER.indexOf(a.severity.toLowerCase()) - SEVERITY_ORDER.indexOf(b.severity.toLowerCase()))[0];
  return (
    <details className="group mt-2">
      <summary className="text-muted hover:text-body inline-flex cursor-pointer list-none items-center gap-1.5 text-[11px] select-none [&::-webkit-details-marker]:hidden">
        <svg width={8} height={8} viewBox="0 0 10 10" aria-hidden="true" className="transition-transform group-open:rotate-90">
          <path d="M3.5 2 6.5 5 3.5 8" fill="none" stroke="currentColor" strokeWidth={1.4} />
        </svg>
        Alerts
        <Badge tone={severityTone(worst.severity)}>{alerts.length}</Badge>
      </summary>
      {alerts.map((a) => (
        <Note key={a.message} tone={severityTone(a.severity)}>
          <span className="text-faint mr-1.5 text-[10px] uppercase">{a.severity}</span>
          {a.message}
        </Note>
      ))}
    </details>
  );
}

function Centered({ children }: { children: ReactNode }) {
  return <div className="flex h-full flex-col items-center justify-center gap-2 p-6 text-center text-xs">{children}</div>;
}

function Card({ title, children, inGrid, actions }: { title: string; children: ReactNode; inGrid?: boolean; actions?: ReactNode }) {
  return (
    <section className={`bg-raised border-subtle shadow-card rounded-xl border px-3 py-2.5 ${inGrid ? "" : "mt-3"}`}>
      <div className="mb-2 flex items-center gap-2">
        <h3 className="text-faint text-[10px] font-semibold tracking-wider uppercase">{title}</h3>
        {actions && <div className="-my-1 ml-auto flex items-center gap-0.5">{actions}</div>}
      </div>
      {children}
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="border-subtle flex flex-col gap-0.5 border-r px-3 py-2 last:border-r-0">
      <span className="text-faint text-[10px]">{label}</span>
      <b className="font-mono text-sm font-semibold tabular-nums">{value}</b>
    </div>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <>
      <dt className="text-muted">{label}</dt>
      <dd className="font-mono tabular-nums">{children}</dd>
    </>
  );
}

function Note({ tone, children }: { tone: Tone; children: ReactNode }) {
  return (
    <div className="text-muted mt-1.5 border-l-2 py-0.5 pl-2.5" style={{ borderLeftColor: toneColor(tone) }}>
      {children}
    </div>
  );
}

function ActionButton({
  children,
  onClick,
  disabled,
  tone,
}: {
  children: ReactNode;
  onClick: () => void;
  disabled?: boolean;
  tone?: "primary" | "bad";
}) {
  const style =
    tone === "primary"
      ? "bg-accent hover:bg-accent-strong border-transparent text-white"
      : tone === "bad"
        ? "bg-raised border-subtle text-accent hover:bg-accent-soft"
        : "bg-raised border-subtle text-body hover:bg-inset";
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex cursor-pointer items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-medium disabled:cursor-not-allowed disabled:opacity-50 ${style}`}
    >
      {children}
    </button>
  );
}

function StopIcon() {
  return (
    <svg width={10} height={10} viewBox="0 0 10 10" aria-hidden="true">
      <rect x={1} y={1} width={8} height={8} rx={1.5} fill="currentColor" />
    </svg>
  );
}

function PlayIcon() {
  return (
    <svg width={10} height={10} viewBox="0 0 10 10" aria-hidden="true">
      <path d="M2 1.2v7.6L8.6 5z" fill="currentColor" />
    </svg>
  );
}

function RefreshIcon() {
  return (
    <svg width={12} height={12} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth={1.6} aria-hidden="true">
      <path d="M13.5 8a5.5 5.5 0 1 1-1.6-3.9M13.5 2.5v3h-3" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
