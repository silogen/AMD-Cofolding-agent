// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useState, type ReactNode } from "react";
import type { Theme } from "../../hooks/useTheme";
import { callTool, download, jobFileUrl } from "../../lib/api";
import { candidatesSdf, candidatesSheet, fileName } from "../../lib/csv";
import { duration } from "../../lib/format";
import { loadRDKit, withMol } from "../../lib/rdkit";
import {
  bestScore,
  canResume,
  candidateScores,
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
  runtimeMs,
  scoringComponents,
  severityTone,
  shortDate,
  stateTone,
  stepsDone,
  toneColor,
  toolError,
  typeLabel,
  uniqueAlerts,
  type Candidate,
  type JobAlert,
  type JobStatus,
  type JobSummary,
  type Tone,
} from "../../lib/jobs";
import { focusRun, notifyRunsChanged, openRun, useFocusedRun } from "../../lib/workbenchBus";
import { DownloadIcon, ExportButtons, TextButton } from "../DataTable/ExportButtons";
import { MolDepiction } from "../MolDepiction/MolDepiction";
import { SmilesText } from "../MolDepiction/SmilesText";
import { ScoreTrend } from "../RunBar/ScoreTrend";
import { StarButton } from "../Shortlist/StarButton";
import { Badge } from "./Badge";
import { ContinueRun } from "./ContinueRun";
import { TrajectoryChart } from "./TrajectoryChart";

interface Props {
  /** The conversation on screen; a new one reloads the list. Jobs themselves are global. */
  sessionId?: string;
  theme: Theme;
  /** Hand a request to the chat agent, for what the panel cannot do itself. */
  onAsk: (prompt: string) => void;
  onOpenMolecule?: (smiles: string) => void;
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
 * REINVENT runs: every job the MCP job manager tracks, and for the selected
 * one its progress, score trajectory, diagnostics, alerts and best molecules.
 * Stop and resume call the tools directly behind a confirmation; anything
 * that needs judgement (adjusting the TTP) goes to the agent through onAsk.
 */
export function JobsPanel({ sessionId, theme, onAsk, onOpenMolecule }: Props) {
  const [jobs, setJobs] = useState<JobSummary[] | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [alerts, setAlerts] = useState<JobAlert[]>([]);
  const selectedId = useFocusedRun();
  const [tick, setTick] = useState(0);
  const [status, setStatus] = useState<Loaded<JobStatus | null> | null>(null);
  const [candidates, setCandidates] = useState<Loaded<Candidate[]> | null>(null);
  const [confirm, setConfirm] = useState<"stop" | "continue" | null>(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<Notice | null>(null);

  const activeId = selectedId && jobs?.some((j) => j.id === selectedId) ? selectedId : (jobs?.[0]?.id ?? null);
  const summary = jobs?.find((j) => j.id === activeId) ?? null;
  const current = status && status.id === activeId ? status : null;
  const job: JobSummary | JobStatus | null = current?.value ?? summary;
  const live = (jobs ?? []).some((j) => isActive(j.state));
  const refresh = () => setTick((t) => t + 1);

  useEffect(() => {
    let cancelled = false;
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

  useEffect(() => {
    if (!live) return;
    const timer = window.setInterval(() => {
      if (!document.hidden) setTick((t) => t + 1);
    }, POLL_MS);
    return () => window.clearInterval(timer);
  }, [live]);

  function select(id: string) {
    focusRun(id);
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

  if (jobs === null) {
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

  if (!jobs.length || !job) {
    return (
      <Centered>
        <p className="text-body font-medium">No REINVENT runs yet</p>
        <p className="text-muted max-w-sm">Optimization and transfer-learning jobs the agent launches show up here with their progress.</p>
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
  const runtime = runtimeMs(job);
  const best = bestScore(s, top?.value ?? []);

  return (
    <div className="@container flex h-full min-h-0 text-xs">
      <aside className="border-subtle w-44 shrink-0 overflow-y-auto border-r p-2 @2xl:w-52">
        <div className="mb-1 flex items-center px-1">
          <h3 className="text-faint text-[10px] font-semibold tracking-wider uppercase">Runs</h3>
          <span className="text-faint ml-1 text-[10px]">{jobs.length}</span>
          <button type="button" onClick={refresh} title="Refresh" aria-label="Refresh" className="text-faint hover:text-body ml-auto cursor-pointer p-0.5">
            <RefreshIcon />
          </button>
        </div>
        {listError && <p className="px-1 pb-1 text-[10px]" style={{ color: "var(--color-accent)" }}>{listError}</p>}
        {jobs.map((j) => {
          const count = uniqueAlerts(alerts.filter((a) => a.jobId === j.id)).length;
          return (
            <button
              key={j.id}
              type="button"
              onClick={() => select(j.id)}
              aria-current={j.id === job.id}
              className={`mb-0.5 flex w-full cursor-pointer flex-col items-start gap-0.5 rounded-md px-2 py-1.5 text-left ${j.id === job.id ? "bg-inset" : "hover:bg-sunken"}`}
            >
              <span className="flex w-full items-center gap-1.5">
                <b className="min-w-0 flex-1 truncate font-semibold">{jobTitle(j)}</b>
                {count > 0 && (
                  <span className="shrink-0 text-[10px] font-semibold" style={{ color: toneColor("bad") }} title={`${count} alert${count === 1 ? "" : "s"}`}>
                    ⚠ {count}
                  </span>
                )}
              </span>
              <span className="text-faint w-full truncate font-mono text-[10px]">
                {j.id} · {shortDate(j.createdAt)}
              </span>
              <Badge tone={stateTone(j.state)}>{j.state}</Badge>
            </button>
          );
        })}
      </aside>

      <div className="min-w-0 flex-1 overflow-y-auto p-3">
        <header className="flex flex-wrap items-start gap-2">
          <div className="min-w-0 flex-1">
            <h2 className="flex items-center gap-2 text-sm font-semibold">
              <span className="truncate">{jobTitle(job)}</span>
              <Badge tone={stateTone(job.state)}>{job.state}</Badge>
              {diagnostics?.convergence && <Badge tone={convergenceTone(diagnostics.convergence)}>{diagnostics.convergence}</Badge>}
              <ScoreTrend status={s} width={72} className="hidden @md:block" />
            </h2>
            <p className="text-faint mt-0.5 truncate font-mono text-[10px]">
              {job.id} · {typeLabel(job.type)} · {job.model ?? "default model"}
              {runtime !== null && ` · ran ${duration(runtime)}`}
            </p>
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
              <ActionButton onClick={() => openRun(job.id, "results")}>Results</ActionButton>
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
          <ContinueRun key={job.id} job={job} running={jobs.find((j) => isActive(j.state)) ?? null} onClose={() => setConfirm(null)} onAsk={onAsk} />
        )}
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

        <Card title="Score trajectory · mean / best / components">
          {s ? <TrajectoryChart status={s} /> : <p className="text-faint py-8 text-center">{current ? "No status available." : "Loading status…"}</p>}
        </Card>

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

        {jobAlerts.length > 0 && (
          <section className="mt-3">
            <h3 className="text-faint mb-1 text-[10px] font-semibold tracking-wider uppercase">Alerts · {jobAlerts.length}</h3>
            {jobAlerts.map((a) => (
              <Note key={a.message} tone={severityTone(a.severity)}>
                <span className="text-faint mr-1.5 text-[10px] uppercase">{a.severity}</span>
                {a.message}
              </Note>
            ))}
          </section>
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
              <div className="-mx-3 overflow-x-auto">
                <table className="w-full text-[11px]">
                  <thead>
                    <tr className="text-muted border-subtle border-b text-left">
                      <th className="px-3 py-1.5 font-medium">Molecule</th>
                      <th className="px-2 py-1.5 text-right font-medium">Total</th>
                      {top.value[0].components.map((c) => (
                        <th key={c.name} className="max-w-24 truncate px-2 py-1.5 text-right font-medium" title={c.name}>
                          {c.name}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {top.value.map((c) => (
                      <tr key={c.smiles} className="group border-subtle border-b last:border-b-0">
                        <td className="px-3 py-1">
                          <span className="flex items-center gap-2">
                            <StarButton smiles={c.smiles} source={job.id} values={() => candidateScores(c)} session={sessionId ?? "new"} />
                            <button
                              type="button"
                              onClick={() => onOpenMolecule?.(c.smiles)}
                              disabled={!onOpenMolecule}
                              className="enabled:hover:bg-inset h-12 w-16 shrink-0 rounded enabled:cursor-pointer"
                              title="Open the 2D drawing"
                            >
                              <MolDepiction smiles={c.smiles} theme={theme} width={128} height={96} className="size-full" />
                            </button>
                            <span className="flex min-w-0 flex-col">
                              <SmilesText smiles={c.smiles} className="max-w-28 text-[10px] @xl:max-w-44" />
                            </span>
                          </span>
                        </td>
                        <td className="px-2 py-1 text-right font-mono font-semibold tabular-nums">{fmt(c.total)}</td>
                        {top.value[0].components.map((col) => {
                          const value = c.components.find((x) => x.name === col.name);
                          return (
                            <td key={col.name} className="px-2 py-1 text-right font-mono tabular-nums" title={value?.raw != null ? `raw ${value.raw}` : undefined}>
                              {fmt(value?.transformed, 2)}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        )}

        <p className="text-faint mt-3 font-mono text-[10px] break-all">
          {job.workingDir}
          {live && " · refreshing every 10 s"}
        </p>
      </div>
    </div>
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
