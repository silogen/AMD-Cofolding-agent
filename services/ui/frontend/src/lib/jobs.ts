// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * REINVENT jobs as the MCP job manager reports them (reinvent4_mcp/jobs/manager.py).
 *
 * The tools answer loosely typed JSON: snake_case keys, fields that only
 * exist for one job type, a trajectory that is missing until the first CSV
 * row is written. Everything is normalised here so the panel only deals with
 * camelCase fields that are either present or null.
 */

export type JobState = "queued" | "running" | "completed" | "failed" | "stopped" | "unknown";

export type Tone = "ok" | "warn" | "bad" | "info" | "muted";

export interface JobSummary {
  id: string;
  ttpId: string | null;
  ttpVersion: number | null;
  model: string | null;
  type: string;
  state: JobState;
  createdAt: string | null;
  startedAt: string | null;
  finishedAt: string | null;
  error: string | null;
  maxSteps: number | null;
  numEpochs: number | null;
  csvPath: string | null;
  checkpointPath: string | null;
  hasCheckpoint: boolean;
  workingDir: string | null;
}

export interface TrajectoryPoint {
  step: number;
  mean: number;
  max: number | null;
  molecules: number | null;
}

export interface ComponentStats {
  name: string;
  mean: number | null;
  max: number | null;
  min: number | null;
}

export interface ComponentHealth {
  name: string;
  status: string;
  nanRate: number | null;
  mean: number | null;
}

export interface JobWarning {
  type: string;
  severity: string;
  message: string;
  component: string | null;
}

export interface JobDiagnostics {
  convergence: string | null;
  message: string | null;
  plateau: boolean;
  plateauSince: number | null;
  diverging: boolean;
  improvementRate: number | null;
  progressPct: number | null;
  trend: { early: number | null; mid: number | null; recent: number | null; overall: number | null } | null;
  recentValidity: number | null;
  health: ComponentHealth[];
  warnings: JobWarning[];
  recommendations: string[];
}

export interface JobStatus extends JobSummary {
  currentStep: number | null;
  molecules: number | null;
  valid: number | null;
  validity: number | null;
  meanScore: number | null;
  bestScore: number | null;
  latestMean: number | null;
  components: ComponentStats[];
  trajectory: TrajectoryPoint[];
  componentTrajectories: Record<string, { step: number; mean: number }[]>;
  diagnostics: JobDiagnostics | null;
}

export interface JobAlert {
  jobId: string;
  severity: string;
  type: string;
  message: string;
}

export interface Candidate {
  smiles: string;
  total: number | null;
  components: { name: string; transformed: number | null; raw: number | null }[];
}

type Json = Record<string, unknown>;

const obj = (v: unknown): Json | null => (v && typeof v === "object" && !Array.isArray(v) ? (v as Json) : null);
const str = (v: unknown): string | null => (typeof v === "string" && v !== "" ? v : null);
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const list = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);

const STATES: readonly JobState[] = ["queued", "running", "completed", "failed", "stopped"];

/**
 * The `error` a tool answers instead of a result, if any. The gateway passes
 * on a result that is not JSON as text, which for these tools is a failure.
 */
export function toolError(result: unknown): string | null {
  if (typeof result === "string") return result.trim() || "The tool returned no result";
  return str(obj(result)?.error);
}

export function parseJob(raw: unknown): JobSummary | null {
  const d = obj(raw);
  const id = str(d?.job_id);
  if (!d || !id) return null;
  const state = str(d.state);
  return {
    id,
    ttpId: str(d.ttp_id),
    ttpVersion: num(d.ttp_version),
    model: str(d.model),
    type: str(d.job_type) ?? "optimization",
    state: STATES.includes(state as JobState) ? (state as JobState) : "unknown",
    createdAt: str(d.created_at),
    startedAt: str(d.started_at),
    finishedAt: str(d.finished_at),
    error: str(d.error),
    maxSteps: num(d.max_steps),
    numEpochs: num(d.num_epochs),
    csvPath: str(d.csv_path),
    checkpointPath: str(d.checkpoint_path),
    hasCheckpoint: d.has_checkpoint === true,
    workingDir: str(d.working_dir),
  };
}

/** list_jobs, newest first. */
export function parseJobs(result: unknown): JobSummary[] {
  return list(result)
    .map(parseJob)
    .filter((j): j is JobSummary => j !== null)
    .sort((a, b) => (b.createdAt ?? "").localeCompare(a.createdAt ?? ""));
}

function parseWarning(raw: unknown): JobWarning | null {
  const d = obj(raw);
  const message = str(d?.message);
  if (!d || !message) return null;
  return {
    type: str(d.type) ?? "warning",
    severity: str(d.severity) ?? "medium",
    message,
    component: str(d.component),
  };
}

function parseDiagnostics(raw: unknown): JobDiagnostics | null {
  const d = obj(raw);
  if (!d) return null;
  const trend = obj(d.trend);
  return {
    convergence: str(d.convergence_status),
    message: str(d.message),
    plateau: d.plateau_detected === true,
    plateauSince: num(d.plateau_since_step),
    diverging: d.diverging === true,
    improvementRate: num(d.improvement_rate),
    progressPct: num(d.progress_pct),
    trend: trend
      ? { early: num(trend.early_mean), mid: num(trend.mid_mean), recent: num(trend.recent_mean), overall: num(trend.overall_improvement) }
      : null,
    recentValidity: num(d.recent_validity_rate),
    health: Object.entries(obj(d.component_health) ?? {}).map(([name, h]) => {
      const e = obj(h);
      return { name, status: str(e?.status) ?? "unknown", nanRate: num(e?.nan_rate), mean: num(e?.mean) };
    }),
    warnings: list(d.warnings)
      .map(parseWarning)
      .filter((w): w is JobWarning => w !== null),
    recommendations: list(d.recommendations).filter((r): r is string => typeof r === "string"),
  };
}

/** get_optimization_status: the job plus its trajectory, when the CSV exists. */
export function parseStatus(result: unknown): JobStatus | null {
  const job = parseJob(result);
  if (!job) return null;
  const t = obj(obj(result)?.trajectory) ?? {};
  const trajectory = list(t.score_trajectory)
    .map((p): TrajectoryPoint | null => {
      const e = obj(p);
      const step = num(e?.step);
      const mean = num(e?.mean_score);
      return step === null || mean === null ? null : { step, mean, max: num(e?.max_score), molecules: num(e?.n_molecules) };
    })
    .filter((p): p is TrajectoryPoint => p !== null)
    .sort((a, b) => a.step - b.step);
  const componentTrajectories: JobStatus["componentTrajectories"] = {};
  for (const [name, points] of Object.entries(obj(t.per_component_trajectory) ?? {})) {
    const clean = list(points)
      .map((p) => {
        const e = obj(p);
        const step = num(e?.step);
        const mean = num(e?.mean_score);
        return step === null || mean === null ? null : { step, mean };
      })
      .filter((p): p is { step: number; mean: number } => p !== null);
    if (clean.length) componentTrajectories[name] = clean;
  }
  return {
    ...job,
    maxSteps: num(t.max_steps) ?? job.maxSteps,
    currentStep: num(t.current_step),
    molecules: num(t.n_molecules_total),
    valid: num(t.n_valid),
    validity: num(t.validity_rate),
    meanScore: num(t.mean_total_score),
    bestScore: num(t.best_total_score),
    latestMean: num(t.latest_mean_score),
    components: Object.entries(obj(t.per_component) ?? {}).map(([name, s]) => {
      const e = obj(s);
      return { name, mean: num(e?.mean), max: num(e?.max), min: num(e?.min) };
    }),
    trajectory,
    componentTrajectories,
    diagnostics: parseDiagnostics(t.diagnostics),
  };
}

/** get_job_alerts: a list over every running job. */
export function parseAlerts(result: unknown): JobAlert[] {
  return list(result)
    .map((raw): JobAlert | null => {
      const d = obj(raw);
      const jobId = str(d?.job_id);
      const message = str(d?.message);
      if (!d || !jobId || !message) return null;
      return { jobId, message, severity: str(d.severity) ?? "medium", type: str(d.type) ?? "alert" };
    })
    .filter((a): a is JobAlert => a !== null);
}

const SEVERITY_RANK: Record<string, number> = { critical: 4, error: 4, high: 3, medium: 2, warning: 2, low: 1, info: 0 };

/**
 * One entry per message: get_job_alerts repeats the diagnostics' warnings,
 * and repeats itself on every poll. The most severe report of a message wins.
 */
export function uniqueAlerts(items: readonly { severity: string; message: string; type?: string }[]): { severity: string; message: string; type: string }[] {
  const byMessage = new Map<string, { severity: string; message: string; type: string }>();
  for (const item of items) {
    const key = item.message.trim().replace(/\s+/g, " ").toLowerCase();
    const seen = byMessage.get(key);
    const rank = SEVERITY_RANK[item.severity.toLowerCase()] ?? 2;
    if (!seen || rank > (SEVERITY_RANK[seen.severity.toLowerCase()] ?? 2)) {
      byMessage.set(key, { severity: item.severity, message: item.message.trim(), type: item.type ?? seen?.type ?? "alert" });
    }
  }
  return [...byMessage.values()];
}

/** The line of a job error worth reading first: a traceback's last line, after what exited. */
export function errorSummary(error: string): string {
  const lines = error.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  if (lines.length <= 1 || !lines.some((line) => line.includes("Traceback"))) return lines[0] ?? "";
  const head = lines[0].split(/\.?\s*Last stderr:/)[0].replace(/Traceback.*$/, "").trim();
  const last = lines[lines.length - 1];
  return head ? `${head}: ${last}` : last;
}

/** REINVENT 4 likelihood columns that get_top_candidates reports as if they were components. */
const NOT_COMPONENTS = new Set(["agent", "prior", "target", "score"]);

/** get_top_candidates: ranked unique molecules with their component scores. */
export function parseCandidates(result: unknown): Candidate[] {
  return list(obj(result)?.candidates)
    .map((raw): Candidate | null => {
      const d = obj(raw);
      const smiles = str(d?.smiles);
      if (!d || !smiles) return null;
      const entries = Object.entries(obj(d.components) ?? {});
      // REINVENT 4 names the total "Score", so it arrives among the components.
      const score = entries.find(([name]) => name.toLowerCase() === "score");
      const names = new Set(entries.map(([name]) => name));
      // "std (<component>)" and the like describe a component rather than score.
      const detail = (name: string) => names.has(/^.+?\s*\((.+)\)$/.exec(name)?.[1] ?? "");
      return {
        smiles,
        total: num(d.total_score) ?? num(obj(score?.[1])?.transformed) ?? num(obj(score?.[1])?.raw),
        components: entries
          .filter(([name]) => !NOT_COMPONENTS.has(name.toLowerCase()) && !detail(name))
          .map(([name, c]) => {
            const e = obj(c);
            return { name, transformed: num(e?.transformed), raw: num(e?.raw) };
          }),
      };
    })
    .filter((c): c is Candidate => c !== null);
}

/** A top candidate's scores by the results CSV's column names. */
export function candidateScores(c: Candidate): Record<string, number> {
  const out: Record<string, number> = {};
  if (c.total !== null) out.Score = c.total;
  for (const { name, transformed, raw } of c.components) {
    if (transformed !== null) out[name] = transformed;
    if (raw !== null) out[`${name} (raw)`] = raw;
  }
  return out;
}

/**
 * The job's results CSV as a file name the gateway serves
 * (/api/jobs/<id>/files/<name>); REINVENT writes results_1.csv for a single stage.
 */
export function resultsFile(job: Pick<JobSummary, "csvPath">): string {
  const name = job.csvPath?.split(/[\\/]/).pop() ?? "";
  return /^[\w.-]+\.csv$/.test(name) ? name : "results_1.csv";
}

/** Whether a job can still change, so the panel keeps polling. */
export const isActive = (state: JobState) => state === "running" || state === "queued";

export function stateTone(state: JobState): Tone {
  if (state === "completed") return "ok";
  if (state === "failed") return "bad";
  if (state === "running" || state === "queued" || state === "stopped") return "warn";
  return "muted";
}

export function severityTone(severity: string): Tone {
  const s = severity.toLowerCase();
  if (s === "critical" || s === "high" || s === "error") return "bad";
  if (s === "low" || s === "info") return "info";
  return "warn";
}

export function healthTone(status: string): Tone {
  if (status === "healthy") return "ok";
  if (status === "failing") return "bad";
  if (status === "degraded") return "warn";
  return "muted";
}

export function convergenceTone(status: string | null): Tone {
  if (status === "improving" || status === "completed") return "ok";
  if (status === "plateau" || status === "stable") return "warn";
  if (status === "diverging") return "bad";
  return "info";
}

/** The CSS colour of a tone; badges mix it into a tinted background. */
export function toneColor(tone: Tone): string {
  switch (tone) {
    case "ok":
      return "var(--color-ok)";
    case "bad":
      return "var(--color-accent)";
    case "info":
      return "var(--color-structure)";
    case "warn":
      return "var(--color-warn)";
    default:
      return "var(--text-muted)";
  }
}

/** Fraction of the steps done, 0..1, from the trajectory or the diagnostics. */
export function progress(job: JobSummary | JobStatus): number | null {
  const status = job as Partial<JobStatus>;
  if (job.state === "completed") return 1;
  if (status.currentStep != null && job.maxSteps) return Math.min(1, status.currentStep / job.maxSteps);
  if (status.diagnostics?.progressPct != null) return Math.min(1, status.diagnostics.progressPct / 100);
  return null;
}

export function jobTitle(job: JobSummary): string {
  if (job.type === "transfer_learning") return "Transfer learning";
  if (!job.ttpId) return `Job ${job.id}`;
  return job.ttpVersion != null ? `${job.ttpId} v${job.ttpVersion}` : job.ttpId;
}

export function typeLabel(type: string): string {
  return type === "transfer_learning" ? "transfer learning" : type === "optimization" ? "RL optimization" : type.replace(/_/g, " ");
}

/** "12 Oct 14:02". */
export function shortDate(iso: string | null): string {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString([], { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}

/** Milliseconds a finished job ran for, when both ends are known. */
export function runtimeMs(job: JobSummary): number | null {
  if (!job.startedAt || !job.finishedAt) return null;
  const ms = new Date(job.finishedAt).getTime() - new Date(job.startedAt).getTime();
  return Number.isFinite(ms) && ms >= 0 ? ms : null;
}

export function fmt(value: number | null | undefined, digits = 3): string {
  return value == null ? "—" : value.toFixed(digits);
}

export function pct(value: number | null | undefined): string {
  return value == null ? "—" : `${Math.round(value * 100)}%`;
}

export function int(value: number | null | undefined): string {
  return value == null ? "—" : Math.round(value).toLocaleString();
}

// -- Resume ------------------------------------------------------------------

/** Learning settings of a job, read back from the config.toml it ran with. */
export interface RunOverrides {
  batchSize: number | null;
  sigma: number | null;
  learningRate: number | null;
  /** Mol2Mol runs read their seeds from a smiles_file and load a mol2mol prior. */
  mol2mol?: boolean;
}

export function parseRunConfig(toml: string): RunOverrides {
  const value = (key: string) => {
    const match = toml.match(new RegExp(`^\\s*${key}\\s*=\\s*([-+0-9.eE]+)\\s*(?:#.*)?$`, "m"));
    const n = match ? Number(match[1]) : NaN;
    return Number.isFinite(n) ? n : null;
  };
  const prior = toml.match(/^\s*prior_file\s*=\s*"([^"]*)"/m)?.[1] ?? "";
  return {
    batchSize: value("batch_size"),
    sigma: value("sigma"),
    learningRate: value("rate"),
    mol2mol: /mol2mol|m2m/i.test(prior) || /^\s*smiles_file\s*=/m.test(toml),
  };
}

export function canResume(job: JobSummary): boolean {
  return (
    job.type === "optimization" &&
    (job.state === "stopped" || job.state === "failed" || job.state === "completed") &&
    job.hasCheckpoint &&
    !!job.ttpId &&
    !!job.checkpointPath
  );
}

export function isMol2Mol(job: Pick<JobSummary, "model">, config?: RunOverrides | null): boolean {
  return !!config?.mol2mol || /mol2mol|m2m/i.test(job.model ?? "");
}

/** Steps the job finished: get_optimization_status reports the next step (last CSV step + 1). */
export function stepsDone(job: JobSummary | JobStatus): number {
  if (job.state === "completed" && job.maxSteps) return job.maxSteps;
  const next = (job as Partial<JobStatus>).currentStep;
  return next == null ? 0 : Math.max(0, next - 1);
}

/** Steps a resumed run still has to do, so it ends where the original would have. */
export function remainingSteps(job: JobSummary | JobStatus): number {
  const max = job.maxSteps ?? 300;
  return Math.max(1, max - stepsDone(job));
}

/** Default length of a continuation: what is left of a stopped run, else 100 more. */
export function continueSteps(job: JobSummary | JobStatus): number {
  const left = (job.maxSteps ?? 0) - stepsDone(job);
  return job.state !== "completed" && left > 0 ? left : 100;
}

export interface ContinueOptions {
  steps?: number;
  config?: RunOverrides | null;
  /** The TTP at the job's version, sent inline so a newer saved version does not change the scoring. */
  ttp?: unknown;
  seeds?: string[];
}

/** Arguments of run_optimization that continue *job* from its checkpoint. */
export function resumeArguments(job: JobSummary | JobStatus, { steps, config, ttp, seeds }: ContinueOptions = {}): Record<string, unknown> {
  const args: Record<string, unknown> = ttp ? { ttp } : { ttp_id: job.ttpId };
  args.max_steps = steps ?? remainingSteps(job);
  args.resume_from = job.checkpointPath;
  if (job.model) args.model = job.model;
  if (config?.batchSize != null) args.batch_size = config.batchSize;
  if (config?.sigma != null) args.sigma = config.sigma;
  if (config?.learningRate != null) args.learning_rate = config.learningRate;
  if (seeds?.length && isMol2Mol(job, config)) args.seed_smiles = seeds;
  return args;
}

/** What *steps* more cost: "6,400 molecules · ~8 min at the last pace". */
export function continueCost(job: JobSummary | JobStatus, steps: number, batchSize: number | null): string {
  const parts: string[] = [];
  if (batchSize) parts.push(`${(steps * batchSize).toLocaleString()} molecules`);
  const ms = runtimeMs(job);
  const done = stepsDone(job);
  if (ms && done > 0) {
    const minutes = Math.max(1, Math.round((ms / done) * steps / 60_000));
    parts.push(minutes >= 90 ? `~${(minutes / 60).toFixed(1)} h at the last pace` : `~${minutes} min at the last pace`);
  }
  return parts.join(" · ");
}

/** The job a run_optimization call started. */
export const startedJobId = (result: unknown): string | null => str(obj(result)?.job_id);

// -- Prompts for the agent -----------------------------------------------------

export function reviewPrompt(job: JobSummary | JobStatus): string {
  const status = (job as Partial<JobStatus>).diagnostics?.convergence;
  return (
    `Review REINVENT job ${job.id}${job.ttpId ? ` (TTP ${job.ttpId})` : ""}` +
    `${status ? `, which is ${status}` : ""}. Check get_optimization_status and get_job_alerts, ` +
    "explain the diagnostics and recommend whether to keep it running, or stop it, adjust the TTP and resume from the checkpoint."
  );
}

export function resumePrompt(job: JobSummary): string {
  return (
    `Resume REINVENT job ${job.id}${job.ttpId ? ` (TTP ${job.ttpId})` : ""} from its checkpoint` +
    `${job.checkpointPath ? ` ${job.checkpointPath}` : ""} with run_optimization. ` +
    "If the diagnostics call for it, adjust the TTP first and tell me what you changed."
  );
}

// -- Chart geometry ------------------------------------------------------------

/** The y range to plot: 0..1 for normal scores, widened to fit anything outside. */
export function scoreDomain(values: number[]): [number, number] {
  const finite = values.filter(Number.isFinite);
  if (!finite.length) return [0, 1];
  const lo = Math.min(0, ...finite);
  const hi = Math.max(1, ...finite);
  return [lo, hi === lo ? lo + 1 : hi];
}

/** Evenly spaced tick values from lo to hi, inclusive. */
export function ticks(lo: number, hi: number, count = 5): number[] {
  if (count < 2 || hi <= lo) return [lo];
  const step = (hi - lo) / (count - 1);
  return Array.from({ length: count }, (_, i) => Number((lo + i * step).toFixed(6)));
}

export interface Scale {
  (value: number): number;
}

export function linear(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0 || 1;
  return (value) => r0 + ((value - d0) / span) * (r1 - r0);
}

/** An SVG path through the points, in order; empty for no points. */
export function linePath(points: { x: number; y: number }[], x: Scale, y: Scale): string {
  return points.map((p, i) => `${i ? "L" : "M"}${x(p.x).toFixed(1)},${y(p.y).toFixed(1)}`).join("");
}

/** The best score reached so far at each step: a never-decreasing line. */
/**
 * The per-step scores to chart. REINVENT only reports score_trajectory on some
 * runs; the total score component carries the same means, without a per-step best.
 */
export function trajectoryPoints(status: JobStatus): { points: TrajectoryPoint[]; from: string | null } {
  if (status.trajectory.length) return { points: status.trajectory, from: null };
  for (const key of ["Score", "total_score"]) {
    const series = status.componentTrajectories[key];
    if (series?.length) return { points: series.map((p) => ({ step: p.step, mean: p.mean, max: null, molecules: null })), from: key };
  }
  return { points: [], from: null };
}

export function runningBest(points: TrajectoryPoint[]): { x: number; y: number }[] {
  let best = -Infinity;
  return points.map((p) => {
    best = Math.max(best, p.max ?? p.mean);
    return { x: p.step, y: best };
  });
}
