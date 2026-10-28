// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { useJobOwners, useRunsKnown, useSessionRuns } from "../../hooks/useSessions";
import type { Theme } from "../../hooks/useTheme";
import { callTool, jobFileUrl } from "../../lib/api";
import { conversationJobs, errorSummary, isActive, jobTitle, parseJobs, resultsFile, toolError, type JobSummary } from "../../lib/jobs";
import { rowMolecule } from "../../lib/jobStructures";
import type { MoleculeContext } from "../../lib/moleculePage";
import { countMask, moleculesOf, selectionMask, spreads, stepStats, topByWeight, TOTAL, type Endpoint, type Molecules, type Ranges, type StepStats } from "../../lib/runAnalysis";
import { parseRunResults, type RunTable } from "../../lib/runResults";
import { loadTtp } from "../../lib/scoreWhy";
import { objectivesOf } from "../../lib/ttp";
import { focusRun, useFocusedRun } from "../../lib/workbenchBus";
import { SelectionTable } from "./SelectionTable";
import { endpointColours } from "./plotUtils";
import { Card, PickMenu } from "./shared";
import { TrajectoryPlot } from "./TrajectoryPlot";
import { DENSITY, STEP, UmapPlot } from "./UmapPlot";
import { useUmap } from "./useUmap";
import { ViolinPlot } from "./ViolinPlot";

interface Props {
  sessionId: string | null;
  theme: Theme;
  onOpenMolecule?: (smiles: string, context?: MoleculeContext) => void;
}

/** A live run's CSV grows by one batch per step; read again this often, as Results does. */
const REFRESH_MS = 60_000;

interface Loaded {
  id: string;
  table: RunTable | null;
  molecules: Molecules | null;
  stats: Map<string, StepStats[]>;
  error: string | null;
  /** Read after the run ended: it will not change. */
  final: boolean;
}

/** The last run read, so coming back to the view shows it at once. */
let lastLoaded: Loaded | null = null;

interface Picks {
  jobId: string | null;
  trajectory: Set<string> | null;
  violins: string[] | null;
  ranges: Ranges;
  region: Set<string> | null;
}
const noPicks = (jobId: string | null): Picks => ({ jobId, trajectory: null, violins: null, ranges: {}, region: null });

const errorText = (error: unknown) => (error instanceof Error ? error.message : String(error));

/**
 * A run's results at a glance: the score trajectory, the distribution of each
 * endpoint and a map of chemical space, linked by one selection (ranges on the
 * violins AND a region of the map) that the table below lists.
 */
export function RunAnalysis({ sessionId, theme, onOpenMolecule }: Props) {
  const [jobs, setJobs] = useState<JobSummary[] | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);
  const focused = useFocusedRun();
  const sessionRuns = useSessionRuns(sessionId);
  const owners = useJobOwners();
  const known = useRunsKnown();

  const runs = useMemo(() => (jobs ?? []).filter((j) => j.type === "optimization"), [jobs]);
  const own = useMemo(() => (known ? conversationJobs(runs, sessionRuns, focused, owners, sessionId).own : runs), [known, runs, sessionRuns, focused, owners, sessionId]);
  const started = (j: JobSummary) => j.state !== "failed" && j.state !== "queued";
  const job = runs.find((j) => j.id === focused) ?? own.find(started) ?? own[0] ?? runs.find(started) ?? runs[0] ?? null;
  const runId = job?.id ?? null;
  const running = job ? isActive(job.state) : false;
  const listed = own.length ? own : runs;
  const options = job && !listed.includes(job) ? [job, ...listed] : listed;

  useEffect(() => {
    let cancelled = false;
    callTool("list_jobs")
      .then(({ result }) => {
        if (cancelled) return;
        const error = toolError(result);
        setListError(error);
        if (!error) setJobs(parseJobs(result));
      })
      .catch((error: unknown) => {
        if (!cancelled) setListError(errorText(error));
      });
    return () => {
      cancelled = true;
    };
  }, [sessionId, tick]);

  const [loaded, setLoaded] = useState<Loaded | null>(lastLoaded);
  const current = loaded && loaded.id === runId ? loaded : null;
  const file = job ? resultsFile(job) : null;
  const queued = job?.state === "queued";
  useEffect(() => {
    if (!runId || !file || queued) return;
    if (tick === 0 && lastLoaded?.id === runId && lastLoaded.final) return;
    let cancelled = false;
    const final = !running;
    fetch(jobFileUrl(runId, file))
      .then(async (response) => {
        if (cancelled) return;
        if (!response.ok) {
          setLoaded({ id: runId, table: null, molecules: null, stats: new Map(), final: false, error: response.status === 404 ? "No results yet." : `HTTP ${response.status}` });
          return;
        }
        const text = await response.text();
        if (cancelled) return;
        const table = parseRunResults(text);
        const molecules = moleculesOf(table);
        const stats = stepStats(
          table,
          molecules.endpoints.filter((e) => e.kind !== "raw"),
        );
        const next = { id: runId, table, molecules, stats, error: null, final };
        lastLoaded = next;
        setLoaded(next);
      })
      .catch((error: unknown) => {
        if (!cancelled) setLoaded({ id: runId, table: null, molecules: null, stats: new Map(), final: false, error: errorText(error) });
      });
    return () => {
      cancelled = true;
    };
  }, [runId, file, queued, running, tick]);

  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => {
      if (!document.hidden) setTick((t) => t + 1);
    }, REFRESH_MS);
    return () => window.clearInterval(timer);
  }, [running]);

  // Weights of the TTP's objectives, to pick the few lines and violins shown first.
  const ttpKey = job?.ttpId ? `${job.ttpId}@${job.ttpVersion ?? ""}` : null;
  const [weights, setWeights] = useState<{ key: string; list: { name: string; weight: number }[] } | null>(null);
  useEffect(() => {
    if (!ttpKey || !job?.ttpId) return;
    let cancelled = false;
    loadTtp(job.ttpId, job.ttpVersion).then((ttp) => {
      if (cancelled) return;
      const list = ttp ? [...objectivesOf(ttp, "pre_scoring"), ...objectivesOf(ttp, "scoring")].map((o) => ({ name: o.name, weight: o.weight })) : [];
      setWeights({ key: ttpKey, list });
    });
    return () => {
      cancelled = true;
    };
  }, [ttpKey, job?.ttpId, job?.ttpVersion]);

  const table = current?.table ?? null;
  const molecules = current?.molecules ?? null;
  const endpoints = useMemo(() => molecules?.endpoints ?? [], [molecules]);
  const colours = useMemo(() => endpointColours(endpoints), [endpoints]);
  const components = useMemo(() => endpoints.filter((e) => e.kind === "component").map((e) => e.key), [endpoints]);
  const weightList = weights && weights.key === ttpKey ? weights.list : null;
  const spread = useMemo(() => (molecules ? spreads(molecules.values) : undefined), [molecules]);
  const top = useMemo(() => topByWeight(components, weightList ?? [], 3, spread), [components, weightList, spread]);
  const defaults = useMemo(() => [...(endpoints.some((e) => e.kind === "total") ? [TOTAL] : []), ...top], [endpoints, top]);

  const [picks, setPicks] = useState<Picks>(() => noPicks(runId));
  if (picks.jobId !== runId) setPicks(noPicks(runId));
  const trajectoryShown = useMemo(() => picks.trajectory ?? new Set(defaults), [picks.trajectory, defaults]);
  const violins = picks.violins ?? defaults;
  const mask = useMemo(() => (molecules ? selectionMask(molecules, picks.ranges, picks.region) : null), [molecules, picks.ranges, picks.region]);
  const selected = countMask(mask);
  const filtered = mask !== null;

  const [colourBy, setColourBy] = useState<string>(TOTAL);
  const [mode, setMode] = useState<"box" | "lasso">("lasso");
  const [hoverSmiles, setHoverSmiles] = useState<string | null>(null);
  const hovered = hoverSmiles !== null && molecules ? (molecules.index.get(hoverSmiles) ?? null) : null;
  const onHover = useCallback((i: number | null) => setHoverSmiles(i === null || !molecules ? null : molecules.smiles[i]), [molecules]);

  const { layout, working } = useUmap(runId, molecules);

  const onRange = useCallback(
    (key: string, range: [number, number] | null) =>
      setPicks((p) => {
        const ranges = { ...p.ranges };
        if (range) ranges[key] = range;
        else delete ranges[key];
        return { ...p, ranges };
      }),
    [],
  );
  const onRegion = useCallback(
    (smiles: string[] | null, add: boolean) =>
      setPicks((p) => {
        if (!smiles || (!smiles.length && !add)) return { ...p, region: null };
        const region = add && p.region ? new Set([...p.region, ...smiles]) : new Set(smiles);
        return { ...p, region };
      }),
    [],
  );
  const clear = () => setPicks((p) => ({ ...p, ranges: {}, region: null }));
  const toggle = (set: ReadonlySet<string>, key: string) => {
    const next = new Set(set);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    return next;
  };
  const open = useCallback(
    (i: number) => {
      if (!molecules || !table || !job || !onOpenMolecule) return;
      onOpenMolecule(molecules.smiles[i], rowMolecule(table, molecules.rows[i], job.id, job));
    },
    [molecules, table, job, onOpenMolecule],
  );

  useEffect(() => {
    if (!filtered) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !(event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement)) setPicks((p) => ({ ...p, ranges: {}, region: null }));
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [filtered]);

  if (jobs === null) {
    return <Centered>{listError ? <p style={{ color: "var(--color-accent)" }}>Could not load the runs: {listError}</p> : <p className="text-faint">Loading runs…</p>}</Centered>;
  }
  if (!job) {
    return (
      <Centered>
        <p className="text-body font-medium">No REINVENT runs yet</p>
        <p className="text-muted max-w-sm">A run's trajectory, distributions and chemical space show up here.</p>
      </Centered>
    );
  }

  const chip = filtered && (
    <span className="bg-inset inline-flex items-center gap-1.5 rounded-full py-0.5 pr-1 pl-2 tabular-nums">
      <span className="text-body font-medium">{selected.toLocaleString()}</span>
      <span className="text-faint">selected</span>
      <button type="button" onClick={clear} title="Clear the selection (Esc)" className="text-muted hover:text-body hover:bg-raised cursor-pointer rounded-full px-1.5">
        clear
      </button>
    </span>
  );

  return (
    <div className="@container flex h-full min-h-0 flex-col text-[11px]">
      <div className="border-subtle flex shrink-0 flex-wrap items-center gap-x-2 gap-y-1 border-b px-3 py-1.5">
        <select
          value={job.id}
          onChange={(event) => focusRun(event.target.value)}
          aria-label="Run"
          className="bg-sunken border-subtle max-w-64 min-w-0 cursor-pointer truncate rounded-md border px-1.5 py-1 text-[11px] outline-none"
        >
          {options.map((j) => (
            <option key={j.id} value={j.id}>
              {jobTitle(j)} · {j.id} · {j.state}
            </option>
          ))}
        </select>
        {molecules && (
          <span className="text-faint tabular-nums">
            {molecules.n.toLocaleString()} molecules · step {molecules.lastStep}
            {job.maxSteps ? `/${job.maxSteps}` : ""}
          </span>
        )}
        {running && <span className="size-1.5 animate-pulse rounded-full" style={{ background: "var(--color-ok)" }} title="Live: refreshed every minute" />}
        <span className="ml-auto flex items-center gap-1">
          {chip}
          <button type="button" onClick={() => setTick((t) => t + 1)} title="Reload" aria-label="Reload" className="text-faint hover:text-body cursor-pointer p-0.5">
            <RefreshIcon />
          </button>
        </span>
      </div>
      {job.state === "failed" && job.error && (
        <p className="border-subtle text-muted shrink-0 truncate border-b px-3 py-1" style={{ boxShadow: "inset 2px 0 0 var(--color-accent)" }} title={job.error}>
          {errorSummary(job.error)}
        </p>
      )}

      {queued ? (
        <Centered>
          <p className="text-faint">Queued: nothing to analyse yet.</p>
        </Centered>
      ) : !current ? (
        <Centered>
          <p className="text-faint">Loading results…</p>
        </Centered>
      ) : !molecules || !table ? (
        <Centered>
          <p className="text-muted max-w-sm">{job.state === "failed" ? "The run wrote no results." : (current.error ?? "No results.")}</p>
        </Centered>
      ) : molecules.n === 0 ? (
        <Centered>
          <p className="text-muted">No valid molecules yet.</p>
        </Centered>
      ) : (
        <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3">
          <Card title="Score trajectory">
            <TrajectoryPlot
              endpoints={endpoints.filter((e) => e.kind !== "raw")}
              stats={current.stats}
              shown={trajectoryShown}
              onToggle={(key) => setPicks((p) => ({ ...p, trajectory: toggle(p.trajectory ?? new Set(defaults), key) }))}
              colours={colours}
              live={running}
            />
          </Card>
          <div className="grid gap-3 @3xl:grid-cols-2">
            <Card
              title="Distribution"
              actions={
                <PickMenu
                  label="Endpoints"
                  items={endpoints.map((e) => ({ key: e.key, label: e.label, colour: colours.get(e.key), group: e.kind === "raw" ? "Raw values" : "Scores" }))}
                  checked={new Set(violins)}
                  onToggle={(key) =>
                    setPicks((p) => {
                      const list = p.violins ?? defaults;
                      return { ...p, violins: list.includes(key) ? list.filter((k) => k !== key) : [...list, key] };
                    })
                  }
                />
              }
            >
              <ViolinPlot molecules={molecules} endpoints={endpoints} shown={violins} mask={mask} ranges={picks.ranges} onRange={onRange} colours={colours} hovered={hovered} />
            </Card>
            <Card
              title="Chemical space"
              actions={
                <>
                  <select
                    value={colourBy}
                    onChange={(event) => setColourBy(event.target.value)}
                    aria-label="Colour by"
                    title="Colour by"
                    className="bg-sunken border-subtle max-w-36 cursor-pointer truncate rounded border px-1 py-0.5 text-[11px] outline-none"
                  >
                    {endpoints.map((e: Endpoint) => (
                      <option key={e.key} value={e.key}>
                        {e.label}
                      </option>
                    ))}
                    <option value={STEP}>Step</option>
                    <option value={DENSITY}>Density</option>
                  </select>
                  <span role="group" aria-label="Selection tool" className="bg-sunken flex rounded p-px">
                    <ModeButton on={mode === "lasso"} onClick={() => setMode("lasso")} title="Lasso (shift adds)">
                      <path d="M8 3c3.3 0 5.5 1.5 5.5 3.5S11.3 10 8 10 2.5 8.5 2.5 6.5 4.7 3 8 3Z M4.5 9.5c-.8 1.2-.5 2.6.8 3" />
                    </ModeButton>
                    <ModeButton on={mode === "box"} onClick={() => setMode("box")} title="Box (shift adds)">
                      <rect x={3} y={4} width={10} height={8} rx={1} strokeDasharray="2 1.5" />
                    </ModeButton>
                  </span>
                </>
              }
            >
              <UmapPlot
                molecules={molecules}
                layout={layout}
                working={working}
                colourBy={endpoints.some((e) => e.key === colourBy) || colourBy === STEP || colourBy === DENSITY ? colourBy : DENSITY}
                mask={mask}
                mode={mode}
                onRegion={onRegion}
                hovered={hovered}
                onHover={onHover}
                onOpen={open}
                theme={theme}
              />
            </Card>
          </div>
          <Card title={filtered ? "Selection" : "Molecules"} actions={chip || undefined}>
            <SelectionTable
              molecules={molecules}
              mask={mask}
              columns={endpoints.filter((e) => top.includes(e.key))}
              colours={colours}
              hovered={hovered}
              onHover={onHover}
              onOpen={open}
              theme={theme}
            />
          </Card>
        </div>
      )}
    </div>
  );
}

function ModeButton({ on, onClick, title, children }: { on: boolean; onClick: () => void; title: string; children: ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={on}
      title={title}
      aria-label={title}
      className={`grid size-5 cursor-pointer place-items-center rounded-sm ${on ? "bg-raised text-body shadow-sm" : "text-faint hover:text-muted"}`}
    >
      <svg width={14} height={14} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth={1.3} strokeLinecap="round" aria-hidden="true">
        {children}
      </svg>
    </button>
  );
}

function Centered({ children }: { children: ReactNode }) {
  return <div className="flex flex-1 flex-col items-center justify-center gap-2 p-6 text-center text-xs">{children}</div>;
}

function RefreshIcon() {
  return (
    <svg width={12} height={12} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth={1.6} aria-hidden="true">
      <path d="M13.5 8a5.5 5.5 0 1 1-1.6-3.9M13.5 2.5v3h-3" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
