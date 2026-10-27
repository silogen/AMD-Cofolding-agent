// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { lazy, Suspense, useEffect, useMemo, useState, type ReactNode } from "react";
import { callTool, download } from "../../lib/api";
import { parseJobs, shortDate, stateTone, toneColor, type JobSummary } from "../../lib/jobs";
import {
  applyPrompt,
  asTtp,
  asTtpList,
  changes,
  chatPrompt,
  createdTimes,
  curveLabel,
  filterTtps,
  hasCofolding,
  isDirty,
  latestTtp,
  loadDraft,
  makeDraft,
  objectivesOf,
  parseResult,
  patchObjective,
  payloadError,
  removeObjective,
  saveDraft,
  sortTtps,
  sourceLabel,
  targetRange,
  targetSummary,
  ttpCreated,
  ttpFileName,
  ttpGeneration,
  ttpUpdates,
  versionDiff,
  isSameVersion,
  weightRange,
  weightShares,
  type Objective,
  type ToolEvent,
  type Ttp,
  type TtpDraft,
  type TtpSection,
  type TtpSummary,
} from "../../lib/ttp";
import { modelLabel } from "../../lib/ttpRun";
import { onRunsChanged, openRun } from "../../lib/workbenchBus";
import { RunCard } from "../TtpBuilder/RunFields";
import { ObjectiveCard } from "./ObjectiveCard";
import { TransformCurve } from "./TransformCurve";

const TtpBuilder = lazy(() => import("../TtpBuilder/TtpBuilder").then((module) => ({ default: module.TtpBuilder })));

interface Props {
  /** Edits are kept per session, in localStorage. */
  sessionId: string;
  /** The session's tool calls, from `toolEvents(artifacts.steps)`; picks the TTP the session uses. */
  events?: ToolEvent[];
  /** Put a prompt in the composer: "Use in chat", and "Apply" to have the agent adopt edits. */
  onAsk: (prompt: string) => void;
  /** Show this stored TTP instead of the session's. */
  ttpId?: string;
}

/** Stored Target Product Profiles: browse, read, reuse, and edit weights and windows. */
export function TtpPanel(props: Props) {
  return <TtpEditor key={props.sessionId} {...props} />;
}

type Remote = { id: string; ttp: Ttp | null; error: string | null };
type SaveState = { state: "idle" } | { state: "busy" } | { state: "error"; message: string } | { state: "saved"; version: number };
type ListState = { state: "loading" } | { state: "error"; message: string } | { state: "ready"; items: TtpSummary[] };

const message = (err: unknown) => (err instanceof Error ? err.message : String(err));

function TtpEditor({ sessionId, events = [], onAsk, ttpId }: Props) {
  const fromSession = useMemo(() => latestTtp(events), [events]);
  const [stored, setStored] = useState<TtpDraft | null>(() => loadDraft(sessionId));
  const [list, setList] = useState<ListState>({ state: "loading" });
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [created, setCreated] = useState(() => createdTimes());
  const [choice, setChoice] = useState<string | null>(null);
  const [remote, setRemote] = useState<Remote | null>(null);
  const [reload, setReload] = useState(0);
  const [save, setSave] = useState<SaveState>({ state: "idle" });
  const [revision, setRevision] = useState(0);
  const [query, setQuery] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [builder, setBuilder] = useState<{ initial?: Ttp } | null>(null);

  const [jobsReload, setJobsReload] = useState(0);

  useEffect(() => {
    let cancelled = false;
    callTool("list_ttps")
      .then((response) => !cancelled && setList({ state: "ready", items: asTtpList(response.result) }))
      .catch((err: unknown) => !cancelled && setList({ state: "error", message: message(err) }));
    return () => {
      cancelled = true;
    };
  }, [reload]);

  useEffect(() => {
    let cancelled = false;
    callTool("list_jobs")
      .then((response) => !cancelled && setJobs(parseJobs(response.result)))
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [reload, jobsReload]);

  useEffect(() => onRunsChanged(() => setJobsReload((n) => n + 1)), []);

  const activity = useMemo(() => {
    const latest: Record<string, string> = { ...created };
    for (const job of jobs) if (job.ttpId && job.createdAt && job.createdAt > (latest[job.ttpId] ?? "")) latest[job.ttpId] = job.createdAt;
    return latest;
  }, [created, jobs]);
  const items = useMemo(() => sortTtps(list.state === "ready" ? list.items : [], activity), [list, activity]);

  const selectedId = ttpId ?? choice ?? fromSession.id ?? stored?.base.id ?? items[0]?.id ?? null;
  const sessionTtp = fromSession.ttp && fromSession.ttp.id === selectedId ? fromSession.ttp : null;
  const sessionVersion = sessionTtp?.version ?? 0;

  useEffect(() => {
    if (!selectedId) return;
    let cancelled = false;
    callTool("get_ttp", { ttp_id: selectedId })
      .then((response) => {
        if (cancelled) return;
        const error = payloadError(response.result);
        setRemote({ id: selectedId, ttp: error ? null : asTtp(parseResult(response.result)), error });
      })
      .catch((err: unknown) => !cancelled && setRemote({ id: selectedId, ttp: null, error: message(err) }));
    return () => {
      cancelled = true;
    };
  }, [selectedId, reload, sessionVersion]);

  const remoteTtp = remote && remote.id === selectedId ? remote.ttp : null;
  const loading = Boolean(selectedId) && remote?.id !== selectedId;
  const onServer = remoteTtp !== null;
  const candidates = [remoteTtp, sessionTtp, stored && stored.base.id === selectedId ? stored.base : null].filter((t): t is Ttp => t !== null);
  const base = candidates.sort((a, b) => b.version - a.version)[0] ?? null;

  const refresh = () => setReload((n) => n + 1);
  const select = (id: string) => {
    setChoice(id);
    setSave({ state: "idle" });
  };
  const afterCreate = (id: string) => {
    setBuilder(null);
    setCreated(createdTimes());
    setChoice(id);
    refresh();
  };

  const shell = (children: ReactNode) => (
    <div className="@container flex min-h-0 flex-1 flex-col">
      <div className="border-subtle flex shrink-0 items-center gap-2 border-b px-3 py-2">
        <div className="min-w-0 flex-1">
          <h2 className="text-xs font-semibold">Target product profiles</h2>
          <p className="text-faint text-[10px]">
            {list.state === "ready" ? `${items.length} stored · most recently used first` : list.state === "loading" ? "Loading…" : "TTP store unavailable"}
          </p>
        </div>
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search TTPs"
          aria-label="Search TTPs"
          className="bg-sunken border-subtle focus:border-strong w-36 rounded-md border px-2 py-1 text-[11px] outline-none @lg:w-48"
        />
        <button type="button" onClick={refresh} title="Reload the TTPs and runs" className="bg-inset text-muted hover:text-body shrink-0 cursor-pointer rounded-md px-2 py-1 text-[11px] font-medium touch:min-h-11">
          Refresh
        </button>
        <button
          type="button"
          onClick={() => setBuilder({})}
          title="Define a new target product profile"
          className="bg-inset text-body hover:bg-sunken shrink-0 cursor-pointer rounded-md px-2 py-1 text-[11px] font-semibold touch:min-h-11"
        >
          + New TTP
        </button>
      </div>
      <div className="flex min-h-0 flex-1">
        <TtpList items={filterTtps(items, query)} total={items.length} jobs={jobs} selectedId={selectedId} sessionId={fromSession.id} onSelect={select} />
        <div className="flex min-h-0 min-w-0 flex-1 flex-col">
          <div className="border-subtle shrink-0 border-b px-3 py-1.5 @xl:hidden">
            <select
              value={selectedId ?? ""}
              onChange={(event) => select(event.target.value)}
              aria-label="TTP"
              className="bg-sunken border-subtle w-full rounded-md border px-1.5 py-1 text-[11px]"
            >
              {selectedId && !items.some((t) => t.id === selectedId) && <option value={selectedId}>{selectedId}</option>}
              {filterTtps(items, query).map((t) => (
                <option key={t.id} value={t.id}>
                  {t.name} · v{t.version}
                </option>
              ))}
            </select>
          </div>
          {children}
        </div>
      </div>
      {builder && (
        <Suspense fallback={null}>
          <TtpBuilder
            initial={builder.initial}
            onClose={() => setBuilder(null)}
            onCreated={(ttp, request) => {
              afterCreate(ttp.id);
              onAsk(request);
            }}
          />
        </Suspense>
      )}
    </div>
  );

  if (!selectedId || (!base && !loading)) {
    return shell(
      <p className="text-muted px-6 py-10 text-center text-xs">
        {remote?.error && selectedId
          ? `Could not load TTP "${selectedId}": ${remote.error}`
          : list.state === "error"
            ? `Could not list TTPs: ${list.message}`
            : list.state === "loading"
              ? "Loading…"
              : "No TTP yet. Create one with + New TTP, or ask the agent to."}
      </p>,
    );
  }
  if (!base) return shell(<p className="text-muted px-6 py-10 text-center text-xs">Loading {selectedId}…</p>);

  const mine = stored && stored.base.id === base.id ? stored : null;
  const draft = mine?.draft ?? base;
  const dirty = isDirty(base, draft);
  const editing = editingId === base.id || dirty;
  const runs = jobs.filter((j) => j.ttpId === base.id);

  if (!editing) {
    return shell(
      <TtpReadView
        ttp={base}
        onServer={onServer}
        inSession={fromSession.id === base.id}
        runs={runs}
        onAsk={onAsk}
        onUse={() => onAsk(chatPrompt(base))}
        onDuplicate={() => setBuilder({ initial: base })}
        onDownload={() => download(ttpFileName(base), JSON.stringify(base, null, 2), "application/json")}
        onEdit={() => setEditingId(base.id)}
        saved={save.state === "saved" ? save.version : null}
      />,
    );
  }

  const outdated = dirty && mine !== null && mine.base.version < base.version;
  const diff = changes(base, draft);
  const shares = weightShares(draft);
  const range = weightRange(draft);

  const edit = (next: Ttp) => {
    const value = makeDraft(mine?.base ?? base, next, base);
    setStored(value);
    saveDraft(sessionId, value);
    setSave({ state: "idle" });
  };

  const discard = () => {
    setStored(null);
    saveDraft(sessionId, null);
    setRevision((n) => n + 1);
    setSave({ state: "idle" });
    setEditingId(null);
  };

  const rebase = () => {
    if (!mine) return;
    const value = makeDraft(base, mine.draft);
    setStored(value);
    saveDraft(sessionId, value);
  };

  const saveDirect = async () => {
    setSave({ state: "busy" });
    try {
      const response = await callTool("update_ttp", { ttp_id: base.id, updates: ttpUpdates(base, draft) });
      const error = payloadError(response.result);
      if (error) {
        setSave({ state: "error", message: error });
        return;
      }
      const payload = parseResult(response.result) as { ttp?: unknown } | null;
      const updated = asTtp(payload?.ttp);
      if (updated) setRemote({ id: base.id, ttp: updated, error: null });
      setStored(null);
      saveDraft(sessionId, null);
      setRevision((n) => n + 1);
      setEditingId(null);
      setSave({ state: "saved", version: updated?.version ?? base.version + 1 });
    } catch (err) {
      setSave({ state: "error", message: message(err) });
    }
  };

  const sections: { section: TtpSection; title: string }[] = [
    { section: "pre_scoring", title: "Pre-scoring" },
    { section: "scoring", title: "Objectives" },
  ];

  return shell(
    <>
      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3">
        <div className="flex flex-wrap items-center gap-2">
          <div className="min-w-0 flex-1">
            <h3 className="truncate text-sm font-semibold">Editing {base.name}</h3>
            <p className="text-faint text-[10px]">
              <span className="font-mono">{base.id}</span> · v{base.version} · edits are kept in this browser until saved or applied
            </p>
          </div>
          <label className="text-muted flex items-center gap-1 text-[11px]">
            Combine
            <select
              value={draft.scoring.aggregation ?? "geometric_mean"}
              onChange={(event) => edit({ ...draft, scoring: { ...draft.scoring, aggregation: event.target.value as "geometric_mean" | "arithmetic_mean" } })}
              className="bg-sunken border-subtle rounded-md border px-1 py-0.5 text-[11px]"
            >
              <option value="geometric_mean">geometric mean</option>
              <option value="arithmetic_mean">arithmetic mean</option>
            </select>
          </label>
        </div>

        {outdated && (
          <p className="bg-sunken border-subtle rounded-lg border px-3 py-2 text-[11px]">
            The stored TTP moved to v{base.version} since you started editing v{mine?.base.version}.{" "}
            <button type="button" onClick={rebase} className="cursor-pointer font-semibold underline">
              Keep my edits on v{base.version}
            </button>{" "}
            or{" "}
            <button type="button" onClick={discard} className="cursor-pointer font-semibold underline">
              discard them
            </button>
            .
          </p>
        )}

        {sections.map(({ section, title }) => {
          const objectives = objectivesOf(draft, section);
          if (section === "pre_scoring" && objectives.length === 0 && objectivesOf(base, section).length === 0) return null;
          return (
            <section key={section}>
              <h4 className="text-muted mb-1 flex gap-2 text-[11px] font-medium">
                <span>{title}</span>
                <span className="text-faint ml-auto">{objectives.length}</span>
              </h4>
              <ul className="space-y-1.5">
                {objectives.map((objective, index) => (
                  <ObjectiveCard
                    key={`${revision}:${objective.name}`}
                    objective={objective}
                    original={objectivesOf(base, section).find((o) => o.name === objective.name)}
                    share={shares.get(objective.name) ?? 0}
                    weights={range}
                    changed={diff.changed.includes(objective.name)}
                    onChange={(patch) => edit(patchObjective(draft, section, index, patch))}
                    onRemove={objectives.length > 1 || section === "pre_scoring" ? () => edit(removeObjective(draft, section, index)) : undefined}
                  />
                ))}
              </ul>
            </section>
          );
        })}

        {diff.removed.length > 0 && <p className="text-faint text-[11px]">Removed: {diff.removed.join(", ")}</p>}
      </div>

      <footer className="border-subtle flex shrink-0 flex-wrap items-center gap-2 border-t px-3 py-2 text-[11px]">
        <span className="text-muted min-w-0 flex-1 truncate">
          {save.state === "error" ? (
            <span style={{ color: "var(--color-accent)" }}>{save.message}</span>
          ) : dirty ? (
            `Edited: ${[...diff.changed, ...diff.removed.map((n) => `−${n}`)].join(", ") || "aggregation"}`
          ) : (
            "Move a weight, a threshold or drag one on a curve."
          )}
        </span>
        {!dirty && (
          <button type="button" onClick={() => setEditingId(null)} className={GHOST}>
            Done
          </button>
        )}
        {dirty && (
          <button type="button" onClick={discard} className={GHOST}>
            Discard
          </button>
        )}
        {onServer && (
          <button
            type="button"
            onClick={saveDirect}
            disabled={!dirty || save.state === "busy"}
            title="Write the edits to the TTP store with update_ttp"
            className="bg-inset text-muted hover:text-body cursor-pointer rounded-md px-2 py-1 font-medium disabled:cursor-default disabled:opacity-40 touch:min-h-11"
          >
            {save.state === "busy" ? "Saving…" : `Save as v${base.version + 1}`}
          </button>
        )}
        <button
          type="button"
          onClick={() => onAsk(applyPrompt(base, draft))}
          disabled={!dirty}
          title="Ask the agent to update the TTP and use it"
          className="cursor-pointer rounded-md px-2.5 py-1 font-semibold text-white transition-opacity hover:opacity-90 disabled:cursor-default disabled:opacity-40 touch:min-h-11"
          style={{ background: "var(--color-accent)" }}
        >
          Apply
        </button>
      </footer>
    </>,
  );
}

const GHOST = "text-muted hover:text-body hover:bg-inset cursor-pointer rounded-md px-2 py-1 font-medium touch:min-h-11";
const ACTION = "border-subtle text-body hover:bg-inset cursor-pointer rounded-md border px-2 py-0.5 text-[11px] font-medium touch:min-h-11";

function TtpList({
  items,
  total,
  jobs,
  selectedId,
  sessionId,
  onSelect,
}: {
  items: TtpSummary[];
  total: number;
  jobs: JobSummary[];
  selectedId: string | null;
  sessionId: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <nav aria-label="Stored TTPs" className="border-subtle hidden w-52 shrink-0 flex-col overflow-y-auto border-r py-1 @xl:flex">
      {items.length === 0 && <p className="text-faint px-3 py-2 text-[11px]">{total ? "No TTP matches." : "No TTPs stored yet."}</p>}
      {items.map((t) => {
        const runs = jobs.filter((j) => j.ttpId === t.id);
        const running = runs.some((j) => j.state === "running" || j.state === "queued");
        const selected = t.id === selectedId;
        return (
          <button
            key={t.id}
            type="button"
            onClick={() => onSelect(t.id)}
            aria-current={selected || undefined}
            className={`border-l-2 px-3 py-1.5 text-left ${selected ? "bg-sunken" : "hover:bg-sunken border-transparent"}`}
            style={selected ? { borderColor: "var(--color-accent)" } : undefined}
          >
            <span className="block truncate text-[11px] font-medium" title={t.name}>
              {t.name}
            </span>
            <span className="text-faint flex items-center gap-1 text-[10px]">
              <span className="truncate font-mono">{t.id}</span>
              <span className="shrink-0">· v{t.version}</span>
            </span>
            <span className="text-faint flex items-center gap-1 text-[10px]">
              {running && <span className="size-1.5 rounded-full" style={{ background: "var(--color-warn)" }} />}
              {runs.length ? `${runs.length} run${runs.length === 1 ? "" : "s"}${running ? ", running" : ""}` : "not run yet"}
              {t.id === sessionId && <span style={{ color: "var(--color-structure)" }}>· this chat</span>}
            </span>
          </button>
        );
      })}
    </nav>
  );
}

function TtpReadView({
  ttp,
  onServer,
  inSession,
  runs,
  saved,
  onAsk,
  onUse,
  onDuplicate,
  onDownload,
  onEdit,
}: {
  ttp: Ttp;
  onServer: boolean;
  inSession: boolean;
  runs: JobSummary[];
  saved: number | null;
  onAsk: (prompt: string) => void;
  onUse: () => void;
  onDuplicate: () => void;
  onDownload: () => void;
  onEdit: () => void;
}) {
  const [running, setRunning] = useState(false);
  const shares = weightShares(ttp);
  const all = [...objectivesOf(ttp, "pre_scoring"), ...objectivesOf(ttp, "scoring")];
  const largest = all.reduce((m, o) => Math.max(m, o.weight), 0) || 1;
  const created = ttpCreated(ttp);
  const generation = ttpGeneration(ttp);
  const target = targetSummary(ttp);
  const geometric = (ttp.scoring.aggregation ?? "geometric_mean") === "geometric_mean";
  const smarts = (ttp.constraints ?? []).flatMap((c) => (c.smarts ?? []).map((s) => ({ s, type: c.type })));

  return (
    <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-3">
      <div>
        <h3 className="text-sm font-semibold">{ttp.name}</h3>
        <p className="text-faint text-[10px]">
          <span className="font-mono">{ttp.id}</span> · v{ttp.version}
          {created && ` · created ${created}`}
          {inSession && <span style={{ color: "var(--color-structure)" }}> · used in this chat</span>}
          {!onServer && " · from this chat, not in the TTP store"}
          {saved !== null && <span style={{ color: "var(--color-ok)" }}> · saved as v{saved}</span>}
        </p>
        <div className="mt-2 flex flex-wrap gap-1.5">
          <button type="button" onClick={onUse} title="Put a reference to this TTP in the message box" className={ACTION} style={{ borderColor: "var(--color-accent)", color: "var(--color-accent)" }}>
            Use in chat
          </button>
          <button type="button" onClick={() => setRunning(!running)} aria-expanded={running} title="Put a REINVENT run request in the message box" className={ACTION}>
            Run…
          </button>
          <button type="button" onClick={onDuplicate} title="Open a copy in the builder" className={ACTION}>
            Duplicate and edit
          </button>
          <button type="button" onClick={onDownload} title={`Save ${ttpFileName(ttp)}`} className={ACTION}>
            Download JSON
          </button>
          <button type="button" onClick={onEdit} title="Change weights and windows in place" className={ACTION}>
            Edit weights
          </button>
        </div>
        {running && (
          <div className="mt-2">
            <RunCard key={ttp.id} ttp={ttp} cofold={hasCofolding(all)} onAsk={onAsk} onClose={() => setRunning(false)} />
          </div>
        )}
      </div>

      <Block title="Target">
        <p className="text-[11px]">{target ?? <span className="text-muted">None: molecules are scored on ligand properties only.</span>}</p>
        {ttp.target?.protein_sequence && (
          <p className="text-faint truncate font-mono text-[10px]" title={ttp.target.protein_sequence}>
            {ttp.target.protein_sequence}
          </p>
        )}
      </Block>

      <Block title="Objectives" hint={`${all.length} · ${geometric ? "geometric mean" : "arithmetic mean"}`}>
        <ul>
          {objectivesOf(ttp, "pre_scoring").map((o) => (
            <ObjectiveLine key={o.name} objective={o} share={shares.get(o.name) ?? 0} relative={o.weight / largest} tag="pre-scoring" />
          ))}
          {ttp.scoring.objectives.map((o) => (
            <ObjectiveLine key={o.name} objective={o} share={shares.get(o.name) ?? 0} relative={o.weight / largest} />
          ))}
        </ul>
        <p className="text-faint mt-1 text-[10px]">
          {geometric ? "Geometric mean: a molecule must do reasonably on every objective; one zero sinks the total." : "Arithmetic mean: a strong score can make up for a weak one."}
        </p>
      </Block>

      {smarts.length > 0 && (
        <Block title="Constraints" hint="substructures to avoid">
          <div className="flex flex-wrap gap-1">
            {smarts.map(({ s, type }, i) => (
              <span key={`${i}:${s}`} className="bg-sunken border-subtle rounded-md border px-1.5 py-px font-mono text-[10px]" title={type}>
                {s}
              </span>
            ))}
          </div>
        </Block>
      )}

      {generation.length > 0 && (
        <Block title="Generation">
          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[11px]">
            {generation.map((row) => (
              <div key={row.label} className="contents">
                <dt className="text-faint">{row.label}</dt>
                <dd className="truncate font-mono" title={row.value}>
                  {row.value}
                </dd>
              </div>
            ))}
          </dl>
        </Block>
      )}

      <Block title="Runs" hint={runs.length ? `${runs.length}` : undefined}>
        {runs.length === 0 ? (
          <p className="text-faint text-[11px]">Not run yet.</p>
        ) : (
          <ul className="-mx-1.5">
            {runs.map((job) => (
              <li key={job.id}>
                <button
                  type="button"
                  onClick={() => openRun(job.id, job.state === "failed" || job.state === "queued" ? "jobs" : "results")}
                  title={job.state === "failed" || job.state === "queued" ? "Open in Jobs" : "Open its results"}
                  className="hover:bg-sunken group flex w-full cursor-pointer items-center gap-2 rounded-md px-1.5 py-0.5 text-left text-[11px]"
                >
                  <span className="size-1.5 shrink-0 rounded-full" style={{ background: toneColor(stateTone(job.state)) }} />
                  <span className="w-16 shrink-0" style={{ color: toneColor(stateTone(job.state)) }}>
                    {job.state}
                  </span>
                  <span className="text-faint shrink-0 font-mono text-[10px]">{job.id}</span>
                  <span className="text-muted min-w-0 flex-1 truncate">
                    {[modelLabel(job.model), job.maxSteps && `${job.maxSteps} steps`, job.ttpVersion != null && `v${job.ttpVersion}`].filter(Boolean).join(" · ")}
                  </span>
                  <span className="text-faint shrink-0 text-[10px]">{shortDate(job.createdAt)}</span>
                  <span className="text-faint w-3 shrink-0 opacity-0 group-hover:opacity-100" aria-hidden="true">
                    ›
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </Block>

      {ttp.version > 1 && onServer && <VersionHistory key={`${ttp.id}@${ttp.version}`} ttp={ttp} runs={runs} />}
    </div>
  );
}

type Snapshot = { state: "busy" } | { state: "error"; message: string } | { state: "ready"; ttp: Ttp };

/** Earlier versions from the TTP store (get_ttp with a version), two of them compared. */
function VersionHistory({ ttp, runs }: { ttp: Ttp; runs: JobSummary[] }) {
  const [open, setOpen] = useState(false);
  const [from, setFrom] = useState(ttp.version - 1);
  const [to, setTo] = useState(ttp.version);
  const [snapshots, setSnapshots] = useState<Record<number, Snapshot>>({ [ttp.version]: { state: "ready", ttp } });
  const versions = Array.from({ length: ttp.version }, (_, i) => ttp.version - i);

  const load = (wanted: number[]) => {
    for (const version of wanted) {
      if (snapshots[version]) continue;
      setSnapshots((s) => ({ ...s, [version]: { state: "busy" } }));
      callTool("get_ttp", { ttp_id: ttp.id, version })
        .then((response) => {
          const error = payloadError(response.result);
          const loaded = error ? null : asTtp(parseResult(response.result));
          setSnapshots((s) => ({ ...s, [version]: loaded ? { state: "ready", ttp: loaded } : { state: "error", message: error ?? "Not a TTP." } }));
        })
        .catch((err: unknown) => setSnapshots((s) => ({ ...s, [version]: { state: "error", message: message(err) } })));
    }
  };

  const a = snapshots[from];
  const b = snapshots[to];
  const diff = a?.state === "ready" && b?.state === "ready" ? versionDiff(a.ttp, b.ttp) : null;
  const failed = [a, b].find((s): s is Extract<Snapshot, { state: "error" }> => s?.state === "error");
  const usedBy = (v: number) => runs.filter((j) => j.ttpVersion === v);
  const pick = (value: number, onChange: (v: number) => void, label: string) => (
    <select value={value} onChange={(event) => onChange(Number(event.target.value))} aria-label={label} className="bg-sunken border-subtle rounded border px-1 py-px font-mono text-[10px]">
      {versions.map((v) => (
        <option key={v} value={v}>
          v{v}
        </option>
      ))}
    </select>
  );

  return (
    <section>
      <button
        type="button"
        onClick={() => {
          setOpen(!open);
          if (!open) load([from, to]);
        }}
        aria-expanded={open} className="text-muted hover:text-body flex cursor-pointer gap-2 text-[11px] font-medium">
        {open ? "▾" : "▸"} History <span className="text-faint font-normal">{ttp.version} versions</span>
      </button>
      {open && (
        <div className="mt-1.5 space-y-2 pl-3 text-[11px]">
          <ul className="flex flex-wrap gap-x-3 gap-y-0.5">
            {versions.map((v) => {
              const used = usedBy(v);
              return (
                <li key={v} className="text-faint text-[10px]">
                  <span className={`font-mono ${v === ttp.version ? "text-body" : ""}`}>v{v}</span>{" "}
                  {used.length ? (
                    used.map((j) => (
                      <button key={j.id} type="button" onClick={() => openRun(j.id, "jobs")} className="hover:text-body cursor-pointer font-mono underline decoration-dotted" title={`${j.state} · open in Jobs`}>
                        {j.id}
                      </button>
                    ))
                  ) : (
                    <span>not run</span>
                  )}
                </li>
              );
            })}
          </ul>
          <div className="text-muted flex items-center gap-1.5">
            Compare{" "}
            {pick(
              from,
              (v) => {
                setFrom(v);
                load([v]);
              },
              "Older version",
            )}{" "}
            →{" "}
            {pick(
              to,
              (v) => {
                setTo(v);
                load([v]);
              },
              "Newer version",
            )}
          </div>
          {failed ? (
            <p className="text-faint">Could not load that version: {failed.message}</p>
          ) : !diff ? (
            <p className="text-faint">Loading…</p>
          ) : isSameVersion(diff) ? (
            <p className="text-faint">No differences.</p>
          ) : (
            <ul className="space-y-0.5">
              {diff.added.map((n) => (
                <li key={`+${n}`}>
                  <span style={{ color: "var(--color-ok)" }}>+</span> {n}
                </li>
              ))}
              {diff.removed.map((n) => (
                <li key={`-${n}`}>
                  <span style={{ color: "var(--color-accent)" }}>−</span> <span className="text-muted">{n}</span>
                </li>
              ))}
              {diff.changed.map((c) => (
                <li key={c.name}>
                  <span className="text-faint">~</span> {c.name} <span className="text-muted">{c.parts.join(", ")}</span>
                </li>
              ))}
              {diff.other.map((o) => (
                <li key={o} className="text-muted">
                  <span className="text-faint">~</span> {o}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}

function Block({ title, hint, children }: { title: string; hint?: string; children: ReactNode }) {
  return (
    <section>
      <h4 className="text-muted mb-1 flex gap-2 text-[11px] font-medium">
        {title}
        {hint && <span className="text-faint font-normal">{hint}</span>}
      </h4>
      {children}
    </section>
  );
}

function ObjectiveLine({ objective, share, relative, tag }: { objective: Objective; share: number; relative: number; tag?: string }) {
  const transform = objective.transform ?? null;
  const range = targetRange(transform);
  return (
    <li className="border-subtle flex items-center gap-3 border-t py-1.5 first:border-t-0">
      <div className="min-w-0 flex-1">
        <p className="truncate text-[11px] font-medium" title={objective.comment ?? objective.name}>
          {objective.name} <span className="text-faint text-[10px] font-normal">{[sourceLabel(objective), tag].filter(Boolean).join(" · ")}</span>
        </p>
        <p className="text-faint truncate text-[10px]">
          {curveLabel(transform?.type)}
          {range ? ` · ${range}` : ""}
          {objective.depends_on ? ` · after ${objective.depends_on}` : ""}
        </p>
      </div>
      {transform ? (
        <TransformCurve transform={transform} width={84} height={26} weight={relative} compact />
      ) : (
        <span className="text-faint w-[84px] shrink-0 text-center text-[10px]">0–1 as is</span>
      )}
      <div className="w-24 shrink-0" title={`weight ${objective.weight} · ${Math.round(share * 100)}% of the total`}>
        <div className="bg-inset h-1.5 overflow-hidden rounded-full">
          <div className="h-full rounded-full" style={{ width: `${Math.max(3, share * 100)}%`, background: "var(--color-table)" }} />
        </div>
        <p className="text-faint mt-0.5 flex justify-between text-[10px] tabular-nums">
          <span>w {+objective.weight.toFixed(2)}</span>
          <span>{Math.round(share * 100)}%</span>
        </p>
      </div>
    </li>
  );
}
