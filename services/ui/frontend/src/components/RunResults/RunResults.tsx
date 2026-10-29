// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useDeferredValue, useEffect, useMemo, useState, type ReactNode } from "react";
import type { Theme } from "../../hooks/useTheme";
import { callTool, download, jobFileUrl } from "../../lib/api";
import { sameFile, type Artifacts } from "../../lib/artifacts";
import { assetUrl } from "../../lib/assets";
import { fileName } from "../../lib/csv";
import { useJobOwners, useRunsKnown, useSessionRuns } from "../../hooks/useSessions";
import { canResume, conversationJobs, isActive, parseJobs, resultsFile, toolError, type JobSummary } from "../../lib/jobs";
import { scopedJob } from "../../lib/jobScope";
import { openRunStructure, rowMetrics, rowMolecule, runStructurePanel } from "../../lib/jobStructures";
import type { MoleculeContext } from "../../lib/moleculePage";
import { loadRDKit, withMol } from "../../lib/rdkit";
import {
  defaultColumns,
  filterRows,
  formatCell,
  parseRunResults,
  resultsSdf,
  resultsSheet,
  rowValues,
  runSeeds,
  sortRows,
  type ColumnRole,
  type RunRow,
  type RunTable,
  type SortKey,
} from "../../lib/runResults";
import { useFocusedRun } from "../../lib/workbenchBus";
import { DownloadIcon, ExportButtons, TextButton } from "../DataTable/ExportButtons";
import { ContinueRun } from "../JobsPanel/ContinueRun";
import { MoleculeThumb, StructureButton } from "../MoleculeTable/MoleculeTable";
import { StarButton } from "../Shortlist/StarButton";

interface Props {
  artifacts: Artifacts;
  theme: Theme;
  sessionId: string | null;
  onOpenMolecule?: (smiles: string, context?: MoleculeContext) => void;
  onAsk?: (prompt: string) => void;
}

const ROW_HEIGHT = 44;
/** Rows drawn above and below the visible ones, so a fast scroll does not show blanks. */
const OVERSCAN = 6;
/** A running job's CSV grows by one batch per step; re-read it this often. */
const REFRESH_MS = 60_000;
const NUMBER_WIDTH = 84;
const THUMB_WIDTH = 64 + 24;

const ROLE_LABEL: Record<ColumnRole, string> = {
  total: "Score",
  step: "Step",
  component: "Components",
  raw: "Raw values",
  structure: "Structure",
  meta: "Component details",
  likelihood: "Likelihoods",
  state: "SMILES state",
  text: "Text",
};
const ROLE_ORDER: ColumnRole[] = ["total", "step", "component", "raw", "meta", "likelihood", "state", "text"];

interface Loaded {
  id: string;
  table: RunTable | null;
  error: string | null;
}

const errorText = (error: unknown) => (error instanceof Error ? error.message : String(error));

/**
 * Every molecule a REINVENT run generated, with each score component, the
 * failures marked, and for a cofolding run the predicted complex of each
 * row. Read from the run's results CSV; drawn as a virtual list so a run of
 * tens of thousands of rows scrolls as fast as one of ten.
 */
export function RunResults({ artifacts, theme, sessionId, onOpenMolecule, onAsk }: Props) {
  const [jobs, setJobs] = useState<JobSummary[] | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const focused = useFocusedRun();
  const sessionRuns = useSessionRuns(sessionId);
  const owners = useJobOwners();
  const known = useRunsKnown();
  const scoped = useMemo(() => {
    // Until the list says who launched what, no job: the newest of all is often another conversation's.
    if (!jobs || known === null) return null;
    return scopedJob(jobs, known ? conversationJobs(jobs, sessionRuns, focused, owners, sessionId).own : jobs, focused);
  }, [jobs, known, sessionRuns, focused, owners, sessionId]);
  const job = scoped?.type === "optimization" ? scoped : null;
  const runId = job?.id ?? null;

  const [tick, setTick] = useState(0);
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const current = loaded && loaded.id === runId ? loaded : null;
  const table = current?.table ?? null;

  const [sort, setSort] = useState<{ key: SortKey; desc: boolean } | null>(null);
  const [query, setQuery] = useState("");
  const deferredQuery = useDeferredValue(query);
  const [minScore, setMinScore] = useState("");
  const [showFailed, setShowFailed] = useState(true);
  const [unique, setUnique] = useState(false);
  const [columnsFor, setColumnsFor] = useState<{ runId: string | null; keys: Set<string> } | null>(null);
  const [selectedFor, setSelectedFor] = useState<{ runId: string | null; ids: Set<number> }>({ runId, ids: new Set() });
  const [topN, setTopN] = useState(50);
  const [menu, setMenu] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [scroller, setScroller] = useState<HTMLDivElement | null>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewport, setViewport] = useState(600);
  const [continuing, setContinuing] = useState<string | null>(null);

  if (selectedFor.runId !== runId) setSelectedFor({ runId, ids: new Set() });
  const selected = selectedFor.ids;
  const seeds = useMemo(() => (table ? runSeeds(table) : undefined), [table]);
  const session = sessionId ?? "new";

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

  const file = job ? resultsFile(job) : null;
  useEffect(() => {
    if (!runId || !file) return;
    let cancelled = false;
    fetch(jobFileUrl(runId, file))
      .then(async (response) => {
        if (cancelled) return;
        if (!response.ok) {
          setLoaded({ id: runId, table: null, error: response.status === 404 ? "No results yet: REINVENT writes the CSV after the first step." : `HTTP ${response.status}` });
          return;
        }
        const text = await response.text();
        if (!cancelled) setLoaded({ id: runId, table: parseRunResults(text), error: null });
      })
      .catch((error: unknown) => {
        if (!cancelled) setLoaded({ id: runId, table: null, error: errorText(error) });
      });
    return () => {
      cancelled = true;
    };
  }, [runId, file, tick]);

  const running = job ? isActive(job.state) : false;
  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => {
      if (!document.hidden) setTick((t) => t + 1);
    }, REFRESH_MS);
    return () => window.clearInterval(timer);
  }, [running]);

  useEffect(() => {
    if (!scroller) return;
    const observer = new ResizeObserver(([entry]) => setViewport(entry.contentRect.height));
    observer.observe(scroller);
    return () => observer.disconnect();
  }, [scroller]);

  const totalAt = table ? table.columns.findIndex((c) => c.role === "total") : -1;
  const sortKey: SortKey = sort?.key ?? (totalAt >= 0 ? totalAt : "smiles");
  const sortDesc = sort ? sort.desc : totalAt >= 0;
  const min = minScore.trim() === "" ? null : Number(minScore);
  const minValue = min !== null && Number.isFinite(min) ? min : null;
  const rows = useMemo(
    () => (table ? sortRows(filterRows(table, { query: deferredQuery, minScore: minValue, showFailed, unique }), sortKey, sortDesc) : []),
    [table, deferredQuery, minValue, showFailed, unique, sortKey, sortDesc],
  );

  const visibleKeys = columnsFor && columnsFor.runId === runId ? columnsFor.keys : table ? defaultColumns(table) : new Set<string>();
  const shownColumns = table ? table.columns.map((column, index) => ({ column, index })).filter(({ column }) => visibleKeys.has(column.key)) : [];
  const hasStructures = table ? table.structureColumn >= 0 && table.rows.some((r) => r.structure) : false;
  const failures = (table?.failed ?? 0) > 0;
  // The complex right after the structure: a run has a dozen score columns, and the last of them is off screen.
  const template = ["24px", `${THUMB_WIDTH}px`, ...(hasStructures ? ["56px"] : []), ...shownColumns.map(() => `${NUMBER_WIDTH}px`), failures ? "minmax(140px,1fr)" : "minmax(0,1fr)"].join(" ");
  const minWidth = 24 + THUMB_WIDTH + (hasStructures ? 56 : 0) + shownColumns.length * NUMBER_WIDTH + (failures ? 140 : 0) + 16;

  const first = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - OVERSCAN);
  const last = Math.min(rows.length, Math.ceil((scrollTop + viewport) / ROW_HEIGHT) + OVERSCAN);
  const windowRows = rows.slice(first, last);

  function toggleSort(key: SortKey, numeric: boolean) {
    setSort(sortKey === key ? { key, desc: !sortDesc } : { key, desc: numeric });
  }

  function toggleColumn(key: string) {
    const keys = new Set(visibleKeys);
    if (keys.has(key)) keys.delete(key);
    else keys.add(key);
    setColumnsFor({ runId, keys });
  }

  function toggleRow(id: number) {
    const ids = new Set(selected);
    if (ids.has(id)) ids.delete(id);
    else ids.add(id);
    setSelectedFor({ runId, ids });
  }

  const allSelected = rows.length > 0 && rows.every((r) => selected.has(r.id));

  async function saveSdf() {
    if (!table || !runId) return;
    const picked = selected.size ? rows.filter((r) => selected.has(r.id)) : rows.filter((r) => !r.failure).slice(0, topN);
    if (!picked.length) {
      setNotice("Nothing to write: no rows selected or shown.");
      return;
    }
    try {
      const rdkit = await loadRDKit();
      const sdf = resultsSdf(table, picked, (smiles) => withMol(rdkit, smiles, (mol) => mol.get_molblock()), (row) => `${runId} step ${row.step ?? "?"} row ${row.id + 1}`);
      download(fileName(`${runId} ${selected.size ? "selected" : `top ${picked.length}`}`, "sdf"), sdf, "chemical/x-mdl-sdfile");
    } catch (error) {
      setNotice(`Could not write the SD file: ${errorText(error)}`);
    }
  }

  function openStructure(row: RunRow) {
    if (!row.structure || !runId || !table) return;
    const panel = runStructurePanel(row.structure, row.smiles || null, runId, row.step, rowMetrics(table, row));
    if (!panel) {
      setNotice(`${row.structure} is not on the shared volume, so the viewer cannot load it.`);
      return;
    }
    openRunStructure(session, panel, artifacts.structures.some(({ item }) => sameFile(item.path, panel.path)));
  }

  if (jobs === null) {
    return <Centered>{listError ? <p style={{ color: "var(--color-accent)" }}>Could not load the runs: {listError}</p> : <p className="text-faint">Loading runs…</p>}</Centered>;
  }
  if (!job) {
    return (
      <Centered>
        <p className="text-muted">{scoped ? "Results are for optimization runs." : "No job in this conversation yet."}</p>
      </Centered>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col text-[11px]">
      <div className="border-subtle flex shrink-0 flex-wrap items-center gap-x-2 gap-y-1 border-b px-3 py-1.5">
        {table && (
          <span className="text-faint tabular-nums">
            {rows.length === table.rows.length ? table.rows.length.toLocaleString() : `${rows.length.toLocaleString()} of ${table.rows.length.toLocaleString()}`} molecules
            {table.failed > 0 && <span style={{ color: "var(--color-accent)" }}> · {table.failed.toLocaleString()} failed</span>}
            {running && " · live"}
          </span>
        )}
        <button type="button" onClick={() => setTick((t) => t + 1)} title="Reload the results" aria-label="Reload" className="text-faint hover:text-body cursor-pointer p-0.5">
          <RefreshIcon />
        </button>
        {canResume(job) && (
          <TextButton title="Run more steps from this run's checkpoint (asks first)" onClick={() => setContinuing(continuing === job.id ? null : job.id)}>
            Continue…
          </TextButton>
        )}
        <span className="ml-auto flex items-center gap-0.5">
          {table && (
            <>
              <ExportButtons name={`${job.id} results`} sheet={() => resultsSheet(table, rows)} scope={`the ${rows.length.toLocaleString()} rows shown`} />
              <TextButton title={selected.size ? `Download the ${selected.size} selected molecules as an SD file` : `Download the top ${topN} scored molecules shown as an SD file`} onClick={() => void saveSdf()}>
                <DownloadIcon /> SDF {selected.size ? `(${selected.size})` : `top`}
              </TextButton>
              {!selected.size && (
                <input
                  type="number"
                  min={1}
                  max={10000}
                  value={topN}
                  onChange={(event) => setTopN(Math.max(1, Math.min(10000, Number(event.target.value) || 1)))}
                  aria-label="How many top molecules to write"
                  className="bg-sunken border-subtle w-12 rounded border px-1 py-0.5 text-[10px] tabular-nums outline-none"
                />
              )}
            </>
          )}
        </span>
      </div>

      {continuing === job.id && canResume(job) && (
        <div className="shrink-0 px-3 pb-2">
          <ContinueRun key={job.id} job={job} seeds={seeds} running={jobs.find((j) => j.type === "optimization" && isActive(j.state)) ?? null} onClose={() => setContinuing(null)} onAsk={onAsk} />
        </div>
      )}

      {table && (
        <div className="border-subtle flex shrink-0 flex-wrap items-center gap-x-3 gap-y-1 border-b px-3 py-1.5">
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Filter SMILES, scaffold, failure"
            aria-label="Filter rows"
            className="bg-sunken border-subtle focus:border-strong w-48 min-w-0 flex-1 rounded-md border px-2 py-1 text-[11px] outline-none"
          />
          <label className="text-muted flex items-center gap-1">
            Score ≥
            <input
              type="number"
              step={0.05}
              value={minScore}
              onChange={(event) => setMinScore(event.target.value)}
              placeholder="any"
              aria-label="Minimum score"
              className="bg-sunken border-subtle w-14 rounded-md border px-1.5 py-1 text-[11px] tabular-nums outline-none"
            />
          </label>
          <Toggle checked={showFailed} onChange={setShowFailed} title="Rows with an invalid SMILES, a duplicate or a component that returned NaN">
            Failed{table.failed > 0 ? ` ${table.failed.toLocaleString()}` : ""}
          </Toggle>
          <Toggle checked={unique} onChange={setUnique} title="One row per SMILES: its best scoring one">
            Unique
          </Toggle>
          <div className="relative">
            <button
              type="button"
              onClick={() => setMenu((open) => !open)}
              aria-haspopup="menu"
              aria-expanded={menu}
              className="text-muted hover:text-body hover:bg-inset cursor-pointer rounded px-1.5 py-0.5"
            >
              Columns {shownColumns.length}/{table.columns.length} ▾
            </button>
            {menu && (
              <>
                <button type="button" aria-label="Close menu" tabIndex={-1} className="fixed inset-0 z-30 cursor-default" onClick={() => setMenu(false)} />
                <div role="menu" className="bg-raised border-subtle shadow-card absolute top-full right-0 z-40 mt-1 max-h-80 w-64 overflow-y-auto rounded-lg border py-1">
                  {ROLE_ORDER.map((role) => {
                    const group = table.columns.filter((c) => c.role === role);
                    if (!group.length) return null;
                    return (
                      <div key={role} className="py-0.5">
                        <p className="text-faint px-3 pt-1 text-[10px] font-semibold tracking-wider uppercase">{ROLE_LABEL[role]}</p>
                        {group.map((c) => (
                          <label key={c.key} className="hover:bg-sunken flex cursor-pointer items-center gap-2 px-3 py-1" title={c.key}>
                            <input type="checkbox" checked={visibleKeys.has(c.key)} onChange={() => toggleColumn(c.key)} className="accent-[var(--color-accent)]" />
                            <span className="truncate">{c.label}</span>
                          </label>
                        ))}
                      </div>
                    );
                  })}
                  <button type="button" onClick={() => setColumnsFor(null)} className="text-muted hover:text-body w-full cursor-pointer px-3 py-1 text-left">
                    Reset columns
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {notice && (
        <p className="border-subtle flex shrink-0 items-center gap-2 border-b px-3 py-1" style={{ boxShadow: "inset 2px 0 0 var(--color-warn)" }}>
          <span className="text-muted min-w-0 flex-1">{notice}</span>
          <button type="button" onClick={() => setNotice(null)} aria-label="Dismiss" className="text-faint hover:text-body cursor-pointer">
            ×
          </button>
        </p>
      )}

      {!current ? (
        <Centered>
          <p className="text-faint">Loading results…</p>
        </Centered>
      ) : current.error || !table ? (
        <Centered>
          <p className="text-muted max-w-sm">{current.error ?? "No results."}</p>
        </Centered>
      ) : (
        <div ref={setScroller} onScroll={(event) => setScrollTop(event.currentTarget.scrollTop)} className="relative min-h-0 flex-1 overflow-auto">
          <div style={{ minWidth }}>
            <div
              role="row"
              className="bg-raised border-subtle text-muted sticky top-0 z-10 grid items-center border-b font-medium"
              style={{ gridTemplateColumns: template, height: 32 }}
            >
              <span className="grid place-items-center">
                <input
                  type="checkbox"
                  checked={allSelected}
                  onChange={() => setSelectedFor({ runId, ids: allSelected ? new Set() : new Set(rows.map((r) => r.id)) })}
                  aria-label="Select every row shown"
                  className="accent-[var(--color-accent)]"
                />
              </span>
              <span />
              {hasStructures && <span className="px-2">Complex</span>}
              {shownColumns.map(({ column, index }) => (
                <SortHeader
                  key={column.key}
                  label={column.label}
                  title={column.key}
                  align="right"
                  active={sortKey === index}
                  desc={sortDesc}
                  onClick={() => toggleSort(index, column.numeric)}
                />
              ))}
              <span className="px-2">{failures ? "Failure" : ""}</span>
            </div>
            {rows.length === 0 ? (
              <p className="text-faint py-10 text-center">No rows match.</p>
            ) : (
              <div className="relative" style={{ height: rows.length * ROW_HEIGHT }}>
                {windowRows.map((row, i) => (
                  <Row
                    key={row.id}
                    row={row}
                    top={(first + i) * ROW_HEIGHT}
                    template={template}
                    columns={shownColumns.map(({ column, index }) => ({ index, integer: column.integer }))}
                    sortedBy={typeof sortKey === "number" ? sortKey : -1}
                    selected={selected.has(row.id)}
                    onToggle={() => toggleRow(row.id)}
                    theme={theme}
                    structureColumn={hasStructures}
                    onOpenStructure={() => openStructure(row)}
                    onOpen={onOpenMolecule && row.smiles ? () => onOpenMolecule(row.smiles, rowMolecule(table, row, job.id, job)) : undefined}
                    star={<StarButton smiles={row.smiles} source={job.id} session={session} values={() => numbersOf(rowValues(table, row))} />}
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      )}

    </div>
  );
}

function numbersOf(values: Record<string, number | null>): Record<string, number> {
  return Object.fromEntries(Object.entries(values).filter((entry): entry is [string, number] => entry[1] !== null));
}

function Row({
  row,
  top,
  template,
  columns,
  sortedBy,
  selected,
  onToggle,
  theme,
  structureColumn,
  onOpenStructure,
  onOpen,
  star,
}: {
  row: RunRow;
  top: number;
  template: string;
  columns: { index: number; integer: boolean }[];
  sortedBy: number;
  selected: boolean;
  onToggle: () => void;
  theme: Theme;
  structureColumn: boolean;
  onOpenStructure: () => void;
  /** Opens the molecule's page. */
  onOpen?: () => void;
  star: ReactNode;
}) {
  const url = row.structure ? assetUrl(row.structure) : null;
  return (
    <div
      role="row"
      onClick={onOpen}
      title={onOpen ? "Open the molecule" : undefined}
      className={`border-subtle absolute inset-x-0 grid items-center border-b ${onOpen ? "cursor-pointer" : ""} ${selected ? "bg-inset" : "hover:bg-sunken"} ${row.failure ? "text-faint" : ""}`}
      style={{ top, height: ROW_HEIGHT, gridTemplateColumns: template, boxShadow: row.failure ? "inset 2px 0 0 var(--color-accent)" : undefined }}
    >
      <span className="grid place-items-center" onClick={(event) => event.stopPropagation()}>
        <input type="checkbox" checked={selected} onChange={onToggle} aria-label={`Select row ${row.id + 1}`} className="accent-[var(--color-accent)]" />
      </span>
      <span className="h-full py-0.5">
        <MoleculeThumb smiles={row.smiles} theme={theme} width={64} height={ROW_HEIGHT - 4} dim={!!row.failure} actions={row.smiles ? star : null} />
      </span>
      {structureColumn && (
        <span className="flex items-center" onClick={(event) => event.stopPropagation()}>
          {row.structure && (
            <>
              <StructureButton onClick={onOpenStructure} title={`Show ${row.structure} in the viewer`} />
              {url && (
                <a href={url} download title="Download the structure file" className="text-faint hover:text-body rounded p-1" aria-label="Download the structure file">
                  <DownloadIcon />
                </a>
              )}
            </>
          )}
        </span>
      )}
      {columns.map(({ index, integer }) => (
        <span key={index} className={`truncate px-2 text-right font-mono tabular-nums ${index === sortedBy ? "text-body font-semibold" : ""}`}>
          {formatCell(row.cells[index], integer)}
        </span>
      ))}
      <span className="truncate px-2 text-[10px]" style={{ color: "var(--color-accent)" }} title={row.failure || undefined}>
        {row.failure}
      </span>
    </div>
  );
}

function SortHeader({ label, title, active, desc, align, onClick }: { label: string; title?: string; active: boolean; desc: boolean; align?: "right"; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title ? `${title} · click to sort` : "Click to sort"}
      aria-sort={active ? (desc ? "descending" : "ascending") : undefined}
      className={`hover:text-body flex h-full min-w-0 cursor-pointer items-center gap-0.5 px-2 ${align === "right" ? "justify-end text-right" : "text-left"} ${active ? "text-body" : ""}`}
    >
      <span className="line-clamp-2 leading-tight break-words">{label}</span>
      {active && <span className="shrink-0 text-[8px]">{desc ? "▼" : "▲"}</span>}
    </button>
  );
}

function Toggle({ checked, onChange, title, children }: { checked: boolean; onChange: (value: boolean) => void; title?: string; children: ReactNode }) {
  return (
    <label className="text-muted flex cursor-pointer items-center gap-1" title={title}>
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} className="accent-[var(--color-accent)]" />
      {children}
    </label>
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
