// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useDeferredValue, useEffect, useMemo, useState, type ReactNode } from "react";
import type { Theme } from "../../hooks/useTheme";
import { callTool, download, jobFileUrl } from "../../lib/api";
import { sameFile, type Artifacts } from "../../lib/artifacts";
import { assetUrl } from "../../lib/assets";
import { fileName } from "../../lib/csv";
import { canResume, isActive, jobTitle, parseJobs, resultsFile, toolError, type JobSummary } from "../../lib/jobs";
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
  structurePanel,
  type ColumnRole,
  type RunRow,
  type RunTable,
  type SortKey,
} from "../../lib/runResults";
import { addUserStructure } from "../../lib/userStructures";
import { sendToViewer } from "../../lib/viewerBus";
import { focusRun, useFocusedRun } from "../../lib/workbenchBus";
import { DownloadIcon, ExportButtons, TextButton } from "../DataTable/ExportButtons";
import { ContinueRun } from "../JobsPanel/ContinueRun";
import { MolDepiction } from "../MolDepiction/MolDepiction";
import { SmilesText } from "../MolDepiction/SmilesText";
import { StarButton } from "../Shortlist/StarButton";
import { ScoreWhy } from "./ScoreWhy";

interface Props {
  artifacts: Artifacts;
  theme: Theme;
  sessionId: string | null;
  onOpenMolecule?: (smiles: string) => void;
  onAsk?: (prompt: string) => void;
}

const ROW_HEIGHT = 44;
/** Rows drawn above and below the visible ones, so a fast scroll does not show blanks. */
const OVERSCAN = 6;
/** A running job's CSV grows by one batch per step; re-read it this often. */
const REFRESH_MS = 60_000;
const NUMBER_WIDTH = 84;

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
  const runs = useMemo(() => (jobs ?? []).filter((job) => job.type === "optimization"), [jobs]);
  const job = runs.find((j) => j.id === focused) ?? runs.find((j) => j.state !== "failed" && j.state !== "queued") ?? runs[0] ?? null;
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
  const [detailFor, setDetailFor] = useState<{ runId: string; id: number } | null>(null);
  const [continuing, setContinuing] = useState<string | null>(null);

  if (selectedFor.runId !== runId) setSelectedFor({ runId, ids: new Set() });
  const selected = selectedFor.ids;
  const detail = table && detailFor && detailFor.runId === runId ? (table.rows[detailFor.id] ?? null) : null;
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
  const template = ["24px", "64px", "minmax(180px,1fr)", ...shownColumns.map(() => `${NUMBER_WIDTH}px`), ...(hasStructures ? ["72px"] : [])].join(" ");
  const minWidth = 24 + 64 + 180 + shownColumns.length * NUMBER_WIDTH + (hasStructures ? 72 : 0) + 16;

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
    if (!row.structure) return;
    const panel = structurePanel(row.structure, row.smiles || null, `REINVENT ${runId}`);
    if (!panel) {
      setNotice(`${row.structure} is not on the shared volume, so the viewer cannot load it.`);
      return;
    }
    const known = artifacts.structures.some(({ item }) => sameFile(item.path, panel.path));
    if (!known) addUserStructure(sessionId ?? "new", panel);
    // Let the app pick up the added structure before the viewer asks for it.
    window.setTimeout(() => sendToViewer({ path: panel.path, action: { kind: "ligand" }, source: "results" }), known ? 0 : 50);
  }

  if (jobs === null) {
    return <Centered>{listError ? <p style={{ color: "var(--color-accent)" }}>Could not load the runs: {listError}</p> : <p className="text-faint">Loading runs…</p>}</Centered>;
  }
  if (!job) {
    return (
      <Centered>
        <p className="text-body font-medium">No REINVENT runs yet</p>
        <p className="text-muted max-w-sm">Every molecule an optimization run generates shows up here, with its scores.</p>
      </Centered>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col text-[11px]">
      <div className="border-subtle flex shrink-0 flex-wrap items-center gap-x-2 gap-y-1 border-b px-3 py-1.5">
        <select
          value={job.id}
          onChange={(event) => focusRun(event.target.value)}
          aria-label="Run"
          className="bg-sunken border-subtle max-w-64 min-w-0 cursor-pointer truncate rounded-md border px-1.5 py-1 text-[11px] outline-none"
        >
          {runs.map((j) => (
            <option key={j.id} value={j.id}>
              {jobTitle(j)} · {j.id} · {j.state}
            </option>
          ))}
        </select>
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
          <ContinueRun key={job.id} job={job} seeds={seeds} running={runs.find((j) => isActive(j.state)) ?? null} onClose={() => setContinuing(null)} onAsk={onAsk} />
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
              <SortHeader label="SMILES" active={sortKey === "smiles"} desc={sortDesc} onClick={() => toggleSort("smiles", false)} />
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
              {hasStructures && <span className="px-2">Complex</span>}
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
                    onOpenMolecule={onOpenMolecule}
                    structureColumn={hasStructures}
                    onOpenStructure={() => openStructure(row)}
                    isDetail={row === detail}
                    onDetail={() => setDetailFor(row === detail ? null : { runId: job.id, id: row.id })}
                    star={<StarButton smiles={row.smiles} source={job.id} session={session} values={() => numbersOf(rowValues(table, row))} />}
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {detail && table && (
        <div className="border-subtle bg-sunken flex shrink-0 items-start gap-3 border-t px-3 py-2" aria-label="Why this score">
          <button
            type="button"
            onClick={() => onOpenMolecule?.(detail.smiles)}
            disabled={!onOpenMolecule || !detail.smiles}
            className="enabled:hover:bg-inset h-12 w-16 shrink-0 rounded enabled:cursor-pointer"
            title="Open in Molecules"
          >
            {detail.smiles && <MolDepiction smiles={detail.smiles} theme={theme} width={128} height={96} className="size-full" />}
          </button>
          <div className="group flex w-40 min-w-0 shrink-0 flex-col gap-0.5">
            <SmilesText smiles={detail.smiles} className="max-w-32 text-[10px]" />
            <span className="text-faint tabular-nums">
              Score {formatCell(detail.total)}
              {detail.step !== null && ` · step ${detail.step}`}
            </span>
            {detail.failure && <span style={{ color: "var(--color-accent)" }}>{detail.failure}</span>}
          </div>
          <div className="min-w-0 flex-1">
            {detail.state === 0 || detail.state === 2 ? (
              <p className="text-muted">Not scored: REINVENT gives {detail.state === 0 ? "an invalid SMILES" : "a duplicate in the batch"} a score of 0.</p>
            ) : (
              <ScoreWhy values={rowValues(table, detail)} ttpId={job.ttpId} ttpVersion={job.ttpVersion} />
            )}
          </div>
          <button type="button" onClick={() => setDetailFor(null)} aria-label="Close" className="text-faint hover:text-body cursor-pointer px-1">
            ×
          </button>
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
  onOpenMolecule,
  structureColumn,
  onOpenStructure,
  isDetail,
  onDetail,
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
  onOpenMolecule?: (smiles: string) => void;
  structureColumn: boolean;
  onOpenStructure: () => void;
  isDetail: boolean;
  onDetail: () => void;
  star: ReactNode;
}) {
  const url = row.structure ? assetUrl(row.structure) : null;
  const edge = isDetail ? "inset 2px 0 0 var(--color-structure)" : row.failure ? "inset 2px 0 0 var(--color-accent)" : undefined;
  return (
    <div
      role="row"
      onClick={onDetail}
      title="Click for why it scored this"
      className={`border-subtle absolute inset-x-0 grid cursor-pointer items-center border-b ${selected || isDetail ? "bg-inset" : "hover:bg-sunken"} ${row.failure ? "text-faint" : ""}`}
      style={{ top, height: ROW_HEIGHT, gridTemplateColumns: template, boxShadow: edge }}
    >
      <span className="grid place-items-center" onClick={(event) => event.stopPropagation()}>
        <input type="checkbox" checked={selected} onChange={onToggle} aria-label={`Select row ${row.id + 1}`} className="accent-[var(--color-accent)]" />
      </span>
      <button
        type="button"
        onClick={(event) => {
          event.stopPropagation();
          onOpenMolecule?.(row.smiles);
        }}
        disabled={!onOpenMolecule || !row.smiles}
        title="Open in Molecules"
        className="h-full py-0.5 enabled:cursor-pointer"
      >
        {row.smiles && <MolDepiction smiles={row.smiles} theme={theme} width={120} height={80} className={`h-full w-full ${row.failure ? "opacity-60" : ""}`} />}
      </button>
      <span className="flex min-w-0 items-center gap-1 px-2">
        <span className="flex min-w-0 flex-1 flex-col justify-center">
          <span className="truncate font-mono text-[10px]" title={row.smiles}>
            {row.smiles || "—"}
          </span>
          {row.failure && (
            <span className="truncate text-[10px]" style={{ color: "var(--color-accent)" }}>
              {row.failure}
            </span>
          )}
        </span>
        {row.smiles && star}
      </span>
      {columns.map(({ index, integer }) => (
        <span key={index} className={`truncate px-2 text-right font-mono tabular-nums ${index === sortedBy ? "text-body font-semibold" : ""}`}>
          {formatCell(row.cells[index], integer)}
        </span>
      ))}
      {structureColumn && (
        <span className="flex items-center gap-0.5 px-1" onClick={(event) => event.stopPropagation()}>
          {row.structure && (
            <>
              <button
                type="button"
                onClick={onOpenStructure}
                title={`Show ${row.structure} in the viewer`}
                className="cursor-pointer rounded px-1.5 py-0.5 text-[10px] font-semibold"
                style={{ color: "var(--color-structure)" }}
              >
                3D
              </button>
              {url && (
                <a href={url} download title="Download the structure file" className="text-faint hover:text-body rounded p-1" aria-label="Download the structure file">
                  <DownloadIcon />
                </a>
              )}
            </>
          )}
        </span>
      )}
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
      {active && <span className="shrink-0 text-[9px]">{desc ? "▼" : "▲"}</span>}
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
