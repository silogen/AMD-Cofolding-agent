// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useRDKit } from "../../hooks/useRDKit";
import type { Artifacts } from "../../lib/artifacts";
import {
  admetPrompt,
  type Candidate,
  candidatesOf,
  candidateNumbers,
  candidatesToCsv,
  candidatesToSdf,
  candidateValue,
  cofoldPrompt,
  DESCRIPTOR_COLUMNS,
  descriptorValue,
  MAX_COFOLD,
  type MolProps,
  numericColumns,
  ro5Violations,
} from "../../lib/candidates";
import { descriptors, withMol } from "../../lib/rdkit";
import { MolDepiction } from "../MolDepiction/MolDepiction";
import { SmilesText } from "../MolDepiction/SmilesText";
import { StarButton } from "../Shortlist/StarButton";
import { CandidateScatter } from "./CandidateScatter";
import { GateBadge } from "./GateBadge";
import { ScoreBar } from "./ScoreBar";

interface Props {
  artifacts: Artifacts;
  theme: "dark" | "light";
  onOpenMolecule: (smiles: string) => void;
  onAsk: (prompt: string) => void;
}

/** "name", "origin", "gate", a numeric column key as in candidateValue, or "desc:<field>". */
type SortKey = string;
interface Sort {
  key: SortKey;
  dir: 1 | -1;
}

/** A toggleable table column. */
interface Column {
  key: string;
  label: string;
  kind: "score" | "number" | "descriptor" | "origin" | "gate";
  digits?: number;
  /** Shown until the user changes it in the Columns menu. */
  defaultOn: boolean;
}

/** Component columns shown by default; the rest can be turned on in the Columns menu. */
const TABLE_COMPONENTS = 4;
/** Row heights in px: the drawing (h-12 or h-8), its cell padding and the border. */
const ROW_HEIGHT = 57;
const DENSE_ROW_HEIGHT = 37;
/** Rows drawn above and below the visible ones, so a fast scroll does not show blanks. */
const OVERSCAN = 6;
/** Scroll positions are tracked in steps of this many px, so scrolling re-renders the table less often. */
const SCROLL_STEP = 32;
const GATE_RANK = { pass: 2, unknown: 1, fail: 0 } as const;

const control = "bg-sunken border-subtle rounded-md border px-2 py-1 text-xs";
const button =
  "border-subtle hover:bg-sunken inline-flex cursor-pointer items-center gap-1 rounded-md border px-2 py-1 text-[11px] font-medium";
const menuItem = "hover:bg-sunken flex w-full cursor-pointer items-center gap-2 rounded px-2 py-1 text-left disabled:cursor-not-allowed disabled:opacity-50";

function download(name: string, text: string, type: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

/**
 * Every molecule the conversation produced, in one sortable, filterable list:
 * table rows with a SMILES, drawn molecules and cofolded ligands merged by
 * SMILES, with a scatter view. A click opens the molecule's page.
 * Descriptors are computed in the browser with RDKit.
 */
export function CandidatesPanel({ artifacts, theme, onOpenMolecule, onAsk }: Props) {
  const candidates = candidatesOf(artifacts);
  const { rdkit } = useRDKit(candidates.length > 0);
  const props = useMemo(() => {
    const out = new Map<string, MolProps>();
    if (!rdkit) return out;
    for (const c of candidates) {
      const d = descriptors(rdkit, c.smiles);
      if (d) out.set(c.smiles, { mw: d.mw, clogp: d.clogp, tpsa: d.tpsa, hbd: d.hbd, hba: d.hba, rotb: d.rotb, ro5: ro5Violations(d) });
    }
    return out;
  }, [rdkit, candidates]);
  const columns = useMemo(() => numericColumns(candidates), [candidates]);
  const origins = useMemo(() => [...new Set(candidates.map((c) => c.origin))], [candidates]);
  const hasScore = columns.some((c) => c.key === "score");

  const allColumns = useMemo<Column[]>(() => {
    const out: Column[] = [];
    if (hasScore) out.push({ key: "score", label: "Score", kind: "score", defaultOn: true });
    columns
      .filter((c) => c.key !== "score")
      .forEach((c, i) => out.push({ key: c.key, label: c.label, kind: "number", defaultOn: i < TABLE_COMPONENTS }));
    if (props.size > 0) {
      for (const d of DESCRIPTOR_COLUMNS) out.push({ key: d.key, label: d.label, kind: "descriptor", digits: d.digits, defaultOn: d.key !== "desc:rotb" });
    }
    out.push({ key: "origin", label: "Origin", kind: "origin", defaultOn: true });
    out.push({ key: "gate", label: "Gate", kind: "gate", defaultOn: true });
    return out;
  }, [columns, hasScore, props]);

  const [query, setQuery] = useState("");
  const [origin, setOrigin] = useState("");
  const [minScore, setMinScore] = useState("");
  const [gateOnly, setGateOnly] = useState(false);
  const [view, setView] = useState<"table" | "scatter">("table");
  const [dense, setDense] = useState(false);
  const [sort, setSort] = useState<Sort>({ key: "score", dir: -1 });
  const [selected, setSelected] = useState<Set<string>>(() => new Set());
  /** Columns the user turned on or off; the rest follow Column.defaultOn. */
  const [shown, setShown] = useState<Record<string, boolean>>({});
  const [axes, setAxes] = useState<{ x: string; y: string } | null>(null);
  const [copied, setCopied] = useState(false);
  const [scroller, setScroller] = useState<HTMLDivElement | null>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewport, setViewport] = useState(600);

  useEffect(() => {
    if (!scroller) return;
    const observer = new ResizeObserver(([entry]) => setViewport(entry.contentRect.height));
    observer.observe(scroller);
    return () => observer.disconnect();
  }, [scroller]);

  const visible = allColumns.filter((c) => shown[c.key] ?? c.defaultOn);
  const propsOf = (smiles: string) => props.get(smiles);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const min = minScore.trim() === "" ? null : Number(minScore);
    const sortValue = (c: Candidate): string | number | undefined => {
      if (sort.key === "name") return c.name ?? c.smiles;
      if (sort.key === "origin") return c.origin;
      if (sort.key === "gate") return GATE_RANK[c.gate.status];
      if (sort.key.startsWith("desc:")) return descriptorValue(props.get(c.smiles), sort.key);
      return candidateValue(c, sort.key);
    };
    const rows = candidates.filter(
      (c) =>
        (!q || [c.smiles, c.name ?? "", c.origins.join(" ")].join(" ").toLowerCase().includes(q)) &&
        (!origin || c.origin === origin) &&
        (min === null || Number.isNaN(min) || (c.score !== undefined && c.score >= min)) &&
        (!gateOnly || c.gate.status === "pass"),
    );
    return rows.sort((a, b) => {
      const va = sortValue(a);
      const vb = sortValue(b);
      // Missing values go last whichever way the column is sorted.
      if (va === undefined) return vb === undefined ? 0 : 1;
      if (vb === undefined) return -1;
      if (typeof va === "number" && typeof vb === "number") return (va - vb) * sort.dir;
      return String(va).localeCompare(String(vb)) * sort.dir;
    });
  }, [candidates, props, query, origin, minScore, gateOnly, sort]);

  const chosen = candidates.filter((c) => selected.has(c.smiles));
  const scope = chosen.length ? chosen : filtered;
  const scopeLabel = chosen.length ? `${chosen.length} selected` : filtered.length === candidates.length ? `all ${candidates.length}` : `${filtered.length} shown`;
  const defaultX = columns.find((c) => c.key !== "score")?.key ?? columns[0]?.key ?? "";
  const defaultY = hasScore ? "score" : (columns.find((c) => c.key !== defaultX)?.key ?? defaultX);
  const has = (key: string) => columns.some((c) => c.key === key);
  const x = axes && has(axes.x) ? axes.x : defaultX;
  const y = axes && has(axes.y) ? axes.y : defaultY;

  const allVisibleSelected = filtered.length > 0 && filtered.every((c) => selected.has(c.smiles));

  function toggle(smiles: string) {
    setSelected((previous) => {
      const next = new Set(previous);
      if (next.has(smiles)) next.delete(smiles);
      else next.add(smiles);
      return next;
    });
  }

  function toggleAll() {
    setSelected((previous) => {
      const next = new Set(previous);
      for (const c of filtered) {
        if (allVisibleSelected) next.delete(c.smiles);
        else next.add(c.smiles);
      }
      return next;
    });
  }

  function sortBy(key: SortKey) {
    setSort((previous) => {
      if (previous.key !== key) return { key, dir: key === "name" || key === "origin" ? 1 : -1 };
      if (previous.dir === -1) return { key, dir: 1 };
      return { key: "score", dir: -1 };
    });
  }

  function copySmiles() {
    void navigator.clipboard?.writeText(chosen.map((c) => c.smiles).join("\n")).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }

  function exportSdf() {
    if (!rdkit) return;
    const sdf = candidatesToSdf(scope, (smiles) => withMol(rdkit, smiles, (mol) => mol.get_molblock()), propsOf);
    download("candidates.sdf", sdf, "chemical/x-mdl-sdfile");
  }

  if (candidates.length === 0) {
    return (
      <div className="text-muted grid h-full place-items-center p-6 text-center text-xs">
        <div>
          <p className="text-body text-sm font-semibold">No candidates yet</p>
          <p className="mt-1">Molecules appear here once a tool returns SMILES: scoring, optimisation results or cofolding.</p>
        </div>
      </div>
    );
  }

  const cellY = dense ? "py-0.5" : "py-1";
  const rowHeight = dense ? DENSE_ROW_HEIGHT : ROW_HEIGHT;
  const firstRow = Math.max(0, Math.floor(scrollTop / rowHeight) - OVERSCAN);
  const lastRow = Math.min(filtered.length, Math.ceil((scrollTop + viewport + SCROLL_STEP) / rowHeight) + OVERSCAN);
  const spanAll = 3 + visible.length;
  const header = (key: SortKey, label: string, align: "left" | "right" = "left") => {
    const active = sort.key === key;
    return (
      <th
        key={key}
        scope="col"
        aria-sort={active ? (sort.dir === 1 ? "ascending" : "descending") : "none"}
        className={`border-subtle border-b px-2 py-2 font-semibold whitespace-nowrap ${align === "right" ? "text-right" : "text-left"}`}
      >
        <button
          type="button"
          onClick={() => sortBy(key)}
          className={`hover:text-body inline-flex w-full cursor-pointer items-center gap-1 ${align === "right" ? "justify-end" : ""}`}
          style={{ color: active ? "var(--color-accent)" : undefined }}
          title={label}
        >
          <span className="max-w-24 truncate">{label}</span>
          <span aria-hidden="true" className="text-[9px]">
            {active ? (sort.dir === 1 ? "▲" : "▼") : "↕"}
          </span>
        </button>
      </th>
    );
  };

  const cell = (c: Candidate, col: Column) => {
    const base = `border-subtle border-b px-2 ${cellY}`;
    if (col.kind === "score") {
      return (
        <td key={col.key} className={`${base} text-right whitespace-nowrap`}>
          <ScoreBar value={c.score} />
        </td>
      );
    }
    if (col.kind === "origin") {
      return (
        <td key={col.key} className={base}>
          <span className="bg-inset text-muted rounded-full px-1.5 py-px text-[10px] whitespace-nowrap">{c.origin}</span>
        </td>
      );
    }
    if (col.kind === "gate") {
      return (
        <td key={col.key} className={base}>
          <GateBadge gate={c.gate} />
        </td>
      );
    }
    const value = col.kind === "descriptor" ? descriptorValue(props.get(c.smiles), col.key) : candidateValue(c, col.key);
    const warn = col.key === "desc:ro5" && value !== undefined && value > 1;
    return (
      <td
        key={col.key}
        className={`${base} text-right font-mono whitespace-nowrap tabular-nums`}
        style={warn ? { color: "var(--color-warn)" } : undefined}
      >
        {value === undefined ? (
          <span className="text-faint">—</span>
        ) : (
          value.toFixed(col.digits ?? (Number.isInteger(value) ? 0 : 2))
        )}
      </td>
    );
  };

  return (
    <div className="bg-raised flex h-full min-h-0 flex-col">
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <div className="border-subtle @container border-b px-3 py-2">
          <div className="flex flex-wrap items-center gap-2">
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search name, SMILES, tool…"
              aria-label="Search candidates"
              className={`${control} w-full @md:w-auto @md:min-w-36 @md:flex-1 @3xl:w-44 @3xl:flex-none`}
            />
            <select aria-label="Origin" value={origin} onChange={(e) => setOrigin(e.target.value)} className={`${control} max-w-40`}>
              <option value="">All origins</option>
              {origins.map((o) => (
                <option key={o} value={o}>
                  {o}
                </option>
              ))}
            </select>
            <input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={minScore}
              onChange={(e) => setMinScore(e.target.value)}
              placeholder="Min score"
              aria-label="Minimum score"
              className={`${control} w-24`}
            />
            <button
              type="button"
              aria-pressed={gateOnly}
              onClick={() => setGateOnly((v) => !v)}
              title="Only candidates that pass the quality gate"
              className={button}
              style={gateOnly ? { borderColor: "var(--color-ok)", color: "var(--color-ok)" } : undefined}
            >
              Passing
            </button>
            <div className="ml-auto flex items-center gap-1">
              <div className="border-subtle inline-flex overflow-hidden rounded-md border" role="group" aria-label="View">
                {(["table", "scatter"] as const).map((v) => (
                  <button
                    key={v}
                    type="button"
                    aria-pressed={view === v}
                    onClick={() => setView(v)}
                    className={`cursor-pointer px-2.5 py-1 text-[11px] font-medium capitalize ${view === v ? "bg-inset text-body" : "text-muted hover:text-body"}`}
                  >
                    {v}
                  </button>
                ))}
              </div>
              <Menu title="Download and columns">
                {(close) => (
                  <>
                    <p className="text-faint px-2 pt-1 pb-0.5 text-[10px]">Download {scopeLabel}</p>
                    <button
                      type="button"
                      role="menuitem"
                      className={menuItem}
                      onClick={() => {
                        download("candidates.csv", candidatesToCsv(scope, props.size ? propsOf : undefined), "text/csv");
                        close();
                      }}
                    >
                      CSV
                    </button>
                    <button
                      type="button"
                      role="menuitem"
                      className={menuItem}
                      disabled={!rdkit}
                      onClick={() => {
                        exportSdf();
                        close();
                      }}
                      title={rdkit ? "2D molblocks from RDKit, with scores and descriptors as SD fields" : "RDKit is loading"}
                    >
                      SDF
                    </button>
                    <div className="border-subtle my-1 border-t" />
                    <label className={menuItem}>
                      <input type="checkbox" checked={dense} onChange={(e) => setDense(e.target.checked)} className="cursor-pointer" />
                      Compact rows
                    </label>
                    <div className="border-subtle my-1 border-t" />
                    <p className="text-faint px-2 pt-1 pb-0.5 text-[10px]">Columns</p>
                    {allColumns.map((col) => (
                      <label key={col.key} className={menuItem}>
                        <input
                          type="checkbox"
                          checked={shown[col.key] ?? col.defaultOn}
                          onChange={(e) => setShown((previous) => ({ ...previous, [col.key]: e.target.checked }))}
                          className="cursor-pointer"
                        />
                        <span className="truncate">{col.label}</span>
                        {col.kind === "descriptor" && <span className="text-faint ml-auto text-[9px]">RDKit</span>}
                      </label>
                    ))}
                  </>
                )}
              </Menu>
            </div>
          </div>
        </div>

        {chosen.length > 0 && (
          <div className="border-subtle bg-sunken flex flex-wrap items-center gap-2 border-b px-3 py-1.5 text-[11px]">
            <span className="font-semibold">{chosen.length} selected</span>
            <button
              type="button"
              className={button}
              onClick={() => onAsk(cofoldPrompt(chosen))}
              title={`Ask the agent to cofold ${chosen.length > MAX_COFOLD ? `the first ${MAX_COFOLD}` : "them"}`}
            >
              Cofold{chosen.length > MAX_COFOLD ? ` first ${MAX_COFOLD}` : ""}
            </button>
            <button type="button" className={button} onClick={() => onAsk(admetPrompt(chosen))} title="Ask the agent for their ADMET properties">
              ADMET
            </button>
            <button type="button" className={button} onClick={copySmiles}>
              {copied ? "Copied" : "Copy SMILES"}
            </button>
            <button type="button" className="text-muted hover:text-body ml-auto cursor-pointer" onClick={() => setSelected(new Set())}>
              Clear
            </button>
          </div>
        )}

        <p className="text-faint px-3 pt-1.5 text-[11px]">
          Showing {filtered.length} of {candidates.length}
        </p>

        <div
          ref={setScroller}
          onScroll={(e) => setScrollTop(Math.floor(e.currentTarget.scrollTop / SCROLL_STEP) * SCROLL_STEP)}
          className="min-h-0 flex-1 overflow-auto"
        >
          {view === "scatter" ? (
            <CandidateScatter
              candidates={filtered}
              columns={columns}
              x={x}
              y={y}
              onAxes={(nx, ny) => setAxes({ x: nx, y: ny })}
              selected={selected}
              detail={null}
              onPick={onOpenMolecule}
            />
          ) : (
            <table className={`w-full border-collapse ${dense ? "text-[11px]" : "text-xs"}`}>
              <thead className="bg-sunken sticky top-0 z-10">
                <tr>
                  <th scope="col" className="border-subtle w-8 border-b px-2 py-2">
                    <input
                      type="checkbox"
                      aria-label="Select all shown"
                      checked={allVisibleSelected}
                      onChange={toggleAll}
                      className="cursor-pointer"
                    />
                  </th>
                  <th scope="col" className="border-subtle text-muted border-b px-2 py-2 text-left font-semibold">
                    2D
                  </th>
                  {header("name", "Name")}
                  {visible.map((c) => header(c.key, c.label, c.kind === "origin" || c.kind === "gate" ? "left" : "right"))}
                </tr>
              </thead>
              <tbody>
                {firstRow > 0 && (
                  <tr aria-hidden="true">
                    <td colSpan={spanAll} className="p-0" style={{ height: firstRow * rowHeight }} />
                  </tr>
                )}
                {filtered.slice(firstRow, lastRow).map((c) => {
                  return (
                    <tr
                      key={c.smiles}
                      onClick={() => onOpenMolecule(c.smiles)}
                      className={`group hover:bg-sunken cursor-pointer ${selected.has(c.smiles) ? "bg-sunken" : ""}`}
                      style={{ height: rowHeight }}
                    >
                      <td className={`border-subtle border-b px-2 ${cellY}`} onClick={(e) => e.stopPropagation()}>
                        <input
                          type="checkbox"
                          aria-label={`Select ${c.name ?? c.smiles}`}
                          checked={selected.has(c.smiles)}
                          onChange={() => toggle(c.smiles)}
                          className="cursor-pointer"
                        />
                      </td>
                      <td className={`border-subtle border-b px-2 ${cellY}`}>
                        <MolDepiction smiles={c.smiles} theme={theme} width={128} height={96} label={c.name} className={dense ? "h-8 w-11" : "h-12 w-16"} />
                      </td>
                      <td className={`border-subtle border-b px-2 ${cellY}`}>
                        <span className="flex items-center gap-1">
                          <span className="flex min-w-0 flex-col">
                            {c.name && <span className="max-w-44 truncate font-medium">{c.name}</span>}
                            <SmilesText smiles={c.smiles} className="text-muted max-w-44 text-[10px]" />
                          </span>
                          <StarButton smiles={c.smiles} label={c.name} source="candidates" values={() => candidateNumbers(c)} />
                        </span>
                      </td>
                      {visible.map((col) => cell(c, col))}
                    </tr>
                  );
                })}
                {lastRow < filtered.length && (
                  <tr aria-hidden="true">
                    <td colSpan={spanAll} className="p-0" style={{ height: (filtered.length - lastRow) * rowHeight }} />
                  </tr>
                )}
              </tbody>
            </table>
          )}
          {filtered.length === 0 && (
            <p className="text-muted px-3 py-8 text-center text-xs">No matching candidates. Adjust the filters or clear the search.</p>
          )}
        </div>
      </div>
    </div>
  );
}

/** A quiet "⋯" button opening a menu, closed by a click outside or Escape. */
function Menu({ title, children }: { title: string; children: (close: () => void) => ReactNode }) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const outside = (event: PointerEvent) => {
      if (!(event.target instanceof Node) || !rootRef.current?.contains(event.target)) setOpen(false);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", outside);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("pointerdown", outside);
      document.removeEventListener("keydown", escape);
    };
  }, [open]);

  return (
    <div ref={rootRef} className="relative">
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={title}
        title={title}
        onClick={() => setOpen((v) => !v)}
        className={`hover:text-body hover:bg-sunken grid size-7 cursor-pointer place-items-center rounded-md ${open ? "bg-inset text-body" : "text-muted"}`}
      >
        <svg viewBox="0 0 24 24" className="size-4" fill="currentColor" aria-hidden="true">
          <circle cx="5" cy="12" r="1.6" />
          <circle cx="12" cy="12" r="1.6" />
          <circle cx="19" cy="12" r="1.6" />
        </svg>
      </button>
      {open && (
        <div role="menu" className="bg-raised border-subtle shadow-card absolute right-0 z-20 mt-1 max-h-80 w-52 overflow-y-auto rounded-lg border p-1 text-[11px]">
          {children(() => setOpen(false))}
        </div>
      )}
    </div>
  );
}
