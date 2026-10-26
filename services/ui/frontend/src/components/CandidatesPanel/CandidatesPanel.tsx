// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useMemo, useState } from "react";
import type { Artifacts } from "../../lib/artifacts";
import {
  admetPrompt,
  buildCandidates,
  type Candidate,
  candidatesToCsv,
  candidateValue,
  cofoldPrompt,
  MAX_COFOLD,
  numericColumns,
} from "../../lib/candidates";
import { MolDepiction } from "../MolDepiction/MolDepiction";
import { CandidateDetail } from "./CandidateDetail";
import { CandidateScatter } from "./CandidateScatter";
import { GateBadge } from "./GateBadge";
import { ScoreBar } from "./ScoreBar";

interface Props {
  artifacts: Artifacts;
  theme: "dark" | "light";
  onOpenMolecule: (smiles: string) => void;
  onOpenStructure: (path: string) => void;
  onAsk: (prompt: string) => void;
}

/** "name", "origin", "gate", or a numeric column key as in candidateValue. */
type SortKey = string;
interface Sort {
  key: SortKey;
  dir: 1 | -1;
}

/** Component columns shown in the table; the rest are in the drawer and the scatter plot. */
const TABLE_COMPONENTS = 4;
const GATE_RANK = { pass: 2, unknown: 1, fail: 0 } as const;

const control = "bg-sunken border-subtle rounded-md border px-2 py-1 text-xs";
const button =
  "border-subtle hover:bg-sunken inline-flex cursor-pointer items-center gap-1 rounded-md border px-2 py-1 text-[11px] font-medium";

function sortValue(c: Candidate, key: SortKey): string | number | undefined {
  if (key === "name") return c.name ?? c.smiles;
  if (key === "origin") return c.origin;
  if (key === "gate") return GATE_RANK[c.gate.status];
  return candidateValue(c, key);
}

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
 * SMILES, with a scatter view and a detail drawer.
 */
export function CandidatesPanel({ artifacts, theme, onOpenMolecule, onOpenStructure, onAsk }: Props) {
  const candidates = useMemo(() => buildCandidates(artifacts), [artifacts]);
  const columns = useMemo(() => numericColumns(candidates), [candidates]);
  const origins = useMemo(() => [...new Set(candidates.map((c) => c.origin))], [candidates]);
  const tableColumns = columns.filter((c) => c.key !== "score").slice(0, TABLE_COMPONENTS);
  const hasScore = columns.some((c) => c.key === "score");

  const [query, setQuery] = useState("");
  const [origin, setOrigin] = useState("");
  const [minScore, setMinScore] = useState("");
  const [gateOnly, setGateOnly] = useState(false);
  const [view, setView] = useState<"table" | "scatter">("table");
  const [sort, setSort] = useState<Sort>({ key: "score", dir: -1 });
  const [selected, setSelected] = useState<Set<string>>(() => new Set());
  /** undefined until the user picks or closes: the drawer then shows the first row. */
  const [detailKey, setDetailKey] = useState<string | null | undefined>(undefined);
  const [axes, setAxes] = useState<{ x: string; y: string } | null>(null);
  const [copied, setCopied] = useState(false);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const min = minScore.trim() === "" ? null : Number(minScore);
    const rows = candidates.filter(
      (c) =>
        (!q || [c.smiles, c.name ?? "", c.origins.join(" ")].join(" ").toLowerCase().includes(q)) &&
        (!origin || c.origin === origin) &&
        (min === null || Number.isNaN(min) || (c.score !== undefined && c.score >= min)) &&
        (!gateOnly || c.gate.status === "pass"),
    );
    return rows.sort((a, b) => {
      const va = sortValue(a, sort.key);
      const vb = sortValue(b, sort.key);
      // Missing values go last whichever way the column is sorted.
      if (va === undefined) return vb === undefined ? 0 : 1;
      if (vb === undefined) return -1;
      if (typeof va === "number" && typeof vb === "number") return (va - vb) * sort.dir;
      return String(va).localeCompare(String(vb)) * sort.dir;
    });
  }, [candidates, query, origin, minScore, gateOnly, sort]);

  const chosen = candidates.filter((c) => selected.has(c.smiles));
  const detailSmiles = detailKey === undefined ? (filtered[0]?.smiles ?? null) : detailKey;
  const detail = detailSmiles ? (candidates.find((c) => c.smiles === detailSmiles) ?? null) : null;

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

  return (
    <div className="bg-raised flex h-full min-h-0 flex-col lg:flex-row">
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <div className="border-subtle flex flex-wrap items-center gap-2 border-b px-3 py-2">
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search name, SMILES, tool…"
            aria-label="Search candidates"
            className={`${control} w-44 min-w-0 flex-1 sm:flex-none`}
          />
          <select aria-label="Origin" value={origin} onChange={(e) => setOrigin(e.target.value)} className={control}>
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
            className={button}
            style={gateOnly ? { borderColor: "#3ccf91", color: "#3ccf91" } : undefined}
          >
            Gate only
          </button>
          <div className="border-subtle ml-auto inline-flex overflow-hidden rounded-md border" role="group" aria-label="View">
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
        </div>

        {chosen.length > 0 && (
          <div className="border-subtle bg-sunken flex flex-wrap items-center gap-2 border-b px-3 py-1.5 text-[11px]">
            <span className="font-semibold">{chosen.length} selected</span>
            <button type="button" className={button} onClick={() => download("candidates.csv", candidatesToCsv(chosen), "text/csv")}>
              ⇩ CSV
            </button>
            <button type="button" className={button} onClick={copySmiles}>
              {copied ? "Copied" : "Copy SMILES"}
            </button>
            <button
              type="button"
              className={button}
              onClick={() => onAsk(cofoldPrompt(chosen))}
              title={chosen.length > MAX_COFOLD ? `Only the first ${MAX_COFOLD} are cofolded` : undefined}
            >
              Ask the agent to cofold{chosen.length > MAX_COFOLD ? ` first ${MAX_COFOLD}` : ""}
            </button>
            <button type="button" className={button} onClick={() => onAsk(admetPrompt(chosen))}>
              Ask for ADMET
            </button>
            <button type="button" className="text-muted hover:text-body ml-auto cursor-pointer" onClick={() => setSelected(new Set())}>
              Clear
            </button>
          </div>
        )}

        <p className="text-faint px-3 pt-1.5 text-[11px]">
          Showing {filtered.length} of {candidates.length}
          {chosen.length ? ` · ${chosen.length} selected` : ""}
        </p>

        <div className="min-h-0 flex-1 overflow-auto">
          {view === "scatter" ? (
            <CandidateScatter
              candidates={filtered}
              columns={columns}
              x={x}
              y={y}
              onAxes={(nx, ny) => setAxes({ x: nx, y: ny })}
              selected={selected}
              detail={detail?.smiles ?? null}
              onPick={setDetailKey}
            />
          ) : (
            <table className="w-full border-collapse text-xs">
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
                  {hasScore && header("score", "Score", "right")}
                  {tableColumns.map((c) => header(c.key, c.label, "right"))}
                  {header("origin", "Origin")}
                  {header("gate", "Gate")}
                </tr>
              </thead>
              <tbody>
                {filtered.map((c) => {
                  const isDetail = c.smiles === detail?.smiles;
                  return (
                    <tr
                      key={c.smiles}
                      onClick={() => setDetailKey(c.smiles)}
                      className={`hover:bg-sunken cursor-pointer ${selected.has(c.smiles) ? "bg-sunken" : ""}`}
                      style={isDetail ? { boxShadow: "inset 3px 0 0 var(--color-accent)" } : undefined}
                    >
                      <td className="border-subtle border-b px-2 py-1" onClick={(e) => e.stopPropagation()}>
                        <input
                          type="checkbox"
                          aria-label={`Select ${c.name ?? c.smiles}`}
                          checked={selected.has(c.smiles)}
                          onChange={() => toggle(c.smiles)}
                          className="cursor-pointer"
                        />
                      </td>
                      <td className="border-subtle border-b px-2 py-1">
                        <MolDepiction smiles={c.smiles} theme={theme} width={120} height={80} label={c.name} className="h-10 w-14" />
                      </td>
                      <td className="border-subtle max-w-[14rem] border-b px-2 py-1">
                        {c.name && <span className="block truncate font-medium">{c.name}</span>}
                        <span className="text-muted block truncate font-mono text-[10px]" title={c.smiles}>
                          {c.smiles}
                        </span>
                      </td>
                      {hasScore && (
                        <td className="border-subtle border-b px-2 py-1 text-right whitespace-nowrap">
                          <ScoreBar value={c.score} />
                        </td>
                      )}
                      {tableColumns.map((col) => {
                        const value = candidateValue(c, col.key);
                        return (
                          <td key={col.key} className="border-subtle border-b px-2 py-1 text-right font-mono whitespace-nowrap tabular-nums">
                            {value === undefined ? <span className="text-faint">—</span> : value.toFixed(Number.isInteger(value) ? 0 : 2)}
                          </td>
                        );
                      })}
                      <td className="border-subtle border-b px-2 py-1">
                        <span className="bg-inset text-muted rounded-full px-1.5 py-px text-[10px] whitespace-nowrap">{c.origin}</span>
                      </td>
                      <td className="border-subtle border-b px-2 py-1">
                        <GateBadge gate={c.gate} />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
          {filtered.length === 0 && (
            <p className="text-muted px-3 py-8 text-center text-xs">No matching candidates. Adjust the filters or clear the search.</p>
          )}
        </div>
      </div>

      {detail && (
        <div className="border-subtle flex max-h-[50%] min-h-0 shrink-0 flex-col border-t lg:max-h-none lg:w-80 lg:border-t-0 lg:border-l">
          <CandidateDetail
            candidate={detail}
            theme={theme}
            onClose={() => setDetailKey(null)}
            onOpenMolecule={onOpenMolecule}
            onOpenStructure={onOpenStructure}
            onAsk={onAsk}
          />
        </div>
      )}
    </div>
  );
}
