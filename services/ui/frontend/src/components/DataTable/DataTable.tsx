// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { TablePanel } from "../../types/api";
import { ExportButtons } from "./ExportButtons";

interface Props {
  panel: TablePanel;
}

type SortDirection = "asc" | "desc";

/** How many rows to show before the user asks for the rest. */
const PAGE = 25;

/**
 * Track whether a scroll container has content hidden to the right.
 *
 * On a narrow screen a wide table simply looks like it ends, and the columns
 * past the edge are never found. This drives a visible hint instead.
 */
function useHasMoreToTheRight(): [React.RefObject<HTMLDivElement | null>, boolean] {
  const ref = useRef<HTMLDivElement>(null);
  const [more, setMore] = useState(false);

  const measure = useCallback(() => {
    const node = ref.current;
    if (!node) return;
    setMore(node.scrollWidth - node.scrollLeft - node.clientWidth > 4);
  }, []);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    measure();
    node.addEventListener("scroll", measure, { passive: true });
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    return () => {
      node.removeEventListener("scroll", measure);
      observer.disconnect();
    };
  }, [measure]);

  return [ref, more];
}

/**
 * Predictions and measurements per molecule.
 *
 * Numbers are what a decision is made on, so the table is sortable and the
 * number columns are right-aligned and monospaced: that is what makes two
 * affinity values comparable at a glance.
 */
export function DataTable({ panel }: Props) {
  const [sort, setSort] = useState<{ column: string; direction: SortDirection } | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [scrollRef, hasMoreToTheRight] = useHasMoreToTheRight();

  const numericColumns = useMemo(() => {
    const numeric = new Set<string>();
    for (const column of panel.columns) {
      const values = panel.rows.map((row) => row[column]).filter((value) => value !== null);
      if (values.length > 0 && values.every((value) => typeof value === "number")) {
        numeric.add(column);
      }
    }
    return numeric;
  }, [panel]);

  const sorted = useMemo(() => {
    if (!sort) return panel.rows;
    const factor = sort.direction === "asc" ? 1 : -1;
    return [...panel.rows].sort((left, right) => {
      const a = left[sort.column];
      const b = right[sort.column];
      // Missing values sort last whichever way the column is sorted, so an
      // empty cell never looks like the best result.
      if (a === null || a === undefined) return 1;
      if (b === null || b === undefined) return -1;
      if (typeof a === "number" && typeof b === "number") return (a - b) * factor;
      return String(a).localeCompare(String(b)) * factor;
    });
  }, [panel.rows, sort]);

  const visible = expanded ? sorted : sorted.slice(0, PAGE);
  const hidden = sorted.length - visible.length;

  function toggleSort(column: string) {
    setSort((previous) => {
      if (previous?.column !== column) return { column, direction: "asc" };
      if (previous.direction === "asc") return { column, direction: "desc" };
      return null;
    });
  }

  return (
    <div className="bg-raised border-subtle shadow-card overflow-hidden rounded-xl border">
      <div className="border-subtle flex items-center gap-2 border-b px-3 py-2">
        <span
          aria-hidden="true"
          className="size-2 rounded-full"
          style={{ background: "var(--color-table)" }}
        />
        <span className="text-xs font-semibold">Results</span>
        <span className="text-faint ml-auto text-[11px]">
          {panel.totalRows} {panel.totalRows === 1 ? "row" : "rows"}
          {panel.truncated ? ` (first ${panel.rows.length} shown)` : ""}
        </span>
        <ExportButtons
          name={panel.tool}
          sheet={() => ({ columns: panel.columns, rows: sorted })}
          scope={panel.truncated ? `the ${panel.rows.length} rows received` : `all ${panel.rows.length} rows`}
        />
      </div>

      <div className="relative">
        {/* A fade at the edge, plus a word, because a fade alone is easy to
            miss and says nothing to a screen reader. */}
        {hasMoreToTheRight && (
          <>
            <div
              aria-hidden="true"
              className="pointer-events-none absolute inset-y-0 right-0 z-20 w-10"
              style={{
                background: "linear-gradient(to right, transparent, var(--surface-raised))",
              }}
            />
            <p className="text-faint absolute right-2 bottom-1 z-20 text-[10px] sm:hidden">
              more columns →
            </p>
          </>
        )}

        <div ref={scrollRef} className="max-h-96 overflow-auto">
          <table className="w-full border-collapse text-xs">
          <thead className="bg-sunken sticky top-0 z-10">
            <tr>
              {panel.columns.map((column) => {
                const active = sort?.column === column;
                return (
                  <th
                    key={column}
                    scope="col"
                    aria-sort={active ? (sort.direction === "asc" ? "ascending" : "descending") : "none"}
                    className={`border-subtle border-b px-2 py-2 font-semibold whitespace-nowrap sm:px-3 ${
                      numericColumns.has(column) ? "text-right" : "text-left"
                    }`}
                  >
                    <button
                      type="button"
                      onClick={() => toggleSort(column)}
                      // Fill the header cell so the whole column heading is
                      // tappable, not just the few characters of its label.
                      className={`hover:text-body inline-flex w-full cursor-pointer items-center gap-1 touch:min-h-11 ${
                        numericColumns.has(column) ? "justify-end" : "justify-start"
                      }`}
                      style={{ color: active ? "var(--color-accent)" : undefined }}
                    >
                      {column}
                      <span aria-hidden="true" className="text-[9px]">
                        {active ? (sort.direction === "asc" ? "▲" : "▼") : "↕"}
                      </span>
                    </button>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {visible.map((row, index) => (
              <tr key={index} className="hover:bg-sunken">
                {panel.columns.map((column) => {
                  const numeric = numericColumns.has(column);
                  return (
                    <td
                      key={column}
                      className={`border-subtle border-b px-2 py-1.5 sm:px-3 ${
                        numeric
                          ? "text-right font-mono tabular-nums"
                          : // Every text column is kept to one line. Real tool
                            // results put whole sentences in a cell — a scoring
                            // component's description, for example — and a
                            // wrapped paragraph makes a row tall enough to push
                            // the rest of the table off the screen. The full
                            // value stays available on hover and in the raw
                            // tool output. SMILES get less room on a phone so
                            // the numbers, which are the point of the table,
                            // stay visible.
                            `max-w-[10rem] truncate sm:max-w-[22rem] ${
                              column === "smiles" ? "max-w-[7rem] font-mono sm:max-w-[18rem]" : ""
                            }`
                      }`}
                      title={numeric ? undefined : String(row[column] ?? "")}
                    >
                      {formatCell(row[column])}
                    </td>
                  );
                })}
              </tr>
            ))}
            </tbody>
          </table>
        </div>
      </div>

      {hidden > 0 && (
        <button
          type="button"
          onClick={() => setExpanded(true)}
          className="text-muted hover:text-body border-subtle w-full cursor-pointer border-t py-2 text-[11px] font-medium touch:min-h-11"
        >
          Show {hidden} more {hidden === 1 ? "row" : "rows"}
        </button>
      )}
    </div>
  );
}

/** Render a cell so that missing data is visibly missing, not an empty cell. */
function formatCell(value: string | number | boolean | null | undefined) {
  if (value === null || value === undefined) {
    return <span className="text-faint">—</span>;
  }
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "number") {
    if (Number.isInteger(value)) return value.toString();
    // Four decimals keeps scores and affinities comparable without the noise
    // of full float precision.
    return value.toFixed(4);
  }
  return value;
}
