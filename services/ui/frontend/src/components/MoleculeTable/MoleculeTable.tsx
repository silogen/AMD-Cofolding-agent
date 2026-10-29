// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useRef, useState, type MouseEvent, type ReactNode, type RefObject } from "react";
import type { Theme } from "../../hooks/useTheme";
import { CopyButton } from "../CopyButton/CopyButton";
import { MolDepiction } from "../MolDepiction/MolDepiction";

export interface MoleculeColumn {
  key: string;
  label: string;
  /** Under the label, e.g. the unit. */
  unit?: string;
  title?: string;
  /** Bold, as the total is. */
  strong?: boolean;
  /** Faint, for bookkeeping such as the step. */
  muted?: boolean;
  /** The colour of the column's score bars. */
  colour?: string;
  /** Words rather than numbers, such as a name: left-aligned in the body font. */
  text?: boolean;
}

export interface MoleculeCell {
  text: string;
  /** A 0–1 score, drawn as a hairline under the value. */
  score?: number | null;
  title?: string;
  /** A faint second line in a `text` column, e.g. where the molecule came from. */
  sub?: string;
}

export interface MoleculeRow {
  smiles: string;
  cells: Readonly<Record<string, MoleculeCell | undefined>>;
  /** The 3D button's tooltip; no button without it. */
  complex?: string;
}

interface Props<R extends MoleculeRow> {
  columns: readonly MoleculeColumn[];
  rows: readonly R[];
  theme: Theme;
  /** Clickable headers when given. */
  sort?: { key: string; desc: boolean };
  onSort?: (key: string) => void;
  hovered?: string | null;
  onHover?: (smiles: string | null) => void;
  onOpen?: (row: R) => void;
  onOpenStructure?: (row: R) => void;
  /** Buttons beside the copy one, e.g. a star. */
  actions?: (row: R) => ReactNode;
  empty?: ReactNode;
  /** Scrolls back to the top when it changes, e.g. the page. */
  scrollKey?: unknown;
  /** Null to grow with the parent instead, which then gives it its height. */
  maxHeight?: number | null;
  /** Below the rows, inside the scrolling box, e.g. a "Show more" button. */
  footer?: ReactNode;
}

const THUMB = 88;
const THUMB_CELL = THUMB + 24;

/**
 * Molecules as rows of a structure and one column per endpoint: the value in
 * its units over a hairline of its 0–1 score. Scrolls both ways inside its
 * own box, the header and the structures staying in view.
 */
export function MoleculeTable<R extends MoleculeRow>({ columns, rows, theme, sort, onSort, hovered, onHover, onOpen, onOpenStructure, actions, empty, scrollKey, maxHeight = 560, footer }: Props<R>) {
  const scroller = useRef<HTMLDivElement>(null);
  const complexes = !!onOpenStructure && rows.some((r) => r.complex);

  useEffect(() => {
    scroller.current?.scrollTo?.({ top: 0 });
  }, [scrollKey]);

  return (
    <div
      ref={scroller}
      className={`border-subtle overflow-auto rounded-md border text-[11px] ${maxHeight === null ? "min-h-0 flex-1" : ""}`}
      style={maxHeight === null ? undefined : { maxHeight }}
    >
      <table className="w-full border-separate border-spacing-0">
        <thead>
          <tr>
            <th scope="col" className="bg-raised border-subtle sticky top-0 left-0 z-20 border-b" style={{ width: THUMB_CELL, minWidth: THUMB_CELL }}>
              <span className="sr-only">Structure</span>
            </th>
            {complexes && (
              <th scope="col" className="bg-raised border-subtle sticky top-0 z-10 border-b" style={{ width: 36 }}>
                <span className="sr-only">Complex</span>
              </th>
            )}
            {columns.map((c) => (
              <th key={c.key} scope="col" className="bg-raised border-subtle sticky top-0 z-10 border-b p-0 align-bottom font-medium" style={{ minWidth: c.text ? 112 : 84 }}>
                <HeaderLabel column={c} sort={sort} onSort={onSort} />
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr>
              <td colSpan={columns.length + (complexes ? 2 : 1)} className="text-faint py-8 text-center">
                {empty ?? "No molecules."}
              </td>
            </tr>
          ) : (
            rows.map((row) => {
              const active = hovered === row.smiles;
              const cell = `border-subtle border-b ${active ? "bg-inset" : "group-hover:bg-sunken"}`;
              return (
                <tr
                  key={row.smiles}
                  onClick={onOpen ? () => onOpen(row) : undefined}
                  onPointerEnter={onHover ? () => onHover(row.smiles) : undefined}
                  onPointerLeave={onHover ? () => onHover(null) : undefined}
                  title={onOpen ? "Open the molecule" : undefined}
                  className={`group ${onOpen ? "cursor-pointer" : ""}`}
                >
                  <td className={`${cell} sticky left-0 z-[5] p-0 ${active ? "" : "bg-raised"}`} style={{ width: THUMB_CELL, minWidth: THUMB_CELL }}>
                    <MoleculeThumb smiles={row.smiles} theme={theme} root={scroller} actions={actions?.(row)} />
                  </td>
                  {complexes && (
                    <td className={`${cell} px-1 text-center`}>
                      {row.complex && (
                        <StructureButton
                          title={row.complex}
                          onClick={(event) => {
                            event.stopPropagation();
                            onOpenStructure?.(row);
                          }}
                        />
                      )}
                    </td>
                  )}
                  {columns.map((c) => {
                    const value = row.cells[c.key];
                    const score = value?.score;
                    if (c.text) {
                      return (
                        <td key={c.key} title={value?.title} className={`${cell} max-w-44 px-2 text-left ${c.strong ? "font-semibold" : ""}`}>
                          {value?.text && <span className="block truncate font-medium">{value.text}</span>}
                          {value?.sub && <span className="text-faint block truncate text-[10px]">{value.sub}</span>}
                        </td>
                      );
                    }
                    return (
                      <td
                        key={c.key}
                        title={value?.title}
                        className={`${cell} px-2 text-right font-mono whitespace-nowrap tabular-nums ${c.strong ? "text-body font-semibold" : c.muted ? "text-muted" : ""}`}
                      >
                        {value?.text ?? "—"}
                        {score != null && Number.isFinite(score) && (
                          <span className="bg-inset mt-0.5 ml-auto block h-[2px] w-10 overflow-hidden rounded-full" aria-hidden="true" title={`Score ${score.toFixed(2)}`}>
                            <span className="block h-full rounded-full" style={{ width: `${Math.round(Math.min(1, Math.max(0, score)) * 100)}%`, background: c.colour ?? "var(--color-structure)", opacity: 0.55 }} />
                          </span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              );
            })
          )}
        </tbody>
      </table>
      {footer}
    </div>
  );
}

function HeaderLabel({ column, sort, onSort }: { column: MoleculeColumn; sort?: { key: string; desc: boolean }; onSort?: (key: string) => void }) {
  const active = sort?.key === column.key;
  const body = (
    <>
      <span className="whitespace-nowrap">
        {column.label}
        {active && <span className="ml-0.5 text-[8px]">{sort.desc ? "▼" : "▲"}</span>}
      </span>
      <span className="text-faint text-[9.5px] font-normal whitespace-nowrap">{column.unit || "\u00a0"}</span>
    </>
  );
  const layout = `flex w-full flex-col px-2 py-1 leading-tight ${column.text ? "items-start text-left" : "items-end text-right"}`;
  if (!onSort) {
    return (
      <span title={column.title} className={`text-muted ${layout}`}>
        {body}
      </span>
    );
  }
  return (
    <button
      type="button"
      onClick={() => onSort(column.key)}
      title={column.title}
      aria-sort={active ? (sort.desc ? "descending" : "ascending") : undefined}
      className={`${layout} cursor-pointer ${active ? "text-body" : "text-muted hover:text-body"}`}
    >
      {body}
    </button>
  );
}

export function StructureButton({ title, onClick }: { title: string; onClick: (event: MouseEvent) => void }) {
  return (
    <button type="button" onClick={onClick} title={title} className="hover:bg-inset cursor-pointer rounded px-1.5 py-0.5 text-[10px] font-semibold" style={{ color: "var(--color-structure)" }}>
      3D
    </button>
  );
}

/**
 * A molecule's 2D drawing, its SMILES on hover, with a copy button and any
 * other actions in a narrow column beside it. Given a scrolling root, the
 * drawing waits until it scrolls near the view.
 */
export function MoleculeThumb({
  smiles,
  theme,
  root,
  actions,
  width = THUMB,
  height = 64,
  dim,
}: {
  smiles: string;
  theme: Theme;
  root?: RefObject<HTMLElement | null>;
  actions?: ReactNode;
  width?: number;
  height?: number;
  dim?: boolean;
}) {
  const box = useRef<HTMLSpanElement>(null);
  const [seen, setSeen] = useState(() => !root || typeof IntersectionObserver === "undefined");
  useEffect(() => {
    const element = box.current;
    if (!element || seen) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) setSeen(true);
      },
      { root: root?.current, rootMargin: "200px" },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, [seen, root]);
  return (
    <span className="flex h-full items-center">
      <span ref={box} title={smiles || undefined} className={`grid shrink-0 place-items-center ${dim ? "opacity-60" : ""}`} style={{ width, height }}>
        {seen && smiles && <MolDepiction smiles={smiles} theme={theme} width={width * 2} height={height * 2} className="size-full" />}
      </span>
      <span className="flex w-6 shrink-0 flex-col items-center gap-0.5">
        {smiles && <CopyButton value={smiles} label="Copy SMILES" icon />}
        {actions}
      </span>
    </span>
  );
}
