// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { memo, useEffect, useMemo, useRef, useState, type RefObject } from "react";
import type { Theme } from "../../hooks/useTheme";
import { formatEndpoint, heading, sortedIndices, TOTAL, type Endpoint, type Molecules } from "../../lib/runAnalysis";
import { MolDepiction } from "../MolDepiction/MolDepiction";
import { SmilesText } from "../MolDepiction/SmilesText";

interface Props {
  molecules: Molecules;
  mask: Uint8Array | null;
  /** The total and every component, in their order. */
  columns: readonly Endpoint[];
  colours: ReadonlyMap<string, string>;
  hovered: number | null;
  onHover: (index: number | null) => void;
  onOpen: (index: number) => void;
  /** Show the molecule's cofolded complex in the viewer. */
  onOpenStructure?: (index: number) => void;
  theme: Theme;
}

const PAGE = 20;
const THUMB = 88;

/**
 * The molecules the plots select (all when nothing is), best total first, a
 * page at a time: raw values in their units, each with a hairline of its
 * 0–1 score. Scrolls both ways inside its own box, the header and the
 * structures staying in view.
 */
export const SelectionTable = memo(function SelectionTable({ molecules, mask, columns, colours, hovered, onHover, onOpen, onOpenStructure, theme }: Props) {
  const complexes = !!onOpenStructure && molecules.rows.some((r) => r.structure);
  const [sort, setSort] = useState<{ key: string; desc: boolean }>({ key: TOTAL, desc: true });
  const [paging, setPaging] = useState<{ mask: Uint8Array | null; page: number }>({ mask, page: 0 });
  const scroller = useRef<HTMLDivElement>(null);
  const page = paging.mask === mask ? paging.page : 0;
  const rows = useMemo(() => sortedIndices(molecules, mask, sort.key, sort.desc), [molecules, mask, sort]);
  const pages = Math.max(1, Math.ceil(rows.length / PAGE));
  const at = Math.min(page, pages - 1);
  const shown = rows.slice(at * PAGE, at * PAGE + PAGE);
  const go = (p: number) => {
    setPaging({ mask, page: Math.max(0, Math.min(pages - 1, p)) });
    scroller.current?.scrollTo({ top: 0 });
  };
  const header = (key: string, label: string, unit: string, title: string) => (
    <th key={key} scope="col" className="bg-raised border-subtle sticky top-0 z-10 border-b p-0 align-bottom font-medium" style={{ minWidth: 84 }}>
      <button
        type="button"
        onClick={() => setSort(sort.key === key ? { key, desc: !sort.desc } : { key, desc: key !== "step" })}
        title={title}
        className={`flex w-full cursor-pointer flex-col items-end px-2 py-1 text-right leading-tight ${sort.key === key ? "text-body" : "text-muted hover:text-body"}`}
      >
        <span className="whitespace-nowrap">
          {label}
          {sort.key === key && <span className="ml-0.5 text-[8px]">{sort.desc ? "▼" : "▲"}</span>}
        </span>
        <span className="text-faint text-[9.5px] font-normal whitespace-nowrap">{unit || "\u00a0"}</span>
      </button>
    </th>
  );

  return (
    <div className="text-[11px]">
      <div ref={scroller} className="border-subtle max-h-[560px] overflow-auto rounded-md border">
        <table className="w-full border-separate border-spacing-0">
          <thead>
            <tr>
              <th scope="col" className="bg-raised border-subtle sticky top-0 left-0 z-20 border-b" style={{ width: THUMB, minWidth: THUMB }}>
                <span className="sr-only">Structure</span>
              </th>
              {complexes && (
                <th scope="col" className="bg-raised border-subtle sticky top-0 z-10 border-b" style={{ width: 36 }}>
                  <span className="sr-only">Complex</span>
                </th>
              )}
              <th scope="col" className="bg-raised border-subtle text-muted sticky top-0 z-10 border-b px-2 py-1 text-left align-bottom font-medium" style={{ minWidth: 180 }}>
                SMILES
              </th>
              {header("step", "Step", "", "The step of its best score")}
              {columns.map((c) => header(c.key, c.label, c.source === "score" ? "score" : c.unit, `${heading(c)}${c.key !== c.label ? ` · ${c.key}` : ""}${c.source === "score" ? " · 0–1 score: no raw value in the CSV" : c.source === "raw" ? " · raw value; the bar is its 0–1 score" : ""}`))}
            </tr>
          </thead>
          <tbody>
            {shown.length === 0 ? (
              <tr>
                <td colSpan={columns.length + (complexes ? 4 : 3)} className="text-faint py-8 text-center">
                  No molecule matches the selection.
                </td>
              </tr>
            ) : (
              shown.map((i) => {
                const active = hovered === i;
                const cell = `border-subtle border-b ${active ? "bg-inset" : "group-hover:bg-sunken"}`;
                return (
                  <tr key={molecules.smiles[i]} onClick={() => onOpen(i)} onPointerEnter={() => onHover(i)} onPointerLeave={() => onHover(null)} title="Open the molecule" className="group cursor-pointer">
                    <td className={`${cell} sticky left-0 z-[5] p-0 ${active ? "" : "bg-raised"}`} style={{ width: THUMB, minWidth: THUMB }}>
                      <LazyDepiction smiles={molecules.smiles[i]} theme={theme} root={scroller} />
                    </td>
                    {complexes && (
                      <td className={`${cell} px-1 text-center`}>
                        {molecules.rows[i]?.structure && (
                          <button
                            type="button"
                            onClick={(event) => {
                              event.stopPropagation();
                              onOpenStructure?.(i);
                            }}
                            title="Show the complex in the viewer"
                            className="hover:bg-inset cursor-pointer rounded px-1.5 py-0.5 text-[10px] font-semibold"
                            style={{ color: "var(--color-structure)" }}
                          >
                            3D
                          </button>
                        )}
                      </td>
                    )}
                    <td className={`${cell} max-w-[260px] px-2`}>
                      <SmilesText smiles={molecules.smiles[i]} className="text-[10px]" />
                    </td>
                    <td className={`${cell} text-muted px-2 text-right font-mono tabular-nums`}>{molecules.step[i]}</td>
                    {columns.map((c) => {
                      const v = molecules.values.get(c.key)?.[i] ?? NaN;
                      const score = c.source === "raw" ? (molecules.scores.get(c.key)?.[i] ?? NaN) : NaN;
                      return (
                        <td key={c.key} className={`${cell} px-2 text-right font-mono whitespace-nowrap tabular-nums ${c.key === TOTAL ? "text-body font-semibold" : ""}`}>
                          {formatEndpoint(c, v)}
                          {Number.isFinite(score) && (
                            <span className="bg-inset mt-0.5 ml-auto block h-[2px] w-10 overflow-hidden rounded-full" aria-hidden="true" title={`Score ${score.toFixed(2)}`}>
                              <span className="block h-full rounded-full" style={{ width: `${Math.round(Math.min(1, Math.max(0, score)) * 100)}%`, background: colours.get(c.key), opacity: 0.55 }} />
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
      </div>
      {rows.length > PAGE && (
        <div className="text-faint mt-1.5 flex items-center justify-end gap-1 tabular-nums">
          <span>
            {(at * PAGE + 1).toLocaleString()}–{Math.min(rows.length, (at + 1) * PAGE).toLocaleString()} of {rows.length.toLocaleString()}
          </span>
          <PageButton label="Previous page" disabled={at === 0} onClick={() => go(at - 1)}>
            ‹
          </PageButton>
          <PageButton label="Next page" disabled={at >= pages - 1} onClick={() => go(at + 1)}>
            ›
          </PageButton>
        </div>
      )}
    </div>
  );
});

/** A structure drawn once it scrolls near the view. */
function LazyDepiction({ smiles, theme, root }: { smiles: string; theme: Theme; root: RefObject<HTMLDivElement | null> }) {
  const box = useRef<HTMLDivElement>(null);
  const [seen, setSeen] = useState(() => typeof IntersectionObserver === "undefined");
  useEffect(() => {
    const element = box.current;
    if (!element || seen) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) setSeen(true);
      },
      { root: root.current, rootMargin: "200px" },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, [seen, root]);
  return (
    <div ref={box} className="grid h-16 place-items-center" style={{ width: THUMB }}>
      {seen && <MolDepiction smiles={smiles} theme={theme} width={176} height={128} className="h-16 w-[88px]" />}
    </div>
  );
}

function PageButton({ label, disabled, onClick, children }: { label: string; disabled: boolean; onClick: () => void; children: string }) {
  return (
    <button
      type="button"
      aria-label={label}
      disabled={disabled}
      onClick={onClick}
      className="hover:bg-inset hover:text-body grid size-6 cursor-pointer place-items-center rounded text-sm disabled:cursor-default disabled:opacity-40"
    >
      {children}
    </button>
  );
}
