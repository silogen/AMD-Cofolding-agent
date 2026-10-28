// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { memo, useMemo, useState } from "react";
import type { Theme } from "../../hooks/useTheme";
import { sortedIndices, TOTAL, type Endpoint, type Molecules } from "../../lib/runAnalysis";
import { MolDepiction } from "../MolDepiction/MolDepiction";
import { SmilesText } from "../MolDepiction/SmilesText";
import { short } from "./plotUtils";

interface Props {
  molecules: Molecules;
  mask: Uint8Array | null;
  /** Score columns after the total: the components with the highest weight. */
  columns: readonly Endpoint[];
  colours: ReadonlyMap<string, string>;
  hovered: number | null;
  onHover: (index: number | null) => void;
  onOpen: (index: number) => void;
  theme: Theme;
}

const PAGE = 20;

/** The molecules the plots select (all when nothing is), best total first, a page at a time. */
export const SelectionTable = memo(function SelectionTable({ molecules, mask, columns, colours, hovered, onHover, onOpen, theme }: Props) {
  const [sort, setSort] = useState<{ key: string; desc: boolean }>({ key: TOTAL, desc: true });
  const [paging, setPaging] = useState<{ mask: Uint8Array | null; page: number }>({ mask, page: 0 });
  const page = paging.mask === mask ? paging.page : 0;
  const rows = useMemo(() => sortedIndices(molecules, mask, sort.key, sort.desc), [molecules, mask, sort]);
  const pages = Math.max(1, Math.ceil(rows.length / PAGE));
  const at = Math.min(page, pages - 1);
  const shown = rows.slice(at * PAGE, at * PAGE + PAGE);
  const scoreColumns = [{ key: TOTAL, label: "Total" }, ...columns.filter((c) => c.key !== TOTAL)];
  const template = `64px minmax(120px,1fr) 44px ${scoreColumns.map(() => "minmax(56px,80px)").join(" ")}`;
  const go = (p: number) => setPaging({ mask, page: Math.max(0, Math.min(pages - 1, p)) });
  const header = (key: string, label: string, title?: string) => (
    <button
      type="button"
      onClick={() => setSort(sort.key === key ? { key, desc: !sort.desc } : { key, desc: key !== "step" })}
      title={title ?? label}
      className={`flex min-w-0 cursor-pointer items-center justify-end gap-0.5 px-1.5 text-right ${sort.key === key ? "text-body" : "hover:text-body"}`}
    >
      <span className="truncate">{label}</span>
      {sort.key === key && <span className="shrink-0 text-[8px]">{sort.desc ? "▼" : "▲"}</span>}
    </button>
  );

  return (
    <div className="text-[11px]">
      <div className="overflow-x-auto">
        <div style={{ minWidth: 64 + 120 + 44 + scoreColumns.length * 56 }}>
          <div className="text-muted border-subtle grid h-7 items-center border-b font-medium" style={{ gridTemplateColumns: template }}>
            <span />
            <span className="px-1.5">SMILES</span>
            {header("step", "Step")}
            {scoreColumns.map((c) => header(c.key, c.label, c.key))}
          </div>
          {shown.length === 0 ? (
            <p className="text-faint py-8 text-center">No molecule matches the selection.</p>
          ) : (
            shown.map((i) => (
              <div
                key={molecules.smiles[i]}
                role="row"
                onClick={() => onOpen(i)}
                onPointerEnter={() => onHover(i)}
                onPointerLeave={() => onHover(null)}
                title="Open the molecule"
                className={`group border-subtle grid h-12 cursor-pointer items-center border-b last:border-b-0 ${hovered === i ? "bg-inset" : "hover:bg-sunken"}`}
                style={{ gridTemplateColumns: template }}
              >
                <MolDepiction smiles={molecules.smiles[i]} theme={theme} width={128} height={88} className="h-11 w-16" />
                <span className="min-w-0 px-1.5">
                  <SmilesText smiles={molecules.smiles[i]} className="text-[10px]" />
                </span>
                <span className="text-muted px-1.5 text-right font-mono tabular-nums">{molecules.step[i]}</span>
                {scoreColumns.map((c) => {
                  const v = molecules.values.get(c.key)?.[i] ?? NaN;
                  return (
                    <span key={c.key} className={`flex items-center justify-end gap-1 px-1.5 font-mono tabular-nums ${c.key === TOTAL ? "font-semibold" : ""}`}>
                      {c.key !== TOTAL && Number.isFinite(v) && v >= 0 && v <= 1 && (
                        <span className="bg-inset hidden h-1 w-6 overflow-hidden rounded-full @lg:inline-block" aria-hidden="true">
                          <span className="block h-full rounded-full" style={{ width: `${Math.round(v * 100)}%`, background: colours.get(c.key) }} />
                        </span>
                      )}
                      {short(v)}
                    </span>
                  );
                })}
              </div>
            ))
          )}
        </div>
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
