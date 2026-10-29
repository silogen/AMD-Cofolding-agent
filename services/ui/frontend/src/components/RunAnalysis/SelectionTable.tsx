// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { memo, useMemo, useState } from "react";
import type { Theme } from "../../hooks/useTheme";
import { sortedIndices, TOTAL, type Endpoint, type Molecules } from "../../lib/runAnalysis";
import { endpointCells, endpointColumn } from "../MoleculeTable/columns";
import { MoleculeTable, type MoleculeColumn, type MoleculeRow } from "../MoleculeTable/MoleculeTable";

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

interface Row extends MoleculeRow {
  index: number;
}

/**
 * The molecules the plots select (all when nothing is), best total first, a
 * page at a time: raw values in their units, each with a hairline of its
 * 0–1 score.
 */
export const SelectionTable = memo(function SelectionTable({ molecules, mask, columns, colours, hovered, onHover, onOpen, onOpenStructure, theme }: Props) {
  const [sort, setSort] = useState<{ key: string; desc: boolean }>({ key: TOTAL, desc: true });
  const [paging, setPaging] = useState<{ mask: Uint8Array | null; page: number }>({ mask, page: 0 });
  const page = paging.mask === mask ? paging.page : 0;
  const order = useMemo(() => sortedIndices(molecules, mask, sort.key, sort.desc), [molecules, mask, sort]);
  const pages = Math.max(1, Math.ceil(order.length / PAGE));
  const at = Math.min(page, pages - 1);
  const go = (p: number) => setPaging({ mask, page: Math.max(0, Math.min(pages - 1, p)) });

  const tableColumns = useMemo<MoleculeColumn[]>(
    () => [{ key: "step", label: "Step", title: "The step of its best score", muted: true }, ...columns.map((c) => endpointColumn(c, colours.get(c.key)))],
    [columns, colours],
  );
  const rows = order.slice(at * PAGE, at * PAGE + PAGE).map(
    (i): Row => ({
      index: i,
      smiles: molecules.smiles[i],
      cells: { step: { text: String(molecules.step[i]) }, ...endpointCells(columns, molecules, i) },
      complex: molecules.rows[i]?.structure ? "Show the complex in the viewer" : undefined,
    }),
  );

  return (
    <div className="text-[11px]">
      <MoleculeTable
        columns={tableColumns}
        rows={rows}
        theme={theme}
        sort={sort}
        onSort={(key) => setSort(sort.key === key ? { key, desc: !sort.desc } : { key, desc: key !== "step" })}
        hovered={hovered === null ? null : molecules.smiles[hovered]}
        onHover={(smiles) => onHover(smiles === null ? null : (molecules.index.get(smiles) ?? null))}
        onOpen={(row) => onOpen(row.index)}
        onOpenStructure={onOpenStructure && ((row) => onOpenStructure(row.index))}
        empty="No molecule matches the selection."
        scrollKey={at}
      />
      {order.length > PAGE && (
        <div className="text-faint mt-1.5 flex items-center justify-end gap-1 tabular-nums">
          <span>
            {(at * PAGE + 1).toLocaleString()}–{Math.min(order.length, (at + 1) * PAGE).toLocaleString()} of {order.length.toLocaleString()}
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
