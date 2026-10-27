// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { MoleculePanel } from "../../types/api";
import type { Theme } from "../../hooks/useTheme";
import { openSketcher } from "../../lib/sketcherBus";
import { CopyButton } from "../CopyButton/CopyButton";
import { MolDepiction } from "../MolDepiction/MolDepiction";
import { StarButton } from "../Shortlist/StarButton";

interface Props {
  panel: MoleculePanel;
  theme: Theme;
  onOpen?: (smiles: string) => void;
}

/**
 * A ligand drawn as a 2D graph.
 *
 * REINVENT designs molecules as SMILES, which no chemist reads fluently as
 * text. The card shows the structure and keeps the SMILES available to copy
 * into another tool.
 */
export function MoleculeCard({ panel, theme, onOpen }: Props) {
  return (
    <figure className="bg-raised border-subtle shadow-card m-0 overflow-hidden rounded-xl border">
      <figcaption className="border-subtle flex items-center gap-2 border-b px-3 py-2">
        <span
          aria-hidden="true"
          className="size-2 rounded-full"
          style={{ background: "var(--color-molecule)" }}
        />
        <span className="truncate text-xs font-semibold">{panel.label ?? "Ligand"}</span>
        <span className="text-faint ml-auto text-[11px] whitespace-nowrap">2D structure</span>
        <StarButton smiles={panel.smiles} label={panel.label} source={panel.tool} className="-mr-1" />
      </figcaption>

      <button
        type="button"
        disabled={!onOpen}
        onClick={() => onOpen?.(panel.smiles)}
        title={onOpen ? "Open in the workbench" : undefined}
        className="bg-sunken block h-44 w-full p-2 enabled:cursor-pointer"
      >
        <MolDepiction smiles={panel.smiles} theme={theme} label={panel.label ?? undefined} className="h-full w-full" />
      </button>

      <div className="border-subtle flex items-center gap-2 border-t px-3 py-2">
        <code className="text-muted min-w-0 flex-1 truncate font-mono text-[11px]" title={panel.smiles}>
          {panel.smiles}
        </code>
        <button
          type="button"
          onClick={() => openSketcher(panel.smiles)}
          title="Edit in the sketcher; Use puts the result in the message"
          className="text-muted hover:text-body hover:bg-inset shrink-0 cursor-pointer rounded-md px-2 py-1 text-[11px] transition-colors touch:min-h-11 touch:px-3"
        >
          Edit in sketcher
        </button>
        <CopyButton value={panel.smiles} label="Copy SMILES" />
      </div>
    </figure>
  );
}
