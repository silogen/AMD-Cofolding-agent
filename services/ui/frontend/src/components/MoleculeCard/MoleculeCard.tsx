// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { MoleculePanel } from "../../types/api";
import type { Theme } from "../../hooks/useTheme";
import { CopyButton } from "../CopyButton/CopyButton";
import { MolDepiction } from "../MolDepiction/MolDepiction";

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
        <CopyButton value={panel.smiles} label="Copy SMILES" />
      </div>
    </figure>
  );
}
