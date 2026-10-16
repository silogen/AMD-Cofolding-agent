// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { StructurePanel } from "../../types/api";
import type { Theme } from "../../hooks/useTheme";
import { StructureViewer } from "../StructureViewer/StructureViewer";

interface Props {
  panel: StructurePanel;
  theme: Theme;
  onOpen: (panel: StructurePanel) => void;
  isOpen: boolean;
}

/**
 * A protein-ligand complex, inline in the conversation.
 *
 * The inline view is small on purpose: it tells the user a structure was
 * produced at this point in the conversation. "Open in viewer" moves it to
 * the workspace panel, where there is room to rotate and inspect it.
 */
export function StructureCard({ panel, theme, onOpen, isOpen }: Props) {
  return (
    <figure className="bg-raised border-subtle shadow-card m-0 overflow-hidden rounded-xl border">
      <figcaption className="border-subtle flex items-center gap-2 border-b px-3 py-2">
        <span
          aria-hidden="true"
          className="size-2 rounded-full"
          style={{ background: "var(--color-structure)" }}
        />
        <span className="truncate text-xs font-semibold" title={panel.path}>
          {panel.name}
        </span>
        <span className="text-faint ml-auto text-[11px] uppercase">{panel.format}</span>
      </figcaption>

      <StructureViewer panel={panel} theme={theme} style="cartoon" className="h-56" />

      <div className="border-subtle flex items-center gap-2 border-t px-3 py-2">
        <button
          type="button"
          onClick={() => onOpen(panel)}
          className="cursor-pointer rounded-md px-2 py-1 text-[11px] font-semibold text-white transition-opacity hover:opacity-90 touch:min-h-11 touch:px-3"
          style={{ background: "var(--color-accent)" }}
        >
          {isOpen ? "Showing in viewer" : "Open in viewer"}
        </button>
        <a
          href={panel.url}
          download={panel.name}
          className="text-muted hover:text-body bg-inset ml-auto inline-flex items-center rounded-md px-2 py-1 text-[11px] font-medium touch:min-h-11 touch:px-3"
        >
          Download
        </a>
      </div>
    </figure>
  );
}
