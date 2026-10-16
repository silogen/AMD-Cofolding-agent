// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect } from "react";
import type { Panel, StructurePanel } from "../../types/api";
import type { Theme } from "../../hooks/useTheme";
import { StructureViewer, type ViewStyle } from "../StructureViewer/StructureViewer";

/**
 * ``sidebar`` sits beside the chat on a wide screen. ``sheet`` covers the
 * screen on a narrow one, where there is no room for two columns but the
 * structure still has to be reachable.
 */
export type WorkspaceLayout = "sidebar" | "sheet";

interface Props {
  structures: Panel[];
  active: StructurePanel | null;
  onSelect: (panel: StructurePanel) => void;
  style: ViewStyle;
  onStyleChange: (style: ViewStyle) => void;
  theme: Theme;
  onClose: () => void;
  layout: WorkspaceLayout;
}

const STYLES: { value: ViewStyle; label: string; hint: string }[] = [
  { value: "cartoon", label: "Cartoon", hint: "Protein fold with the ligand as sticks" },
  { value: "surface", label: "Surface", hint: "Binding pocket shape around the ligand" },
  { value: "ligand", label: "Ligand", hint: "The designed molecule on its own" },
];

/**
 * The workspace: the current protein-ligand complex, and every other complex
 * produced this session.
 *
 * The agent's job is designing molecules, so the structure stays reachable
 * while the conversation continues instead of scrolling away with the message
 * that produced it.
 */
export function WorkspacePanel({
  structures,
  active,
  onSelect,
  style,
  onStyleChange,
  theme,
  onClose,
  layout,
}: Props) {
  const list = structures.filter((panel): panel is StructurePanel => panel.kind === "structure");
  const isSheet = layout === "sheet";

  // The sheet covers the page, so Escape must dismiss it and the page behind
  // it must not scroll while it is open.
  useEffect(() => {
    if (!isSheet) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = previous;
    };
  }, [isSheet, onClose]);

  const shell = isSheet
    ? "fixed inset-0 z-50 flex flex-col bg-raised"
    : "flex w-[22rem] xl:w-[26rem] shrink-0 flex-col border-l border-subtle bg-raised";

  return (
    <aside
      className={shell}
      role={isSheet ? "dialog" : undefined}
      aria-modal={isSheet || undefined}
      aria-label="Structure viewer"
    >
      <div className="border-subtle flex shrink-0 items-center gap-2 border-b px-3 py-2.5">
        <h2 className="text-xs font-semibold">Structure viewer</h2>
        <span className="text-faint ml-auto text-[11px]">
          {list.length} {list.length === 1 ? "structure" : "structures"}
        </span>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close the structure viewer"
          title="Close the structure viewer"
          className="text-muted hover:text-body -mr-1 grid size-8 cursor-pointer place-items-center rounded-lg text-base touch:size-11 touch:text-lg"
        >
          ×
        </button>
      </div>

      {active ? (
        <>
          <div
            role="group"
            aria-label="View style"
            className="border-subtle flex shrink-0 gap-1 border-b px-3 py-2"
          >
            {STYLES.map((option) => (
              <button
                key={option.value}
                type="button"
                title={option.hint}
                aria-pressed={style === option.value}
                onClick={() => onStyleChange(option.value)}
                className="flex-1 cursor-pointer rounded-md px-2.5 py-1 text-[11px] font-medium transition-colors lg:flex-none touch:min-h-11"
                style={
                  style === option.value
                    ? { background: "var(--color-accent)", color: "white" }
                    : { background: "var(--surface-inset)", color: "var(--text-muted)" }
                }
              >
                {option.label}
              </button>
            ))}
          </div>

          {/* Keying on the path rebuilds the viewer per structure, which frees
              the previous WebGL context instead of leaking it. */}
          <StructureViewer
            key={active.path}
            panel={active}
            theme={theme}
            style={style}
            className="min-h-0 flex-1"
          />

          <div className="border-subtle shrink-0 border-t px-3 py-2">
            <p className="truncate text-[11px] font-medium" title={active.path}>
              {active.name}
            </p>
            <p className="text-faint truncate text-[11px]">from {active.tool}</p>
          </div>
        </>
      ) : (
        <p className="text-muted flex-1 px-6 py-10 text-center text-xs">
          No structure yet. When the agent folds a complex, it appears here.
        </p>
      )}

      {list.length > 1 && (
        <div className="border-subtle max-h-40 shrink-0 overflow-y-auto border-t lg:max-h-56">
          <ul className="p-2">
            {list.map((panel) => (
              <li key={panel.path}>
                <button
                  type="button"
                  onClick={() => onSelect(panel)}
                  aria-current={active?.path === panel.path}
                  className="hover:bg-sunken flex w-full cursor-pointer items-center rounded-md px-2 py-1.5 text-left text-[11px] touch:min-h-11"
                  style={
                    active?.path === panel.path
                      ? { background: "var(--surface-sunken)", fontWeight: 600 }
                      : undefined
                  }
                  title={panel.path}
                >
                  <span className="truncate">{panel.name}</span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </aside>
  );
}
