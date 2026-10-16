// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useState } from "react";
import type { Panel, StructurePanel, ToolResultEvent } from "../../types/api";
import type { Theme } from "../../hooks/useTheme";
import { StructureCard } from "../StructureCard/StructureCard";
import { MoleculeCard } from "../MoleculeCard/MoleculeCard";
import { DataTable } from "../DataTable/DataTable";

interface Props {
  name: string;
  args: Record<string, unknown>;
  result?: ToolResultEvent;
  theme: Theme;
  onOpenStructure: (panel: StructurePanel) => void;
  openStructurePath: string | null;
}

/**
 * One tool call and what it returned.
 *
 * The call is shown as a compact line so a long chain stays readable, and the
 * data it produced is shown in full underneath. The raw JSON stays one click
 * away, because "what exactly did that tool say" is the question a scientist
 * asks when a number looks wrong.
 */
export function ToolBlock({
  name,
  args,
  result,
  theme,
  onOpenStructure,
  openStructurePath,
}: Props) {
  const [showRaw, setShowRaw] = useState(false);
  const running = !result;
  const failed = result?.isError ?? false;
  const panels = result?.panels ?? [];

  return (
    <div className="space-y-2">
      <div
        className={`bg-sunken border-subtle flex flex-wrap items-center gap-x-2 gap-y-1 rounded-lg border px-3 py-1.5 text-xs ${
          failed ? "border-strong" : ""
        }`}
      >
        <span aria-hidden="true" className="text-faint font-mono text-[10px]">
          {running ? "◌" : failed ? "✕" : "✓"}
        </span>
        <span className="font-mono font-semibold" style={{ color: failed ? "var(--color-accent)" : undefined }}>
          {name}
        </span>
        <span className="text-faint min-w-0 flex-1 truncate font-mono text-[11px]" title={summarise(args)}>
          {summarise(args)}
        </span>

        {running && <span className="text-muted text-[11px]">running…</span>}

        {result && (
          <button
            type="button"
            onClick={() => setShowRaw((value) => !value)}
            className="text-muted hover:text-body cursor-pointer text-[11px] font-medium touch:-my-1 touch:min-h-11 touch:px-1"
            aria-expanded={showRaw}
          >
            {showRaw ? "Hide output" : "Show output"}
          </button>
        )}
      </div>

      {showRaw && result && (
        <pre className="bg-sunken border-subtle max-h-72 overflow-auto rounded-lg border p-3 font-mono text-[11px] leading-relaxed">
          {result.result}
          {result.truncated && (
            <span className="text-faint">{"\n\n… output truncated for display."}</span>
          )}
        </pre>
      )}

      {failed && result && (
        <p className="text-muted text-xs">The tool reported an error. Open the output to see it.</p>
      )}

      {panels.length > 0 && <PanelGrid panels={panels} theme={theme} onOpenStructure={onOpenStructure} openStructurePath={openStructurePath} />}
    </div>
  );
}

function PanelGrid({
  panels,
  theme,
  onOpenStructure,
  openStructurePath,
}: {
  panels: Panel[];
  theme: Theme;
  onOpenStructure: (panel: StructurePanel) => void;
  openStructurePath: string | null;
}) {
  const structures = panels.filter((panel): panel is StructurePanel => panel.kind === "structure");
  const molecules = panels.filter((panel) => panel.kind === "molecule");
  const tables = panels.filter((panel) => panel.kind === "table");

  return (
    <div className="space-y-3">
      {structures.length > 0 && (
        <div className="grid gap-3 sm:grid-cols-2">
          {structures.map((panel) => (
            <StructureCard
              key={panel.path}
              panel={panel}
              theme={theme}
              onOpen={onOpenStructure}
              isOpen={openStructurePath === panel.path}
            />
          ))}
        </div>
      )}

      {molecules.length > 0 && (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {molecules.map((panel) => (
            <MoleculeCard key={panel.smiles} panel={panel} theme={theme} />
          ))}
        </div>
      )}

      {tables.map((panel, index) => (
        <DataTable key={index} panel={panel} />
      ))}
    </div>
  );
}

/**
 * One line describing the call arguments.
 *
 * Tool arguments can hold a whole protein sequence, so long values are cut:
 * the point is to recognise the call, not to read every character.
 */
function summarise(args: Record<string, unknown>): string {
  const entries = Object.entries(args ?? {});
  if (entries.length === 0) return "";
  return entries
    .map(([key, value]) => {
      const text =
        typeof value === "string"
          ? value
          : Array.isArray(value)
            ? `[${value.length} item${value.length === 1 ? "" : "s"}]`
            : JSON.stringify(value);
      const short = (text ?? "").length > 40 ? `${text.slice(0, 40)}…` : text;
      return `${key}=${short}`;
    })
    .join("  ");
}
