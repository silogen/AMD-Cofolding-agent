// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useState } from "react";
import { groupColour, objectiveKey, type CatalogueEntry, type Objective, type ObjectiveGroup } from "../../lib/ttp";

interface Props {
  groups: ObjectiveGroup[];
  objectives: Objective[];
  onToggle: (entry: CatalogueEntry, on: boolean) => void;
}

/** One checkbox per metric, a row per group, tinted by group. */
export function ObjectivePicker({ groups, objectives, onToggle }: Props) {
  const [expanded, setExpanded] = useState<string[]>([]);
  const keys = new Set(objectives.map(objectiveKey));
  return (
    <div className="space-y-1.5">
      {groups.map((group) => {
        const open = expanded.includes(group.name);
        const extras = group.entries.filter((e) => e.extra && !keys.has(e.key)).length;
        const shown = group.entries.filter((e) => open || !e.extra || keys.has(e.key));
        const colour = groupColour(group.name);
        return (
          <div key={group.name} role="group" aria-label={group.name} className="flex flex-wrap gap-1">
              {shown.map((e) => {
                const on = keys.has(e.key);
                const blocked = !on && e.after !== undefined && !keys.has(e.after);
                const needs = blocked ? group.entries.find((x) => x.key === e.after)?.label : undefined;
                return (
                  <Chip key={e.key} on={on} colour={colour} disabled={blocked} title={needs ? `Needs ${needs}` : group.name} onClick={() => onToggle(e, !on)}>
                    {e.label}
                  </Chip>
                );
              })}
              {extras > 0 && (
                <button
                  type="button"
                  onClick={() => setExpanded((list) => (open ? list.filter((g) => g !== group.name) : [...list, group.name]))}
                  className="text-faint hover:text-body cursor-pointer px-1 text-[11px]"
                >
                  {open ? "Fewer" : `${extras} more`}
                </button>
              )}
          </div>
        );
      })}
    </div>
  );
}

function Chip({ on, colour, disabled, title, onClick, children }: { on: boolean; colour: string; disabled?: boolean; title?: string; onClick: () => void; children: string }) {
  return (
    <button
      type="button"
      role="checkbox"
      aria-checked={on}
      disabled={disabled}
      title={title}
      onClick={onClick}
      className={`flex cursor-pointer items-center gap-1 rounded-full border py-px pr-2 pl-1.5 text-[11px] disabled:cursor-default disabled:opacity-40 ${on ? "text-body" : "border-subtle text-muted hover:text-body"}`}
      style={on ? { borderColor: `color-mix(in oklab, ${colour} 45%, var(--border-subtle))`, background: `color-mix(in oklab, ${colour} 10%, transparent)` } : undefined}
    >
      <span
        aria-hidden="true"
        className="flex size-2.5 items-center justify-center rounded-sm border text-[8px] leading-none"
        style={on ? { background: colour, borderColor: colour, color: "var(--surface-raised)" } : { borderColor: `color-mix(in oklab, ${colour} 50%, var(--border-strong))` }}
      >
        {on ? "✓" : ""}
      </span>
      {children}
    </button>
  );
}
