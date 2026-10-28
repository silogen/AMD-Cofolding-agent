// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useRef, useState } from "react";
import { groupState, objectiveKey, type CatalogueEntry, type Objective, type ObjectiveGroup } from "../../lib/ttp";

interface Props {
  groups: ObjectiveGroup[];
  objectives: Objective[];
  onToggle: (entry: CatalogueEntry, on: boolean) => void;
  onToggleGroup: (group: ObjectiveGroup) => void;
}

/** One checkbox per metric, grouped; a group's own checkbox selects or clears its core metrics. */
export function ObjectivePicker({ groups, objectives, onToggle, onToggleGroup }: Props) {
  const [expanded, setExpanded] = useState<string[]>([]);
  const keys = new Set(objectives.map(objectiveKey));
  return (
    <div className="space-y-1.5">
      {groups.map((group) => {
        const open = expanded.includes(group.name);
        const extras = group.entries.filter((e) => e.extra && !keys.has(e.key)).length;
        const shown = group.entries.filter((e) => open || !e.extra || keys.has(e.key));
        const core = group.entries.filter((e) => e.core);
        return (
          <div key={group.name} role="group" aria-label={group.name} className="flex items-start gap-2">
            <span className="w-28 shrink-0 pt-0.5">
              {core.length > 0 ? (
                <GroupBox
                  label={group.name}
                  state={groupState(objectives, group)}
                  title={`All of: ${core.map((e) => e.label).join(", ")}`}
                  onChange={() => onToggleGroup(group)}
                />
              ) : (
                <span className="text-muted text-[11px] font-medium">{group.name}</span>
              )}
            </span>
            <div className="flex min-w-0 flex-1 flex-wrap gap-1">
              {shown.map((e) => {
                const on = keys.has(e.key);
                const blocked = !on && e.after !== undefined && !keys.has(e.after);
                const needs = blocked ? group.entries.find((x) => x.key === e.after)?.label : undefined;
                return (
                  <Chip key={e.key} on={on} disabled={blocked} title={needs ? `Needs ${needs}` : undefined} onClick={() => onToggle(e, !on)}>
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
          </div>
        );
      })}
    </div>
  );
}

function GroupBox({ label, state, title, onChange }: { label: string; state: "all" | "some" | "none"; title: string; onChange: () => void }) {
  const ref = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.indeterminate = state === "some";
  }, [state]);
  return (
    <label title={title} className="text-muted hover:text-body flex cursor-pointer items-center gap-1.5 text-[11px] font-medium">
      <input ref={ref} type="checkbox" checked={state === "all"} onChange={onChange} className="size-3 cursor-pointer accent-[var(--color-structure)]" />
      {label}
    </label>
  );
}

function Chip({ on, disabled, title, onClick, children }: { on: boolean; disabled?: boolean; title?: string; onClick: () => void; children: string }) {
  return (
    <button
      type="button"
      role="checkbox"
      aria-checked={on}
      disabled={disabled}
      title={title}
      onClick={onClick}
      className={`flex cursor-pointer items-center gap-1 rounded-full border py-px pr-2 pl-1.5 text-[11px] disabled:cursor-default disabled:opacity-40 ${on ? "border-strong bg-sunken text-body" : "border-subtle text-muted hover:text-body"}`}
    >
      <span
        aria-hidden="true"
        className="flex size-2.5 items-center justify-center rounded-sm border text-[8px] leading-none text-white"
        style={on ? { background: "var(--color-structure)", borderColor: "var(--color-structure)" } : { borderColor: "var(--border-strong)" }}
      >
        {on ? "✓" : ""}
      </span>
      {children}
    </button>
  );
}
