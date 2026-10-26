// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useRef, useState } from "react";

export interface PaletteItem {
  id: string;
  label: string;
  group: string;
  hint?: string;
  run: () => void;
}

interface Props {
  items: PaletteItem[];
  onClose: () => void;
}

/** Every word of the query appears somewhere in the label, group or hint. */
function matches(item: PaletteItem, query: string): boolean {
  const haystack = `${item.label} ${item.group} ${item.hint ?? ""}`.toLowerCase();
  return query
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .every((word) => haystack.includes(word));
}

/**
 * Keyboard access to every action: Ctrl+K, type, Enter. Mounted only while
 * open, so each opening starts with an empty query.
 */
export function CommandPalette({ items, onClose }: Props) {
  const [query, setQuery] = useState("");
  const [highlight, setHighlight] = useState(0);
  const listRef = useRef<HTMLUListElement>(null);

  const shown = useMemo(() => items.filter((item) => matches(item, query)).slice(0, 60), [items, query]);

  useEffect(() => {
    listRef.current?.querySelector('[aria-selected="true"]')?.scrollIntoView({ block: "nearest" });
  }, [highlight]);

  const run = (item: PaletteItem | undefined) => {
    if (!item) return;
    onClose();
    item.run();
  };

  return (
    <div className="fixed inset-0 z-[60] flex items-start justify-center bg-black/40 p-4 pt-[12vh]" onMouseDown={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Commands"
        onMouseDown={(event) => event.stopPropagation()}
        className="bg-raised border-subtle w-full max-w-lg overflow-hidden rounded-xl border shadow-2xl"
      >
        <input
          autoFocus
          value={query}
          onChange={(event) => {
            setQuery(event.target.value);
            setHighlight(0);
          }}
          onKeyDown={(event) => {
            if (event.key === "Escape") onClose();
            if (event.key === "ArrowDown") {
              event.preventDefault();
              setHighlight((current) => Math.min(current + 1, shown.length - 1));
            }
            if (event.key === "ArrowUp") {
              event.preventDefault();
              setHighlight((current) => Math.max(current - 1, 0));
            }
            if (event.key === "Enter") {
              event.preventDefault();
              run(shown[highlight]);
            }
          }}
          placeholder="Search actions, conversations and prompts…"
          aria-label="Search commands"
          className="border-subtle w-full border-b bg-transparent px-4 py-3 text-sm outline-none"
        />
        <ul ref={listRef} role="listbox" className="max-h-[50vh] overflow-y-auto py-1">
          {shown.length === 0 && <li className="text-faint px-4 py-6 text-center text-xs">Nothing matches.</li>}
          {shown.map((item, index) => {
            const header = index === 0 || shown[index - 1].group !== item.group ? item.group : null;
            return (
              <li key={item.id} role="option" aria-selected={index === highlight}>
                {header && (
                  <p className="text-faint px-4 pt-2 pb-1 text-[10px] font-semibold tracking-wide uppercase">{header}</p>
                )}
                <button
                  type="button"
                  onClick={() => run(item)}
                  onMouseMove={() => setHighlight(index)}
                  className={`flex w-full cursor-pointer items-center gap-3 px-4 py-2 text-left text-xs ${
                    index === highlight ? "bg-sunken" : ""
                  }`}
                >
                  {index === highlight && (
                    <span aria-hidden="true" className="h-3.5 w-0.5 shrink-0 rounded-full" style={{ background: "var(--color-accent)" }} />
                  )}
                  <span className="min-w-0 flex-1 truncate">{item.label}</span>
                  {item.hint && <span className="text-faint shrink-0 truncate text-[11px]">{item.hint}</span>}
                </button>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}
