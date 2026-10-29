// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useState } from "react";
import { budgetLabel, BUDGETS, countdown } from "./autoBudget";

interface Props {
  /** End of the budget, ISO 8601 UTC; null when the agent is not on its own. */
  until: string | null;
  /** Minutes to work alone, or 0 to stop. */
  onSet: (minutes: number) => void;
}

/** What is left of the budget, counting down; 0 when there is none. */
function useRemaining(until: string | null): number {
  const target = until ? Date.parse(until) : 0;
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!target || target <= Date.now()) return;
    const timer = window.setInterval(() => {
      setNow(Date.now());
      // Nothing left to count: the chip is gone and the tick with it.
      if (Date.now() >= target) window.clearInterval(timer);
    }, 1000);
    return () => window.clearInterval(timer);
  }, [target]);
  return target ? Math.max(0, target - now) : 0;
}

/**
 * The autonomy budget, next to the TTP button.
 *
 * Off, it is one more composer icon that offers a few durations. On, it is a
 * chip counting down, with a stop next to it; when the time runs out the chip
 * goes away by itself and the button comes back.
 *
 * Mount it keyed on *until*: the clock it counts from is read once, so a new
 * deadline needs a fresh one.
 */
export function AutoButton({ until, onSet }: Props) {
  const [menu, setMenu] = useState(false);
  const left = useRemaining(until);

  if (left > 0) {
    return (
      <span
        className="flex h-7 shrink-0 items-center gap-0.5 rounded-full border pr-0.5 pl-2.5 text-[11px] touch:h-9"
        style={{ borderColor: "var(--color-accent)" }}
        title="The agent keeps working on its own until this runs out"
      >
        <span className="font-medium">Auto {countdown(left)}</span>
        <button
          type="button"
          onClick={() => onSet(0)}
          aria-label="Stop working on its own"
          title="Stop working on its own"
          className="text-faint hover:text-body grid size-5 cursor-pointer place-items-center rounded-full"
        >
          ×
        </button>
      </span>
    );
  }

  const choose = (minutes: number) => {
    setMenu(false);
    if (minutes > 0) onSet(minutes);
  };

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setMenu((open) => !open)}
        aria-haspopup="menu"
        aria-expanded={menu}
        aria-label="Work on its own"
        title="Let the agent keep working on its own"
        className="text-muted hover:text-body hover:bg-sunken grid size-7 shrink-0 cursor-pointer place-items-center rounded-lg touch:size-9"
      >
        <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M20 12a8 8 0 1 1-2.4-5.7" />
          <path d="M20 4.5V9h-4.5" />
        </svg>
      </button>
      {menu && (
        <div
          role="menu"
          aria-label="Autonomy budget"
          className="bg-raised border-subtle shadow-card absolute bottom-full left-0 z-20 mb-1 w-44 overflow-hidden rounded-lg border py-1 text-xs"
          onMouseLeave={() => setMenu(false)}
        >
          <p className="text-faint px-3 pt-0.5 pb-1 leading-snug">Keep working alone for</p>
          {BUDGETS.map((minutes) => (
            <button
              key={minutes}
              type="button"
              role="menuitem"
              onClick={() => choose(minutes)}
              className="hover:bg-sunken w-full cursor-pointer px-3 py-1.5 text-left"
            >
              {budgetLabel(minutes)}
            </button>
          ))}
          <button
            type="button"
            role="menuitem"
            onClick={() => choose(Number(window.prompt("Minutes to work alone?", "90")))}
            className="hover:bg-sunken text-muted w-full cursor-pointer px-3 py-1.5 text-left"
          >
            Custom…
          </button>
        </div>
      )}
    </div>
  );
}
