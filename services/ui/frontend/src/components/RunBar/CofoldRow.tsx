// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useState } from "react";
import { clock, cofoldView, type CofoldCall, type CofoldStatus } from "../../lib/cofold";

interface Props {
  status: CofoldStatus | null;
  /** When *status* was read. */
  at: number;
  calls: readonly CofoldCall[];
}

/** "Cofolding · 2 molecules ▬▬ 1/2 · 0:41": one line of the run bar while Boltz works. */
export function CofoldRow({ status, at, calls }: Props) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  const view = cofoldView(status, at, calls, now);
  if (!view) return null;
  const fraction = view.total > 0 ? view.done / view.total : 0;

  return (
    <span className="flex min-w-0 flex-1 items-center gap-2 overflow-hidden" title={view.title}>
      <span aria-hidden="true" className="size-1.5 shrink-0 animate-pulse rounded-full" style={{ background: "var(--color-structure)" }} />
      <span className="shrink-0 font-medium">{view.label}</span>
      <span className="text-faint min-w-0 truncate">{view.detail}</span>
      {view.total > 0 && (
        <>
          <span className="bg-inset h-1 w-10 shrink-0 overflow-hidden rounded-full @md:w-20" aria-hidden="true">
            <span className="block h-full rounded-full transition-[width] duration-700" style={{ width: `${Math.round(fraction * 100)}%`, background: "var(--color-structure)" }} />
          </span>
          <span className="text-muted shrink-0 font-mono tabular-nums" title="Molecules cofolded / asked for">
            {view.done}/{view.total}
          </span>
        </>
      )}
      {view.note && <span className="text-faint shrink-0">{view.note}</span>}
      {view.elapsedMs !== null && (
        <span className="text-faint shrink-0 tabular-nums" title="Elapsed">
          {clock(view.elapsedMs)}
        </span>
      )}
    </span>
  );
}
