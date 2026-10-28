// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useState } from "react";
import { clock, cofoldView, type CofoldCall, type CofoldMemory, type CofoldStatus } from "../../lib/cofold";
import { etaLabel } from "../../lib/runProgress";

interface Props {
  status: CofoldStatus | null;
  /** When *status* was read. */
  at: number;
  calls: readonly CofoldCall[];
  memory?: CofoldMemory;
  /** The batch is done and the run prepares the next one. */
  finished?: boolean;
}

/**
 * "Cofolding · 10 molecules ▬▬ 6/10 · 4 queued · 1:23 · ~1 min left": the
 * cells of one run bar line while Boltz works, in the bar's columns.
 */
export function CofoldRow({ status, at, calls, memory, finished = false }: Props) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (finished) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [finished]);
  const view = cofoldView(status, at, calls, now, { memory, finished });
  if (!view) return null;

  return (
    <>
      <span
        aria-hidden="true"
        className={`size-1.5 rounded-full ${finished ? "opacity-50" : "animate-pulse"}`}
        style={{ background: "var(--color-structure)" }}
      />
      <span className="font-medium" title={view.title}>
        {view.label}
      </span>
      <span className="text-faint min-w-0 truncate" title={view.title}>
        {view.detail}
      </span>
      <Bar fraction={view.total > 0 ? view.fraction : null} />
      <span className="text-muted text-right font-mono tabular-nums" title="Molecules cofolded / asked for">
        {view.total > 0 ? `${view.done}/${view.total}` : ""}
      </span>
      <span className="text-faint">{view.note}</span>
      <span className="text-faint text-right tabular-nums" title="Elapsed">
        {view.elapsedMs !== null ? clock(view.elapsedMs) : ""}
      </span>
      <span className="text-faint tabular-nums" title={view.remainingMs !== null ? "Estimated from recent molecule times" : undefined}>
        {view.remainingMs !== null && <Eta remainingMs={view.remainingMs} />}
      </span>
    </>
  );
}

/** "~7 min left", without "left" when the bar is narrow. */
export function Eta({ remainingMs }: { remainingMs: number | null }) {
  const label = etaLabel(remainingMs);
  if (!label.endsWith(" left")) return label;
  return (
    <>
      {label.slice(0, -" left".length)}
      <span className="hidden @lg:inline"> left</span>
    </>
  );
}

/** The bar's progress track; empty without a fraction. */
export function Bar({ fraction, colour = "var(--color-structure)" }: { fraction: number | null; colour?: string }) {
  return (
    <span className="bg-inset block h-1 w-10 overflow-hidden rounded-full @md:w-20" aria-hidden="true">
      {fraction !== null && (
        <span className="block h-full rounded-full transition-[width] duration-1000 ease-linear" style={{ width: `${Math.round(fraction * 1000) / 10}%`, background: colour }} />
      )}
    </span>
  );
}
