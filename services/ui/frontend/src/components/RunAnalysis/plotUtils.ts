// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useState } from "react";
import type { Endpoint } from "../../lib/runAnalysis";

export const TOTAL_COLOUR = "var(--color-structure)";
const PALETTE = [
  "var(--color-molecule)",
  "var(--color-ok)",
  "var(--color-table)",
  "oklch(0.68 0.16 340)",
  "oklch(0.6 0.12 280)",
  "oklch(0.7 0.12 200)",
  "oklch(0.62 0.1 30)",
  "oklch(0.7 0.1 150)",
];

/** One colour per endpoint, the same in every plot; a raw value takes its component's. */
export function endpointColours(endpoints: readonly Endpoint[]): Map<string, string> {
  const map = new Map<string, string>();
  let next = 0;
  for (const e of endpoints) {
    if (e.kind === "total") map.set(e.key, TOTAL_COLOUR);
    else if (e.kind === "component") map.set(e.key, PALETTE[next++ % PALETTE.length]);
  }
  for (const e of endpoints) if (e.kind === "raw") map.set(e.key, map.get(e.key.replace(/\s*\(raw\)$/i, "")) ?? "var(--text-muted)");
  return map;
}

/** The element's content width, followed as it resizes. */
export function useWidth<T extends HTMLElement>(): [(node: T | null) => void, number] {
  const [node, setNode] = useState<T | null>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    if (!node) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.round(entry.contentRect.width)));
    observer.observe(node);
    return () => observer.disconnect();
  }, [node]);
  return [setNode, width];
}

/** "0.72", "412", "1.2e-4": short numbers for axes and tooltips. */
export function short(v: number): string {
  if (!Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a < 0.001 || a >= 100_000)) return v.toExponential(1);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(2);
}
