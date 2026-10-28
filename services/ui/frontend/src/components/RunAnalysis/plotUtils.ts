// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useState } from "react";
import { formatValue, type Endpoint } from "../../lib/runAnalysis";

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

/** One colour per endpoint, the same in every plot. */
export function endpointColours(endpoints: readonly Endpoint[]): Map<string, string> {
  const map = new Map<string, string>();
  let next = 0;
  for (const e of endpoints) map.set(e.key, e.kind === "total" ? TOTAL_COLOUR : PALETTE[next++ % PALETTE.length]);
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

/** "0.72", "41.2", "412": short numbers for score axes and tooltips. */
export function short(v: number): string {
  const a = Math.abs(v);
  return formatValue(v, a >= 100 ? 0 : a >= 10 ? 1 : 2);
}
