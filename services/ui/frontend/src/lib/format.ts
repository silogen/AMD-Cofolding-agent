// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/** "850 ms", "12.4 s", "3 min 05 s", "1 h 02 min". */
export function duration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  const seconds = ms / 1000;
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} min ${String(Math.floor(seconds % 60)).padStart(2, "0")} s`;
  return `${Math.floor(minutes / 60)} h ${String(minutes % 60).padStart(2, "0")} min`;
}

/**
 * One line describing tool call arguments. Arguments can hold a whole
 * protein sequence, so long values are cut: the point is to recognise the
 * call, not to read every character.
 */
export function summarise(args: Record<string, unknown>): string {
  const entries = Object.entries(args ?? {});
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

/** A clock time for a step: "14:02:31". */
export function clock(ms: number): string {
  return new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}
