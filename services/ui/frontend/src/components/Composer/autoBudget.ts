// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/** The budgets offered before "Custom", in minutes. */
export const BUDGETS = [30, 60, 120, 240];

/** *ms* left, short enough to sit in a chip: "2h 05m", "9m", "30s". */
export function countdown(ms: number): string {
  const seconds = Math.max(0, Math.ceil(ms / 1000));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (hours > 0) return `${hours}h ${String(minutes).padStart(2, "0")}m`;
  if (minutes > 0) return `${minutes}m`;
  return `${seconds}s`;
}

export function budgetLabel(minutes: number): string {
  return minutes < 60 ? `${minutes} min` : `${minutes / 60} h`;
}
