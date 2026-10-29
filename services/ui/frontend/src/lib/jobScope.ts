// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * The one job the Jobs tab is about, shared by Summary, Results and Analysis:
 * the job picked with "Other jobs" when there is one, else the conversation's
 * latest job that started, else its latest at all. A conversation that
 * launched nothing shows no job until one is picked.
 */
export function scopedJob<T extends { id: string; state: string }>(jobs: readonly T[], own: readonly T[], focused: string | null): T | null {
  const picked = focused ? jobs.find((j) => j.id === focused) : undefined;
  if (picked) return picked;
  return own.find((j) => j.state !== "failed" && j.state !== "queued") ?? own[0] ?? null;
}
