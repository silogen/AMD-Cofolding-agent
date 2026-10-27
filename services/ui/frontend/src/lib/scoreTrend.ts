// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { JobStatus } from "./jobs";

/**
 * A run's mean total score per step, drawn as a tiny line next to its progress.
 *
 * get_optimization_status reports it as score_trajectory (mean and max per
 * step) or, on the current MCP server, only as the "Score" entry of
 * per_component_trajectory (mean only, about 30 points per run).
 */

export interface ScorePoint {
  step: number;
  mean: number;
  max: number | null;
}

/** Sparklines never need more points than this. */
export const MAX_POINTS = 60;

export function scoreSeries(status: Pick<JobStatus, "trajectory" | "componentTrajectories"> | null | undefined): ScorePoint[] {
  if (!status) return [];
  if (status.trajectory.length) return status.trajectory.map((p) => ({ step: p.step, mean: p.mean, max: p.max }));
  const name = Object.keys(status.componentTrajectories).find((key) => /^(score|total_score)$/i.test(key));
  return (name ? status.componentTrajectories[name] : [])
    .map((p) => ({ step: p.step, mean: p.mean, max: null }))
    .sort((a, b) => a.step - b.step);
}

/** At most `limit` points, evenly spaced, always keeping the first and the last. */
export function downsample<T>(points: readonly T[], limit = MAX_POINTS): T[] {
  if (points.length <= limit) return [...points];
  if (limit < 2) return points.slice(-1);
  const stride = (points.length - 1) / (limit - 1);
  return Array.from({ length: limit }, (_, i) => points[Math.round(i * stride)]);
}

/** An SVG path through the values, spread over the width and scaled to their own range; flat values sit in the middle. */
export function sparkPath(values: readonly number[], width: number, height: number, pad = 1): string {
  const finite = values.filter(Number.isFinite);
  if (finite.length < 2) return "";
  const lo = Math.min(...finite);
  const hi = Math.max(...finite);
  const dx = (width - 2 * pad) / (finite.length - 1);
  const y = (v: number) => (hi === lo ? height / 2 : height - pad - ((v - lo) / (hi - lo)) * (height - 2 * pad));
  return finite.map((v, i) => `${i ? "L" : "M"}${(pad + i * dx).toFixed(1)},${y(v).toFixed(1)}`).join("");
}

/** "Mean score 0.71 → 0.93 over 300 steps". */
export function trendTitle(points: readonly ScorePoint[]): string {
  if (points.length < 2) return "";
  const first = points[0];
  const last = points[points.length - 1];
  return `Mean score ${first.mean.toFixed(2)} → ${last.mean.toFixed(2)} over ${last.step} step${last.step === 1 ? "" : "s"}`;
}
