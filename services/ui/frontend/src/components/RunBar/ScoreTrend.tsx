// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { JobStatus } from "../../lib/jobs";
import { downsample, scoreSeries, sparkPath, trendTitle } from "../../lib/scoreTrend";

interface Props {
  status: Pick<JobStatus, "trajectory" | "componentTrajectories"> | null | undefined;
  width?: number;
  height?: number;
  className?: string;
}

/** The run's mean score per step as a tiny muted line; nothing until there are two points. */
export function ScoreTrend({ status, width = 64, height = 14, className = "" }: Props) {
  const points = downsample(scoreSeries(status));
  const path = sparkPath(points.map((p) => p.mean), width, height);
  if (!path) return null;
  const title = trendTitle(points);
  return (
    <span className={`shrink-0 leading-none ${className}`} title={title}>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title} className="block">
        <path d={path} fill="none" stroke="var(--color-structure)" strokeWidth={1} strokeLinejoin="round" strokeLinecap="round" />
      </svg>
    </span>
  );
}
