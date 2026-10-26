// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { scoreColor } from "../../lib/candidates";

interface Props {
  value: number | undefined;
  /** Width of the bar track; the number follows it. */
  className?: string;
}

/**
 * A 0–1 value as a coloured bar and its number. Values outside 0–1 are not
 * scores, so they get the number alone rather than a bar that means nothing.
 */
export function ScoreBar({ value, className = "w-16" }: Props) {
  if (value === undefined) return <span className="text-faint">—</span>;
  const unit = value >= 0 && value <= 1;
  return (
    <span className="inline-flex items-center gap-1.5">
      {unit && (
        <span aria-hidden="true" className={`bg-inset h-1.5 shrink-0 overflow-hidden rounded-full ${className}`}>
          <span
            className="block h-full rounded-full"
            style={{ width: `${value * 100}%`, background: scoreColor(value) }}
          />
        </span>
      )}
      <span className="font-mono tabular-nums">{value.toFixed(2)}</span>
    </span>
  );
}
