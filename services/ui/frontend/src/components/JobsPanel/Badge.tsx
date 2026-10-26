// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { ReactNode } from "react";
import { toneColor, type Tone } from "../../lib/jobs";

interface Props {
  tone: Tone;
  children: ReactNode;
  title?: string;
}

/** A small tinted label: a job state, a severity, a component's health. */
export function Badge({ tone, children, title }: Props) {
  const color = toneColor(tone);
  return (
    <span
      title={title}
      className="inline-flex shrink-0 items-center rounded-full px-1.5 py-px text-[10px] font-semibold tracking-wide whitespace-nowrap uppercase"
      style={{ color, background: `color-mix(in oklab, ${color} 16%, transparent)` }}
    >
      {children}
    </span>
  );
}
