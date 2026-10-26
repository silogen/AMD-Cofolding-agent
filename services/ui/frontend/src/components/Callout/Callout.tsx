// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { ReactNode } from "react";
import type { CalloutTone } from "../../lib/answer";

const COLOR: Record<CalloutTone, string> = {
  info: "var(--color-structure)",
  warn: "var(--color-warn)",
  bad: "var(--color-accent)",
  ok: "var(--color-ok)",
};

const ICON: Record<CalloutTone, string> = { info: "ⓘ", warn: "⚠", bad: "⚠", ok: "✓" };

/** A note set apart from the answer, coloured by what kind of note it is. */
export function Callout({ tone, title, children }: { tone: CalloutTone; title?: string | null; children: ReactNode }) {
  return (
    <div
      className="my-3 flex items-start gap-2 border-l-2 py-0.5 pl-3 text-[0.875rem]"
      style={{ borderLeftColor: COLOR[tone] }}
      role={tone === "warn" || tone === "bad" ? "note" : undefined}
    >
      <span aria-hidden="true" className="text-xs leading-snug" style={{ color: COLOR[tone] }}>
        {ICON[tone]}
      </span>
      <div className="callout-body min-w-0 flex-1 [&>*:first-child]:mt-0 [&>*:last-child]:mb-0">
        {title && <b className="mb-0.5 block">{title}</b>}
        {children}
      </div>
    </div>
  );
}
