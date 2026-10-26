// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { Gate } from "../../lib/gate";

const STYLE: Record<Gate["status"], { label: string; color: string }> = {
  pass: { label: "pass", color: "var(--color-ok)" },
  fail: { label: "below", color: "var(--color-warn)" },
  unknown: { label: "n/a", color: "var(--text-faint)" },
};

/** The quality-gate verdict of a candidate, with its reasons on hover. */
export function GateBadge({ gate }: { gate: Gate }) {
  const { label, color } = STYLE[gate.status];
  const title =
    gate.status === "unknown"
      ? "No confident complex yet: cofold this molecule to gate it"
      : gate.status === "fail"
        ? `Below the quality gate (${gate.reasons.join("; ")}); read its affinity with caution`
        : "Passes confidence, ligand ipTM and PoseBusters";
  return (
    <span title={title} className="inline-flex items-center gap-1 text-[10px] font-medium whitespace-nowrap" style={{ color }}>
      <span aria-hidden="true" className="size-1.5 rounded-full" style={{ background: color }} />
      {label}
    </span>
  );
}
