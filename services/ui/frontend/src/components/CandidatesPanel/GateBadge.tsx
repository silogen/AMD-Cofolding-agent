// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { Gate } from "../../lib/gate";

const STYLE: Record<Gate["status"], { label: string; color: string }> = {
  pass: { label: "PASS", color: "#3ccf91" },
  fail: { label: "FAIL", color: "#ff4d4f" },
  unknown: { label: "n/a", color: "var(--text-faint)" },
};

/** The confidence-gate verdict of a candidate, with its reasons on hover. */
export function GateBadge({ gate }: { gate: Gate }) {
  const { label, color } = STYLE[gate.status];
  const title =
    gate.status === "unknown"
      ? "No confident complex yet: cofold this molecule to gate it"
      : gate.reasons.join("; ") || "Passes confidence, ligand ipTM and PoseBusters";
  return (
    <span
      title={title}
      className="inline-flex rounded-full border px-1.5 py-px text-[10px] font-semibold tracking-wide"
      style={{ color, borderColor: color }}
    >
      {label}
    </span>
  );
}
