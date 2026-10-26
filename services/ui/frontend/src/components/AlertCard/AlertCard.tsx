// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { openWorkbench } from "../../lib/workbenchBus";
import { ArtifactButton } from "../Transcript/ArtifactButton";

const SEVERITY: Record<string, string> = {
  critical: "var(--color-accent)",
  high: "var(--color-accent)",
  medium: "var(--color-warn)",
  low: "var(--color-structure)",
};

const text = (value: unknown): string | null => (typeof value === "string" || typeof value === "number" ? String(value) : null);

/** What the run monitor flagged about a job. */
export function AlertCard({ alert }: { alert: Record<string, unknown> }) {
  const severity = (text(alert.severity) ?? "medium").toLowerCase();
  const color = SEVERITY[severity] ?? SEVERITY.medium;
  const type = text(alert.type);
  const job = text(alert.job_id);
  const message = text(alert.message) ?? "The monitor flagged a job.";

  return (
    <div role="alert" className="border-l-2 py-0.5 pl-3 text-xs" style={{ borderLeftColor: color }}>
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
        <span className="font-semibold">Monitor alert{type ? `: ${type.replace(/_/g, " ")}` : ""}</span>
        <span className="text-[11px] capitalize" style={{ color }}>
          {severity}
        </span>
        {job && <span className="text-faint font-mono text-[11px]">job {job}</span>}
      </div>
      <p className="text-body mt-1.5 text-[0.8125rem] leading-relaxed">{message}</p>
      <div className="mt-1.5">
        <ArtifactButton label="Open jobs" onClick={() => openWorkbench("jobs")} />
      </div>
    </div>
  );
}
