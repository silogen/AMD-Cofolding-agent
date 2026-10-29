// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { memo, useEffect, useMemo, useState } from "react";
import type { Campaign } from "../../types/api";
import { campaignText, campaignView, relativeTime } from "../../lib/campaign";
import { CopyButton } from "../CopyButton/CopyButton";

/** Findings shown while the card is folded; the rest are one click away. */
const FOLDED_FINDINGS = 2;

interface Props {
  campaign: Campaign | null;
  /** Start unfolded; for tests. */
  defaultOpen?: boolean;
}

/**
 * The agent's campaign log, a slim card above the composer: the goal, the
 * next step and the latest findings. Unfolded it lists every finding with
 * when it was written. Nothing at all while the log is empty.
 */
export const CampaignCard = memo(function CampaignCard({ campaign, defaultOpen = false }: Props) {
  const view = useMemo(() => campaignView(campaign), [campaign]);
  const [open, setOpen] = useState(defaultOpen);
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!open) return;
    const timer = window.setInterval(() => setNow(Date.now()), 60_000);
    return () => window.clearInterval(timer);
  }, [open]);

  if (!view) return null;
  const findings = open ? view.findings : view.findings.slice(0, FOLDED_FINDINGS);
  const hidden = view.findings.length - findings.length;
  const wrap = open ? "whitespace-pre-wrap break-words" : "truncate";

  return (
    <div className="mx-auto w-full max-w-3xl px-3 pb-1.5 sm:px-4">
      <section
        aria-label="Campaign log"
        className="bg-sunken border-subtle grid grid-cols-[auto_minmax(0,1fr)_auto] items-baseline gap-x-2.5 gap-y-0.5 rounded-lg border py-1.5 pr-1 pl-2.5 text-[11px]"
      >
        {view.goal && (
          <>
            <span className="text-faint">Goal</span>
            <span className={`text-muted ${wrap}`} title={open ? undefined : view.goal}>
              {view.goal}
            </span>
          </>
        )}

        {view.next && (
          <>
            <span className="text-faint">Next</span>
            <span className={`text-body text-xs font-medium ${wrap}`} title={open ? undefined : view.next}>
              {view.next}
            </span>
          </>
        )}

        {findings.length > 0 && (
          <>
            <span className="text-faint">Found</span>
            <ul className={`min-w-0 space-y-0.5 ${open ? "max-h-64 overflow-y-auto" : ""}`}>
              {findings.map((finding, index) => (
                <li key={`${finding.ts}-${index}`} className="flex min-w-0 items-baseline gap-2">
                  <span className={`text-muted min-w-0 flex-1 ${wrap}`} title={open ? undefined : finding.text}>
                    {finding.text}
                  </span>
                  {open && <span className="text-faint shrink-0 tabular-nums">{relativeTime(finding.ts, now)}</span>}
                </li>
              ))}
            </ul>
          </>
        )}

        <span className="col-start-3 row-start-1 flex items-center gap-0.5 self-center">
          {open && <CopyButton value={campaignText(view)} label="Copy the log" icon />}
          <button
            type="button"
            onClick={() => {
              setNow(Date.now());
              setOpen((value) => !value);
            }}
            aria-expanded={open}
            title={open ? "Show less" : "Show every finding"}
            className="text-faint hover:text-body hover:bg-inset inline-flex h-5 shrink-0 cursor-pointer items-center gap-1 rounded px-1 tabular-nums touch:min-h-11"
          >
            {!open && hidden > 0 && <span>+{hidden}</span>}
            <span aria-hidden="true" className={`inline-block transition-transform ${open ? "-rotate-90" : "rotate-90"}`}>
              ›
            </span>
          </button>
        </span>
      </section>
    </div>
  );
});
