// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useState } from "react";
import { explainScore, loadTtp, type ScoreValues } from "../../lib/scoreWhy";
import type { Ttp } from "../../lib/ttp";

interface Props {
  values: ScoreValues;
  /** The TTP itself when known, else its id (and version) to fetch it. */
  ttp?: Ttp | null;
  ttpId?: string | null;
  ttpVersion?: number | null;
  /** A heading, shown only when there is something to explain. */
  title?: string;
}

const tone = (d: number | null) => (d === null ? "var(--text-faint)" : d >= 0.7 ? "var(--color-ok)" : d >= 0.4 ? "var(--color-warn)" : "var(--color-accent)");
const short = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));

/**
 * Why a molecule scored what it did: one bar per objective, its desirability
 * after the TTP's transform, worst drag on the total first, and a verdict.
 */
export function ScoreWhy({ values, ttp: given, ttpId, ttpVersion, title }: Props) {
  const key = given ? null : ttpId ? `${ttpId}@${ttpVersion ?? ""}` : null;
  const [loaded, setLoaded] = useState<{ key: string; ttp: Ttp | null } | null>(null);
  useEffect(() => {
    if (!key || !ttpId) return;
    let cancelled = false;
    loadTtp(ttpId, ttpVersion).then((ttp) => {
      if (!cancelled) setLoaded({ key, ttp });
    });
    return () => {
      cancelled = true;
    };
  }, [key, ttpId, ttpVersion]);

  const ttp = given ?? (loaded && loaded.key === key ? loaded.ttp : null);
  const why = useMemo(() => (ttp ? explainScore(ttp, values) : null), [ttp, values]);
  if (!why) return null;

  return (
    <div className="flex min-w-0 flex-col gap-1 text-[11px]">
      {title && <h4 className="text-muted text-[11px] font-semibold tracking-wide uppercase">{title}</h4>}
      <p className="font-medium">{why.verdict}</p>
      <ul className="flex flex-col gap-0.5">
        {why.items.map((item) => (
          <li key={item.name} className="grid grid-cols-[minmax(0,9rem)_5rem_auto_auto] items-center gap-2">
            <span className="text-muted truncate" title={item.name}>
              {item.name}
            </span>
            <span className="bg-inset h-1 overflow-hidden rounded-full" aria-hidden="true">
              <span className="block h-full rounded-full" style={{ width: `${Math.round((item.desirability ?? 0) * 100)}%`, background: tone(item.desirability) }} />
            </span>
            <span className="font-mono tabular-nums" title={item.raw !== null ? "raw value → desirability" : "desirability"}>
              {item.raw !== null && <span className="text-faint">{short(item.raw)} → </span>}
              {item.desirability === null ? "—" : item.desirability.toFixed(2)}
            </span>
            <span className="text-faint tabular-nums" title="Weight, as a share of all weights">
              w {Math.round(item.share * 100)}%
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
