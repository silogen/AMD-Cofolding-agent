// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { toggleStar, useShortlist, useShortlistSession } from "../../lib/shortlist";

interface Props {
  smiles: string;
  label?: string | null;
  /** Where it is starred from, kept with the entry. */
  source?: string;
  /** Scores to remember, read when the star is set. */
  values?: () => Record<string, number> | undefined;
  /** Defaults to the open conversation. */
  session?: string;
  className?: string;
}

/** Adds a molecule to the conversation's shortlist, or takes it off. */
export function StarButton({ smiles, label, source, values, session, className = "" }: Props) {
  const fallback = useShortlistSession();
  const key = session ?? fallback;
  const on = useShortlist(key).some((e) => e.smiles === smiles);
  return (
    <button
      type="button"
      aria-pressed={on}
      aria-label={on ? "Remove from shortlist" : "Add to shortlist"}
      title={on ? "On the shortlist" : "Add to shortlist"}
      onClick={(event) => {
        event.stopPropagation();
        toggleStar(key, { smiles, ...(label ? { label } : {}), ...(source ? { source } : {}), ...(on ? {} : { values: values?.() }) });
      }}
      className={`grid size-5 shrink-0 cursor-pointer place-items-center rounded ${on ? "" : "text-faint hover:text-body"} ${className}`}
      style={on ? { color: "var(--color-warn)" } : undefined}
    >
      <svg width={12} height={12} viewBox="0 0 12 12" aria-hidden="true">
        <path
          d="M6 1.2l1.45 3 3.3.42-2.42 2.27.62 3.27L6 8.58 3.05 10.16l.62-3.27L1.25 4.62l3.3-.42z"
          fill={on ? "currentColor" : "none"}
          stroke="currentColor"
          strokeWidth={1}
          strokeLinejoin="round"
        />
      </svg>
    </button>
  );
}
