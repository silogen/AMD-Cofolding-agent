// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useRef, useState } from "react";

interface Props {
  smiles: string;
  /** Shown instead of the SMILES, e.g. a compound name; the SMILES stays in the tooltip and the copy. */
  label?: string | null;
  /** Width and size classes for the text; it is cut to one line within them. */
  className?: string;
}

/**
 * A SMILES on one line, cut with an ellipsis, with the full string on hover
 * and a copy button that shows on hover or focus of the enclosing `group`.
 */
export function SmilesText({ smiles, label, className = "max-w-28 text-[10px]" }: Props) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<number | null>(null);

  useEffect(
    () => () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    },
    [],
  );

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(smiles);
      setCopied(true);
      if (timer.current !== null) window.clearTimeout(timer.current);
      timer.current = window.setTimeout(() => setCopied(false), 1200);
    } catch {
      // Without a secure context the text can still be selected by hand.
    }
  };

  return (
    <span className="flex min-w-0 items-center gap-1">
      <span className={`block min-w-0 truncate font-mono ${className}`} title={smiles}>
        {label || smiles}
      </span>
      <button
        type="button"
        onClick={(event) => {
          event.stopPropagation();
          void copy();
        }}
        title={copied ? "Copied" : "Copy SMILES"}
        aria-label="Copy SMILES"
        className={`text-faint hover:text-body grid size-5 shrink-0 cursor-pointer place-items-center rounded transition-opacity group-hover:opacity-100 focus-visible:opacity-100 touch:size-11 touch:opacity-100 ${
          copied ? "opacity-100" : "opacity-0"
        }`}
      >
        {copied ? (
          <svg viewBox="0 0 16 16" className="size-3" fill="none" stroke="var(--color-ok)" strokeWidth="2" aria-hidden="true">
            <path d="m3.5 8.5 3 3 6-7" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        ) : (
          <svg viewBox="0 0 16 16" className="size-3" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
            <rect x="5.5" y="5.5" width="8" height="8" rx="1.5" />
            <path d="M10.5 3.5v-.5A1.5 1.5 0 0 0 9 1.5H3A1.5 1.5 0 0 0 1.5 3v6A1.5 1.5 0 0 0 3 10.5h.5" />
          </svg>
        )}
      </button>
    </span>
  );
}
