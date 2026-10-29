// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useCallback, useEffect, useRef, useState, type MouseEvent } from "react";

interface Props {
  value: string;
  label?: string;
  /** A small square icon instead of the label, which becomes its tooltip; `className` can size it. */
  icon?: boolean;
  className?: string;
}

/** Copy a value to the clipboard and confirm it for a moment. */
export function CopyButton({ value, label = "Copy", icon = false, className = "" }: Props) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<number | null>(null);

  useEffect(() => () => {
    if (timer.current !== null) window.clearTimeout(timer.current);
  }, []);

  const copy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      if (timer.current !== null) window.clearTimeout(timer.current);
      timer.current = window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard access needs a secure context. Without it the user can
      // still select the text, so this is not worth an error message.
    }
  }, [value]);

  if (icon) {
    // 20 px so dense rows keep their height; on touch screens ::after widens the target without taking room.
    return (
      <button
        type="button"
        onClick={(event: MouseEvent) => {
          event.stopPropagation();
          void copy();
        }}
        title={copied ? "Copied" : label}
        aria-label={label}
        className={`text-faint hover:text-body hover:bg-inset relative grid size-5 shrink-0 cursor-pointer place-items-center rounded transition-colors touch:after:absolute touch:after:-inset-2 ${className}`}
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
    );
  }

  return (
    <button
      type="button"
      onClick={copy}
      title={label}
      aria-label={label}
      className={`text-muted hover:text-body hover:bg-inset shrink-0 cursor-pointer rounded-md px-2 py-1 text-[11px] transition-colors touch:min-h-11 touch:px-3 ${className}`}
    >
      {copied ? "Copied" : label}
    </button>
  );
}
