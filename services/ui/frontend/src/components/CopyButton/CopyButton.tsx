// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useCallback, useEffect, useRef, useState } from "react";

interface Props {
  value: string;
  label?: string;
}

/** Copy a value to the clipboard and confirm it for a moment. */
export function CopyButton({ value, label = "Copy" }: Props) {
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

  return (
    <button
      type="button"
      onClick={copy}
      title={label}
      aria-label={label}
      className="text-muted hover:text-body hover:bg-inset shrink-0 cursor-pointer rounded-md px-2 py-1 text-[11px] transition-colors touch:min-h-11 touch:px-3"
    >
      {copied ? "Copied" : label}
    </button>
  );
}
