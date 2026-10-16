// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useRef, useState } from "react";

interface Props {
  onSend: (text: string) => void;
  disabled: boolean;
  /** Shown when the composer is disabled, so the reason is never a mystery. */
  disabledReason?: string;
}

/** Example prompts, offered only on an empty conversation. */
const SUGGESTIONS = [
  "What scoring components and models are available?",
  "Design 20 ligands for BRD4 and score their predicted affinity with Boltz.",
  "Take the top 5 candidates and check their ADMET properties.",
];

export function Composer({ onSend, disabled, disabledReason }: Props) {
  const [text, setText] = useState("");
  const [showSuggestions, setShowSuggestions] = useState(true);
  const areaRef = useRef<HTMLTextAreaElement>(null);

  // Grow with the text, up to a point, so a long prompt stays readable
  // without pushing the conversation off the screen.
  useEffect(() => {
    const area = areaRef.current;
    if (!area) return;
    area.style.height = "auto";
    area.style.height = `${Math.min(area.scrollHeight, 200)}px`;
  }, [text]);

  function submit() {
    const content = text.trim();
    if (!content || disabled) return;
    onSend(content);
    setText("");
    setShowSuggestions(false);
  }

  return (
    // The safe-area inset keeps the send button clear of the home indicator
    // on a notched phone. It is zero on every other device.
    <div
      className="border-subtle bg-page border-t"
      style={{ paddingBottom: "env(safe-area-inset-bottom)" }}
    >
      <div className="mx-auto max-w-3xl px-3 py-2 sm:px-4 sm:py-3">
        {showSuggestions && (
          <div className="mb-2 flex flex-wrap gap-2">
            {SUGGESTIONS.map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                disabled={disabled}
                onClick={() => {
                  setText(suggestion);
                  areaRef.current?.focus();
                }}
                className="bg-raised border-subtle text-muted hover:text-body cursor-pointer rounded-full border px-3 py-1 text-xs disabled:cursor-not-allowed disabled:opacity-50 touch:min-h-11"
              >
                {suggestion}
              </button>
            ))}
          </div>
        )}

        <div className="bg-raised border-subtle shadow-card flex items-end gap-2 rounded-2xl border p-2">
          <textarea
            ref={areaRef}
            value={text}
            rows={1}
            disabled={disabled}
            onChange={(event) => setText(event.target.value)}
            onKeyDown={(event) => {
              // Enter sends; Shift+Enter makes a new line. Prompts here are
              // often multi-line, so the escape hatch matters.
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                submit();
              }
            }}
            placeholder={disabled ? (disabledReason ?? "Not connected") : "Ask the agent to design or score a molecule…"}
            aria-label="Message the agent"
            /*
              text-base, not a smaller size: iOS Safari zooms the page when a
              field with text under 16px takes focus, and the zoom is not
              undone when the keyboard closes.
            */
            className="max-h-[200px] min-h-[24px] flex-1 resize-none bg-transparent px-2 py-1.5 text-base outline-none disabled:cursor-not-allowed sm:text-[0.9375rem]"
          />
          <button
            type="button"
            onClick={submit}
            disabled={disabled || !text.trim()}
            aria-label="Send"
            className="grid size-9 shrink-0 cursor-pointer place-items-center rounded-xl text-white transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40 touch:size-11"
            style={{ background: "var(--color-accent)" }}
          >
            <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth="2.2" aria-hidden="true">
              <path d="M12 19V5M5 12l7-7 7 7" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        </div>

        <p className="text-faint mt-1.5 text-center text-[11px]">
          The agent runs real scoring jobs. Check results before acting on them.
        </p>
      </div>
    </div>
  );
}
