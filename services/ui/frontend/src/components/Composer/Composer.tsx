// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import { availableCommands, firstPlaceholder, type SlashCommand } from "../../lib/commands";

interface Props {
  value: string;
  onChange: (text: string) => void;
  onSend: (text: string) => boolean;
  onStop: () => void;
  busy: boolean;
  connected: boolean;
  tools: string[];
}

export interface ComposerHandle {
  /** Put *text* in the box, focus it and select its first placeholder. */
  fill: (text: string) => void;
}

/**
 * The message box.
 *
 * Typing "/" opens the commands, which insert a prompt template with its
 * placeholders selected one after another (Tab jumps to the next one).
 * While the agent works the send button becomes Stop: a turn can run for an
 * hour, and stopping must not need a page reload.
 */
export const Composer = forwardRef<ComposerHandle, Props>(function Composer(
  { value, onChange, onSend, onStop, busy, connected, tools },
  ref,
) {
  const areaRef = useRef<HTMLTextAreaElement>(null);
  const [highlight, setHighlight] = useState(0);
  const [dismissed, setDismissed] = useState(false);

  const commands = useMemo(() => availableCommands(tools), [tools]);
  const slash = /^\/\S*$/.test(value) && !dismissed;
  const matches = slash ? commands.filter((command) => command.cmd.startsWith(value.toLowerCase())) : [];

  useEffect(() => {
    const area = areaRef.current;
    if (!area) return;
    area.style.height = "auto";
    area.style.height = `${Math.min(area.scrollHeight, 220)}px`;
  }, [value]);

  const selectPlaceholder = (text: string, from = 0) => {
    const area = areaRef.current;
    if (!area) return false;
    const found = firstPlaceholder(text.slice(from));
    if (!found) return false;
    area.setSelectionRange(from + found[0], from + found[1]);
    return true;
  };

  const fill = (text: string) => {
    onChange(text);
    setDismissed(true);
    requestAnimationFrame(() => {
      areaRef.current?.focus();
      if (!selectPlaceholder(text)) areaRef.current?.setSelectionRange(text.length, text.length);
    });
  };

  useImperativeHandle(ref, () => ({ fill }));

  function choose(command: SlashCommand) {
    fill(command.template);
  }

  function submit() {
    if (busy || !connected) return;
    const content = value.trim();
    if (!content) return;
    if (firstPlaceholder(content) && !window.confirm("The message still has a {PLACEHOLDER}. Send it anyway?")) return;
    if (onSend(content)) onChange("");
  }

  const placeholder = !connected
    ? "Waiting for the agent…"
    : busy
      ? "The agent is working. You can write your next message."
      : "Ask the agent, or type / for commands";

  return (
    <div className="border-subtle bg-page border-t" style={{ paddingBottom: "env(safe-area-inset-bottom)" }}>
      <div className="relative mx-auto max-w-3xl px-3 py-2 sm:px-4 sm:py-3">
        {matches.length > 0 && (
          <ul
            role="listbox"
            aria-label="Commands"
            className="bg-raised border-subtle shadow-card absolute right-3 bottom-full left-3 mb-1 max-h-72 overflow-y-auto rounded-xl border py-1 sm:right-4 sm:left-4"
          >
            {matches.map((command, index) => (
              <li key={command.cmd} role="option" aria-selected={index === highlight}>
                <button
                  type="button"
                  onMouseDown={(event) => {
                    event.preventDefault();
                    choose(command);
                  }}
                  onMouseEnter={() => setHighlight(index)}
                  className={`flex w-full cursor-pointer items-baseline gap-3 px-3 py-1.5 text-left text-xs ${index === highlight ? "bg-sunken" : ""}`}
                >
                  <span className="w-24 shrink-0 font-mono font-semibold" style={{ color: "var(--color-accent)" }}>
                    {command.cmd}
                  </span>
                  <span className="text-muted">{command.desc}</span>
                </button>
              </li>
            ))}
          </ul>
        )}

        <div className="bg-raised border-subtle shadow-card flex items-end gap-2 rounded-2xl border p-2 focus-within:border-strong">
          <textarea
            ref={areaRef}
            value={value}
            rows={1}
            disabled={!connected}
            onChange={(event) => {
              onChange(event.target.value);
              setDismissed(false);
              setHighlight(0);
            }}
            onKeyDown={(event) => {
              if (matches.length > 0) {
                if (event.key === "ArrowDown" || event.key === "ArrowUp") {
                  event.preventDefault();
                  const step = event.key === "ArrowDown" ? 1 : -1;
                  setHighlight((current) => (current + step + matches.length) % matches.length);
                  return;
                }
                if (event.key === "Enter" || event.key === "Tab") {
                  event.preventDefault();
                  choose(matches[Math.min(highlight, matches.length - 1)]);
                  return;
                }
                if (event.key === "Escape") {
                  setDismissed(true);
                  return;
                }
              }
              if (event.key === "Tab" && !event.shiftKey) {
                const area = event.currentTarget;
                if (selectPlaceholder(value, area.selectionEnd) || selectPlaceholder(value)) event.preventDefault();
                return;
              }
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                submit();
              }
            }}
            placeholder={placeholder}
            aria-label="Message the agent"
            className="max-h-[220px] min-h-[24px] flex-1 resize-none bg-transparent px-2 py-1.5 text-base outline-none disabled:cursor-not-allowed sm:text-[0.9375rem]"
          />
          {busy ? (
            <button
              type="button"
              onClick={onStop}
              aria-label="Stop the agent"
              title="Stop the agent"
              className="border-strong text-body hover:bg-sunken flex h-9 shrink-0 cursor-pointer items-center gap-1.5 rounded-xl border px-3 text-xs font-semibold touch:h-11"
            >
              <span aria-hidden="true" className="size-2.5 rounded-sm" style={{ background: "var(--color-accent)" }} />
              Stop
            </button>
          ) : (
            <button
              type="button"
              onClick={submit}
              disabled={!connected || !value.trim()}
              aria-label="Send"
              className="grid size-9 shrink-0 cursor-pointer place-items-center rounded-xl text-white transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40 touch:size-11"
              style={{ background: "var(--color-accent)" }}
            >
              <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth="2.2" aria-hidden="true">
                <path d="M12 19V5M5 12l7-7 7 7" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
          )}
        </div>

        <p className="text-faint mt-1.5 text-center text-[11px]">
          <kbd className="font-mono">/</kbd> commands · <kbd className="font-mono">Tab</kbd> next placeholder ·{" "}
          <kbd className="font-mono">Shift+Enter</kbd> new line · The agent runs real jobs; check results before acting on them.
        </p>
      </div>
    </div>
  );
});
