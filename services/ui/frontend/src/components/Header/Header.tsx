// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { ReactNode } from "react";
import type { ConnectionState } from "../../types/api";
import type { Theme } from "../../hooks/useTheme";
import type { AgentInfo } from "../../hooks/useSession";

interface Props {
  theme: Theme;
  onToggleTheme: () => void;
  connection: ConnectionState;
  agent: AgentInfo | null;
  gatewayError: string | null;
  title: string;
  /** The protein this conversation works on, once a tool has named it. */
  target: string | null;
  onShowTarget: () => void;
  busy: boolean;
  onReconnect: () => void;
  railOpen: boolean;
  onToggleRail: () => void;
  workbenchOpen: boolean;
  onToggleWorkbench: () => void;
  onOpenPalette: () => void;
}

const CONNECTION_LABEL: Record<ConnectionState, string> = {
  connecting: "Connecting",
  open: "Connected",
  closed: "Reconnecting",
  error: "Disconnected",
};

/**
 * The head of the chat column: which conversation this is, on which target,
 * and whether the agent can be reached. The product name and the model live
 * in the conversation rail, so this line stays about the work.
 */
export function Header({
  theme,
  onToggleTheme,
  connection,
  agent,
  gatewayError,
  title,
  target,
  onShowTarget,
  busy,
  onReconnect,
  railOpen,
  onToggleRail,
  workbenchOpen,
  onToggleWorkbench,
  onOpenPalette,
}: Props) {
  const healthy = connection === "open";
  const status = gatewayError ?? (agent ? null : "Starting the agent…");

  return (
    <header className="border-subtle flex h-14 shrink-0 items-center gap-3 border-b px-4">
      {!railOpen && (
        <IconButton label="Show conversations" onClick={onToggleRail}>
          <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
            <rect x="3" y="4" width="18" height="16" rx="2" />
            <path d="M9 4v16" />
          </svg>
        </IconButton>
      )}

      <div className="min-w-0 flex-1">
        <h1 className="truncate text-sm font-bold" title={title}>
          {title || "New conversation"}
        </h1>
        <div className="mt-0.5 flex items-center gap-2 text-[11px]">
          {target ? (
            <button
              type="button"
              onClick={onShowTarget}
              title="Show the target"
              className="border-subtle text-muted hover:text-body cursor-pointer rounded-full border px-2 font-semibold"
            >
              {target}
            </button>
          ) : (
            <span className="text-faint">No target yet</span>
          )}
          {busy && <span style={{ color: "var(--color-accent)" }}>working…</span>}
        </div>
      </div>

      <div className="flex shrink-0 items-center gap-1 sm:gap-2">
        <button
          type="button"
          onClick={healthy ? undefined : onReconnect}
          disabled={healthy}
          title={status ?? (healthy ? "Connected to the agent" : "Try to reconnect")}
          aria-label={CONNECTION_LABEL[connection]}
          className="text-muted flex items-center gap-1.5 px-1 text-[11px] disabled:cursor-default touch:min-h-11 touch:min-w-11 touch:justify-center"
        >
          <span
            aria-hidden="true"
            className={`size-1.5 rounded-full ${connection === "connecting" || connection === "closed" ? "animate-pulse" : ""}`}
            style={{ background: healthy && !gatewayError ? "var(--color-ok)" : "var(--color-accent)" }}
          />
          <span className="hidden sm:inline">{CONNECTION_LABEL[connection].toLowerCase()}</span>
        </button>

        <button
          type="button"
          onClick={onOpenPalette}
          title="Commands (Ctrl+K)"
          className="text-muted hover:text-body hidden cursor-pointer items-center gap-1.5 rounded-md px-2 py-1 text-[11px] sm:flex"
        >
          <kbd className="border-subtle rounded border px-1 font-mono text-[10px]">Ctrl K</kbd>
        </button>

        <button
          type="button"
          onClick={onToggleWorkbench}
          aria-pressed={workbenchOpen}
          title={`${workbenchOpen ? "Hide" : "Show"} the workbench (Ctrl+B)`}
          aria-label={workbenchOpen ? "Hide the workbench" : "Show the workbench"}
          className={`cursor-pointer rounded-md px-2.5 py-1 text-xs font-semibold ${workbenchOpen ? "text-body bg-inset" : "text-muted hover:text-body"}`}
        >
          Workbench
        </button>

        <IconButton label={theme === "dark" ? "Switch to the light theme" : "Switch to the dark theme"} shortcut="Ctrl+J" onClick={onToggleTheme}>
          {theme === "dark" ? <SunIcon /> : <MoonIcon />}
        </IconButton>
      </div>
    </header>
  );
}

function IconButton({ label, shortcut, onClick, children }: { label: string; shortcut?: string; onClick: () => void; children: ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      title={shortcut ? `${label} (${shortcut})` : label}
      className="text-muted hover:text-body hover:bg-inset grid size-8 shrink-0 cursor-pointer place-items-center rounded-lg touch:size-11"
    >
      {children}
    </button>
  );
}

function SunIcon() {
  return (
    <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" strokeLinecap="round" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
