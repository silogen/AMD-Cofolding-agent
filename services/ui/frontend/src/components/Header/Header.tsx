// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { ConnectionState } from "../../types/api";
import type { Theme } from "../../hooks/useTheme";

interface Props {
  theme: Theme;
  onToggleTheme: () => void;
  connection: ConnectionState;
  model: string | null;
  toolCount: number;
  onReconnect: () => void;
  workspaceOpen: boolean;
  onToggleWorkspace: () => void;
}

const CONNECTION_LABEL: Record<ConnectionState, string> = {
  connecting: "Connecting",
  open: "Connected",
  closed: "Reconnecting",
  error: "Disconnected",
};

export function Header({
  theme,
  onToggleTheme,
  connection,
  model,
  toolCount,
  onReconnect,
  workspaceOpen,
  onToggleWorkspace,
}: Props) {
  const healthy = connection === "open";

  return (
    <header className="border-subtle bg-raised flex shrink-0 items-center gap-3 border-b px-4 py-2.5">
      {/*
        On a phone the full title truncates, so the short name is shown
        instead and the full title returns once there is room for it.
      */}
      <div className="min-w-0">
        <h1 className="truncate text-sm font-semibold">
          <span className="hidden sm:inline">AI Cofolding Drug Design Agent</span>
          <span className="sm:hidden">Cofolding Agent</span>
        </h1>
        <p className="text-faint truncate text-[11px]">
          {model ? `${model} · ${toolCount} tools` : "Starting the agent…"}
        </p>
      </div>

      <div className="ml-auto flex items-center gap-1 sm:gap-2">
        {/*
          On a phone the connection state is a dot with an accessible label.
          The word costs a third of the header width and says nothing the
          colour does not already say.
        */}
        <button
          type="button"
          onClick={healthy ? undefined : onReconnect}
          disabled={healthy}
          title={healthy ? "Connected to the agent" : "Try to reconnect"}
          aria-label={CONNECTION_LABEL[connection]}
          className="bg-inset flex items-center gap-1.5 rounded-full px-3 py-1 text-[11px] font-medium disabled:cursor-default touch:min-h-11 touch:min-w-11 touch:justify-center"
        >
          <span
            aria-hidden="true"
            className={`size-1.5 rounded-full ${connection === "connecting" || connection === "closed" ? "animate-pulse" : ""}`}
            style={{ background: healthy ? "var(--color-molecule)" : "var(--color-accent)" }}
          />
          <span className="hidden sm:inline">{CONNECTION_LABEL[connection]}</span>
        </button>

        <button
          type="button"
          onClick={onToggleWorkspace}
          aria-pressed={workspaceOpen}
          title={workspaceOpen ? "Hide the structure viewer" : "Show the structure viewer"}
          className="bg-inset text-muted hover:text-body hidden cursor-pointer rounded-lg px-2.5 py-1.5 text-[11px] font-medium lg:block"
        >
          {workspaceOpen ? "Hide viewer" : "Show viewer"}
        </button>

        <button
          type="button"
          onClick={onToggleTheme}
          aria-label={theme === "dark" ? "Switch to the light theme" : "Switch to the dark theme"}
          title={theme === "dark" ? "Switch to the light theme" : "Switch to the dark theme"}
          className="bg-inset text-muted hover:text-body grid size-8 cursor-pointer place-items-center rounded-lg touch:size-11"
        >
          {theme === "dark" ? <SunIcon /> : <MoonIcon />}
        </button>
      </div>
    </header>
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
