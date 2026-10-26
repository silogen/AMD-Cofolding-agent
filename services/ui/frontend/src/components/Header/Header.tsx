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

/** The model name without its organisation prefix: "Qwen3.8-27B", not "Qwen/Qwen3.8-27B". */
function shortModel(model: string): string {
  return model.split("/").pop() ?? model;
}

export function Header({
  theme,
  onToggleTheme,
  connection,
  agent,
  gatewayError,
  title,
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
    <header className="border-subtle bg-raised flex shrink-0 items-center gap-3 border-b px-3 py-2">
      <IconButton label={railOpen ? "Hide conversations" : "Show conversations"} pressed={railOpen} onClick={onToggleRail}>
        <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
          <rect x="3" y="4" width="18" height="16" rx="2" />
          <path d="M9 4v16" />
        </svg>
      </IconButton>

      <div className="flex min-w-0 items-center gap-2">
        <span aria-hidden="true" className="h-5 w-1 shrink-0 rounded-full" style={{ background: "var(--color-accent)" }} />
        <div className="min-w-0">
          <h1 className="truncate text-sm font-semibold">
            <span className="hidden md:inline">AI Cofolding Drug Design Agent</span>
            <span className="md:hidden">Cofolding Agent</span>
          </h1>
          <p className="text-faint truncate text-[11px]" title={title}>
            {title || "New conversation"}
            {busy && <span style={{ color: "var(--color-accent)" }}> · working</span>}
          </p>
        </div>
      </div>

      <div className="ml-auto flex items-center gap-1 sm:gap-2">
        {agent && (
          <div className="text-muted hidden items-center gap-1.5 text-[11px] xl:flex">
            <Chip title={`${agent.backend} backend`}>{shortModel(agent.model)}</Chip>
            {agent.reasoningEffort && <Chip title="Reasoning effort">effort {agent.reasoningEffort}</Chip>}
            <Chip title={agent.tools.join(", ")}>{agent.tools.length} tools</Chip>
          </div>
        )}

        <button
          type="button"
          onClick={onOpenPalette}
          title="Commands (Ctrl+K)"
          className="bg-inset text-muted hover:text-body hidden cursor-pointer items-center gap-2 rounded-lg px-2.5 py-1.5 text-[11px] font-medium sm:flex"
        >
          Commands
          <kbd className="border-subtle rounded border px-1 font-mono text-[10px]">Ctrl K</kbd>
        </button>

        <button
          type="button"
          onClick={healthy ? undefined : onReconnect}
          disabled={healthy}
          title={status ?? (healthy ? "Connected to the agent" : "Try to reconnect")}
          aria-label={CONNECTION_LABEL[connection]}
          className="bg-inset flex items-center gap-1.5 rounded-full px-3 py-1 text-[11px] font-medium disabled:cursor-default touch:min-h-11 touch:min-w-11 touch:justify-center"
        >
          <span
            aria-hidden="true"
            className={`size-1.5 rounded-full ${connection === "connecting" || connection === "closed" ? "animate-pulse" : ""}`}
            style={{ background: healthy && !gatewayError ? "var(--color-molecule)" : "var(--color-accent)" }}
          />
          <span className="hidden sm:inline">{CONNECTION_LABEL[connection]}</span>
        </button>

        <IconButton
          label={workbenchOpen ? "Hide the workbench" : "Show the workbench"}
          shortcut="Ctrl+B"
          pressed={workbenchOpen}
          onClick={onToggleWorkbench}
        >
          <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
            <rect x="3" y="4" width="18" height="16" rx="2" />
            <path d="M15 4v16" />
          </svg>
        </IconButton>

        <IconButton label={theme === "dark" ? "Switch to the light theme" : "Switch to the dark theme"} shortcut="Ctrl+J" onClick={onToggleTheme}>
          {theme === "dark" ? <SunIcon /> : <MoonIcon />}
        </IconButton>
      </div>
    </header>
  );
}

function Chip({ children, title }: { children: ReactNode; title?: string }) {
  return (
    <span title={title} className="bg-inset rounded-full px-2 py-0.5 font-mono">
      {children}
    </span>
  );
}

function IconButton({
  label,
  shortcut,
  pressed,
  onClick,
  children,
}: {
  label: string;
  shortcut?: string;
  pressed?: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      aria-pressed={pressed}
      title={shortcut ? `${label} (${shortcut})` : label}
      className="bg-inset text-muted hover:text-body grid size-8 shrink-0 cursor-pointer place-items-center rounded-lg touch:size-11"
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
