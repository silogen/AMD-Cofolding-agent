// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useMemo, useState, type ReactNode } from "react";
import type { SessionSummary } from "../../types/api";
import type { AgentInfo } from "../../hooks/useSession";
import { download, sessionEvents } from "../../lib/api";
import { downloadReport, printReport, sessionToMarkdown, slug } from "../../lib/exportSession";
import { SessionDataDialog } from "../SessionData/SessionDataDialog";

interface Props {
  sessions: SessionSummary[];
  activeId: string | null;
  error: string | null;
  agent: AgentInfo | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onRename: (id: string, title: string) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
  onClose: () => void;
}

/** Day buckets, newest first. */
function groupByDay(sessions: SessionSummary[]): { label: string; items: SessionSummary[] }[] {
  const startOfToday = new Date();
  startOfToday.setHours(0, 0, 0, 0);
  const day = 86_400_000;
  const buckets: [string, (t: number) => boolean][] = [
    ["Today", (t) => t >= startOfToday.getTime()],
    ["Yesterday", (t) => t >= startOfToday.getTime() - day],
    ["Previous 7 days", (t) => t >= startOfToday.getTime() - 7 * day],
    ["Older", () => true],
  ];
  const groups = buckets.map(([label]) => ({ label, items: [] as SessionSummary[] }));
  for (const session of sessions) {
    const time = Date.parse(session.updated);
    const index = buckets.findIndex(([, test]) => test(time));
    groups[index].items.push(session);
  }
  return groups.filter((group) => group.items.length > 0);
}

function timeLabel(iso: string): string {
  const date = new Date(iso);
  const today = new Date();
  if (date.toDateString() === today.toDateString()) {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
  return date.toLocaleDateString([], { day: "numeric", month: "short" });
}

/**
 * The saved conversations.
 *
 * Each one is a working session on a target that can run for hours, so the
 * list is a place to come back to, not a history to scroll: grouped by day,
 * searchable, and each entry can be renamed, exported or deleted.
 */
export function SessionRail({ sessions, activeId, error, agent, onSelect, onNew, onRename, onDelete, onClose }: Props) {
  const [query, setQuery] = useState("");
  const [dataOf, setDataOf] = useState<SessionSummary | null>(null);

  const groups = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const shown = needle ? sessions.filter((s) => (s.title || "untitled").toLowerCase().includes(needle)) : sessions;
    return groupByDay(shown);
  }, [sessions, query]);

  return (
    <nav aria-label="Conversations" className="border-subtle bg-sunken flex h-full w-60 shrink-0 flex-col border-r">
      <div className="flex h-14 shrink-0 items-center justify-between px-4">
        <span className="flex items-center gap-2" title="AI Cofolding Drug Design Agent">
          <AmdLogo />
          <span className="text-faint text-[10px] leading-tight font-semibold tracking-wide uppercase">
            Cofolding
            <br />
            agent
          </span>
        </span>
        <button
          type="button"
          onClick={onClose}
          aria-label="Hide conversations"
          title="Hide conversations"
          className="text-muted hover:text-body grid size-8 cursor-pointer place-items-center rounded-lg text-sm"
        >
          ‹
        </button>
      </div>

      <div className="px-3 pb-2">
        <button
          type="button"
          onClick={onNew}
          className="flex min-h-9 w-full cursor-pointer items-center justify-center gap-2 rounded-md px-3 text-xs font-semibold text-white hover:opacity-90"
          style={{ background: "var(--color-accent)" }}
        >
          <svg viewBox="0 0 24 24" className="size-3.5" fill="none" stroke="currentColor" strokeWidth="2.4" aria-hidden="true">
            <path d="M12 5v14M5 12h14" strokeLinecap="round" />
          </svg>
          New conversation
        </button>
      </div>

      <div className="px-3 pb-2">
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search conversations"
          aria-label="Search conversations"
          className="bg-raised border-subtle w-full rounded-lg border px-2.5 py-1.5 text-xs outline-none focus:border-strong"
        />
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
        {error && <p className="text-muted px-2 py-2 text-[11px]">Could not load the list: {error}</p>}
        {!error && groups.length === 0 && (
          <p className="text-faint px-2 py-4 text-center text-[11px]">
            {query ? "No conversation matches." : "Conversations you start are saved here."}
          </p>
        )}
        {groups.map((group) => (
          <section key={group.label} className="mt-2">
            <h3 className="text-faint px-2 pb-1 text-[10px] font-semibold tracking-wide uppercase">{group.label}</h3>
            <ul>
              {group.items.map((session) => (
                <SessionItem
                  key={session.id}
                  session={session}
                  active={session.id === activeId}
                  onSelect={() => onSelect(session.id)}
                  onRename={(title) => onRename(session.id, title)}
                  onDelete={() => onDelete(session.id)}
                  onData={() => setDataOf(session)}
                />
              ))}
            </ul>
          </section>
        ))}
      </div>

      {!agent && <footer className="border-subtle text-faint shrink-0 border-t px-4 py-3 text-[10px]">Starting the agent…</footer>}

      {dataOf && (
        <SessionDataDialog
          key={dataOf.id}
          session={sessions.find((s) => s.id === dataOf.id) ?? dataOf}
          onClose={() => setDataOf(null)}
          onDelete={() => onDelete(dataOf.id)}
        />
      )}
    </nav>
  );
}

/** The AMD wordmark with its arrow, drawn in the current text colour. */
function AmdLogo() {
  return (
    <svg viewBox="0 0 120 30" className="text-body h-[18px] w-auto" aria-label="AMD" role="img">
      <text x="0" y="25" fill="currentColor" fontFamily="Arial, Helvetica, sans-serif" fontWeight="700" fontSize="30" letterSpacing="-1">
        AMD
      </text>
      <path d="M92 3h25v25l-7.5-7.5V10.5H99.5Z M92 13.5l6-6v13h13l-6 6H92Z" fill="currentColor" />
    </svg>
  );
}

function SessionItem({
  session,
  active,
  onSelect,
  onRename,
  onDelete,
  onData,
}: {
  session: SessionSummary;
  active: boolean;
  onSelect: () => void;
  onRename: (title: string) => Promise<void>;
  onDelete: () => Promise<void>;
  onData: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(session.title);
  const [menu, setMenu] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const [preparing, setPreparing] = useState(false);
  const title = session.title || "Untitled conversation";

  const fail = (reason: unknown) => setProblem(reason instanceof Error ? reason.message : String(reason));

  async function commit() {
    setEditing(false);
    const next = draft.trim();
    if (next && next !== session.title) await onRename(next).catch(fail);
  }

  async function exportAs(format: "md" | "json") {
    setMenu(false);
    try {
      const { session: summary, events } = await sessionEvents(session.id);
      const name = slug(summary.title, summary.id);
      if (format === "md") download(`${name}.md`, sessionToMarkdown(summary, events), "text/markdown");
      else download(`${name}.json`, JSON.stringify({ session: summary, events }, null, 2), "application/json");
    } catch (reason) {
      fail(reason);
    }
  }

  async function report(print: boolean) {
    setMenu(false);
    setProblem(null);
    setPreparing(true);
    try {
      await (print ? printReport(session.id) : downloadReport(session.id));
    } catch (reason) {
      fail(reason);
    } finally {
      setPreparing(false);
    }
  }

  async function remove() {
    setMenu(false);
    if (!window.confirm(`Delete "${title}"? This cannot be undone.`)) return;
    await onDelete().catch(fail);
  }

  if (editing) {
    return (
      <li className="px-1 py-0.5">
        <input
          autoFocus
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onBlur={commit}
          onKeyDown={(event) => {
            if (event.key === "Enter") void commit();
            if (event.key === "Escape") {
              setDraft(session.title);
              setEditing(false);
            }
          }}
          aria-label="Conversation title"
          className="bg-raised border-strong w-full rounded-md border px-2 py-1 text-xs outline-none"
        />
      </li>
    );
  }

  return (
    <li className="group relative">
      <button
        type="button"
        onClick={onSelect}
        onDoubleClick={() => {
          setDraft(session.title);
          setEditing(true);
        }}
        aria-current={active || undefined}
        title={title}
        className={`flex w-full cursor-pointer items-center gap-2 rounded-md py-1.5 pr-7 pl-2 text-left text-xs touch:min-h-11 ${
          active ? "bg-raised shadow-card font-semibold" : "hover:bg-inset"
        }`}
      >
        {active && <span aria-hidden="true" className="h-7 w-0.5 shrink-0 rounded-full" style={{ background: "var(--color-accent)" }} />}
        <span className="min-w-0 flex-1">
          <span className="block truncate">{title}</span>
          <span className="text-faint block text-[10px] font-normal">
            {session.turns} turn{session.turns === 1 ? "" : "s"}
          </span>
        </span>
        {session.busy ? (
          <span
            title="Working"
            aria-label="Working"
            className="size-1.5 shrink-0 animate-pulse rounded-full"
            style={{ background: "var(--color-accent)" }}
          />
        ) : (
          <span className="text-faint shrink-0 self-start pt-0.5 text-[10px] font-normal">{timeLabel(session.updated)}</span>
        )}
      </button>

      <button
        type="button"
        onClick={() => setMenu((open) => !open)}
        aria-label={`Actions for ${title}`}
        aria-expanded={menu}
        className="text-muted hover:text-body absolute top-1/2 right-1 grid size-6 -translate-y-1/2 cursor-pointer place-items-center rounded opacity-0 group-hover:opacity-100 focus:opacity-100 aria-expanded:opacity-100 touch:opacity-100"
      >
        ⋯
      </button>

      {menu && (
        <div
          role="menu"
          className="bg-raised border-subtle shadow-card absolute top-full right-1 z-20 mt-0.5 w-40 overflow-hidden rounded-lg border py-1 text-xs"
          onMouseLeave={() => setMenu(false)}
        >
          <MenuItem
            onClick={() => {
              setMenu(false);
              setDraft(session.title);
              setEditing(true);
            }}
          >
            Rename
          </MenuItem>
          <MenuItem onClick={() => report(false)} disabled={preparing}>
            Export report
          </MenuItem>
          <MenuItem onClick={() => report(true)} disabled={preparing}>
            Print / Save as PDF
          </MenuItem>
          <MenuItem onClick={() => exportAs("md")}>Export as Markdown</MenuItem>
          <MenuItem onClick={() => exportAs("json")}>Export as JSON</MenuItem>
          <MenuItem
            onClick={() => {
              setMenu(false);
              onData();
            }}
          >
            Data and cleanup…
          </MenuItem>
          <MenuItem onClick={remove} danger disabled={session.busy}>
            {session.busy ? "Delete (stop it first)" : "Delete"}
          </MenuItem>
        </div>
      )}

      {preparing && <p className="text-faint px-2 pb-1 text-[10px]">Preparing the report…</p>}
      {problem && (
        <p role="alert" className="px-2 pb-1 text-[10px]" style={{ color: "var(--color-accent)" }}>
          {problem}
        </p>
      )}
    </li>
  );
}

function MenuItem({
  children,
  onClick,
  danger,
  disabled,
}: {
  children: ReactNode;
  onClick: () => void;
  danger?: boolean;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      role="menuitem"
      onClick={onClick}
      disabled={disabled}
      className="hover:bg-sunken block w-full cursor-pointer px-3 py-1.5 text-left disabled:cursor-not-allowed disabled:opacity-50"
      style={danger ? { color: "var(--color-accent)" } : undefined}
    >
      {children}
    </button>
  );
}
