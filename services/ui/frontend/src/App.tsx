// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTheme } from "./hooks/useTheme";
import { useSession, type SessionRequest } from "./hooks/useSession";
import { useSessions } from "./hooks/useSessions";
import { Header } from "./components/Header/Header";
import { SessionRail } from "./components/SessionRail/SessionRail";
import { Transcript } from "./components/Transcript/Transcript";
import { Composer, type ComposerHandle } from "./components/Composer/Composer";
import { Workbench, type WorkbenchTab } from "./components/Workbench/Workbench";
import { CommandPalette, type PaletteItem } from "./components/CommandPalette/CommandPalette";
import type { ViewStyle } from "./components/StructureViewer/StructureViewer";
import type { SessionSummary, StructurePanel } from "./types/api";
import { collectArtifacts } from "./lib/artifacts";
import { availableCommands } from "./lib/commands";
import { download, sessionEvents } from "./lib/api";
import { sessionToMarkdown, slug } from "./lib/exportSession";

/** The conversation in the address, so a reload or a shared link reopens it. */
function sessionFromHash(): string | null {
  const match = /^#\/s\/(s-[0-9-]+-[0-9a-f]+)$/.exec(window.location.hash);
  return match ? match[1] : null;
}

const wide = () => window.matchMedia("(min-width: 1024px)").matches;

/**
 * Conversations, chat and workbench.
 *
 * Three columns on a wide screen: the saved conversations, the chat, and the
 * workbench with what the chat produced. On a narrow screen the chat fills
 * the screen and the other two open over it.
 */
export default function App() {
  const { theme, toggle } = useTheme();
  const [request, setRequest] = useState<SessionRequest>(() => ({ id: sessionFromHash(), nonce: 0 }));
  const sessions = useSessions();
  const { refresh: refreshSessions } = sessions;

  const onAttached = useCallback(
    (session: SessionSummary) => {
      const hash = `#/s/${session.id}`;
      if (window.location.hash !== hash) window.history.replaceState(null, "", hash);
      refreshSessions();
    },
    [refreshSessions],
  );

  const chat = useSession(request, onAttached);
  const { transcript, session, connection, agent } = chat;
  const busy = transcript.busy;
  const connected = connection === "open" && !chat.gatewayError;

  const [railOpen, setRailOpen] = useState(wide);
  const [workbenchOpen, setWorkbenchOpen] = useState(true);
  const [sheetOpen, setSheetOpen] = useState(false);
  const [tab, setTab] = useState<WorkbenchTab>("structure");
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [draft, setDraft] = useState("");
  const composerRef = useRef<ComposerHandle>(null);

  const artifacts = useMemo(() => collectArtifacts(transcript.turns), [transcript.turns]);
  const [picked, setPicked] = useState<StructurePanel | null>(null);
  const [style, setStyle] = useState<ViewStyle>("cartoon");
  const [selectedSmiles, setSelectedSmiles] = useState<string | null>(null);
  const activeStructure = picked ?? artifacts.structures[0]?.item ?? null;

  const open = useCallback((id: string | null) => {
    setRequest((previous) => ({ id, nonce: previous.nonce + 1 }));
    setPicked(null);
    setSelectedSmiles(null);
    if (!wide()) setRailOpen(false);
  }, []);

  // Back and forward move between conversations.
  useEffect(() => {
    const onHash = () => {
      const id = sessionFromHash();
      if (id && id !== session?.id) open(id);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, [open, session?.id]);

  // The list shows "working" from the session; refresh it when a turn ends.
  useEffect(() => {
    refreshSessions();
  }, [busy, session?.title, refreshSessions]);

  const showWorkbench = useCallback((next: WorkbenchTab) => {
    setTab(next);
    setWorkbenchOpen(true);
    if (!wide()) setSheetOpen(true);
  }, []);

  const openStructure = useCallback(
    (panel: StructurePanel) => {
      setPicked(panel);
      showWorkbench("structure");
    },
    [showWorkbench],
  );

  const openMolecule = useCallback(
    (smiles: string) => {
      setSelectedSmiles(smiles);
      showWorkbench("molecules");
    },
    [showWorkbench],
  );

  const ask = useCallback((text: string) => {
    setSheetOpen(false);
    composerRef.current?.fill(text);
  }, []);

  const exportCurrent = useCallback(
    async (format: "md" | "json") => {
      if (!session) return;
      const { session: summary, events } = await sessionEvents(session.id);
      const name = slug(summary.title, summary.id);
      if (format === "md") download(`${name}.md`, sessionToMarkdown(summary, events), "text/markdown");
      else download(`${name}.json`, JSON.stringify({ session: summary, events }, null, 2), "application/json");
    },
    [session],
  );

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setPaletteOpen((value) => !value);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const paletteItems = useMemo<PaletteItem[]>(() => {
    const items: PaletteItem[] = [
      { id: "new", group: "Conversation", label: "New conversation", run: () => open(null) },
      ...(busy ? [{ id: "stop", group: "Conversation", label: "Stop the agent", run: chat.cancel }] : []),
      ...(session
        ? [
            { id: "export-md", group: "Conversation", label: "Export as Markdown", run: () => void exportCurrent("md") },
            { id: "export-json", group: "Conversation", label: "Export as JSON", run: () => void exportCurrent("json") },
          ]
        : []),
      { id: "tab-structure", group: "Workbench", label: "Show the structure", run: () => showWorkbench("structure") },
      { id: "tab-molecules", group: "Workbench", label: "Show molecules", run: () => showWorkbench("molecules") },
      { id: "tab-tables", group: "Workbench", label: "Show tables", run: () => showWorkbench("tables") },
      { id: "tab-activity", group: "Workbench", label: "Show tool activity", run: () => showWorkbench("activity") },
      { id: "rail", group: "View", label: railOpen ? "Hide conversations" : "Show conversations", run: () => setRailOpen((v) => !v) },
      {
        id: "workbench",
        group: "View",
        label: workbenchOpen ? "Hide the workbench" : "Show the workbench",
        run: () => setWorkbenchOpen((v) => !v),
      },
      { id: "theme", group: "View", label: theme === "dark" ? "Light theme" : "Dark theme", run: toggle },
      ...availableCommands(agent?.tools ?? []).map((command) => ({
        id: `cmd-${command.cmd}`,
        group: "Prompts",
        label: command.desc,
        hint: command.cmd,
        run: () => ask(command.template),
      })),
      ...sessions.sessions.map((entry) => ({
        id: `s-${entry.id}`,
        group: "Open conversation",
        label: entry.title || "Untitled conversation",
        hint: new Date(entry.updated).toLocaleDateString(),
        run: () => open(entry.id),
      })),
    ];
    return items;
  }, [open, busy, chat.cancel, session, exportCurrent, showWorkbench, railOpen, workbenchOpen, theme, toggle, agent, ask, sessions.sessions]);

  const workbenchProps = {
    tab,
    onTabChange: setTab,
    artifacts,
    theme,
    activeStructure,
    onSelectStructure: setPicked,
    style,
    onStyleChange: setStyle,
    selectedSmiles,
    onSelectSmiles: setSelectedSmiles,
    onAsk: ask,
  };

  const rail = (
    <SessionRail
      sessions={sessions.sessions}
      activeId={session?.id ?? null}
      error={sessions.error}
      onSelect={(id) => open(id)}
      onNew={() => open(null)}
      onRename={sessions.rename}
      onDelete={async (id) => {
        await sessions.remove(id);
        if (id === session?.id) open(null);
      }}
      onClose={wide() ? undefined : () => setRailOpen(false)}
    />
  );

  return (
    <div className="bg-page flex h-full flex-col">
      <Header
        theme={theme}
        onToggleTheme={toggle}
        connection={connection}
        agent={agent}
        gatewayError={chat.gatewayError}
        title={session?.title ?? ""}
        busy={busy}
        onReconnect={chat.reconnect}
        railOpen={railOpen}
        onToggleRail={() => setRailOpen((value) => !value)}
        workbenchOpen={wide() ? workbenchOpen : sheetOpen}
        onToggleWorkbench={() => (wide() ? setWorkbenchOpen((value) => !value) : setSheetOpen((value) => !value))}
        onOpenPalette={() => setPaletteOpen(true)}
      />

      {chat.gatewayError && (
        <p role="alert" className="border-subtle border-b px-4 py-1.5 text-center text-xs" style={{ background: "var(--color-accent-soft)" }}>
          {chat.gatewayError}
        </p>
      )}

      <div className="relative flex min-h-0 flex-1">
        {railOpen && (
          <>
            <div className="hidden lg:flex">{rail}</div>
            <div className="fixed inset-0 z-40 flex lg:hidden">
              {rail}
              <button type="button" aria-label="Close" className="flex-1 bg-black/40" onClick={() => setRailOpen(false)} />
            </div>
          </>
        )}

        <main className="flex min-w-0 flex-1 flex-col">
          <div className="relative flex min-h-0 flex-1 flex-col">
            <Transcript
              turns={transcript.turns}
              theme={theme}
              onOpenStructure={openStructure}
              openStructurePath={activeStructure?.path ?? null}
              onOpenMolecule={openMolecule}
              onPickPrompt={ask}
            />
            {activeStructure && !sheetOpen && (
              <button
                type="button"
                onClick={() => setSheetOpen(true)}
                className="absolute right-4 bottom-4 z-30 flex min-h-11 items-center gap-2 rounded-full px-4 text-xs font-semibold text-white shadow-lg lg:hidden"
                style={{ background: "var(--color-accent)" }}
              >
                Workbench
              </button>
            )}
          </div>
          <Composer
            ref={composerRef}
            value={draft}
            onChange={setDraft}
            onSend={chat.send}
            onStop={chat.cancel}
            busy={busy}
            connected={connected}
            tools={agent?.tools ?? []}
          />
        </main>

        {workbenchOpen && (
          <div className="hidden lg:flex">
            <Workbench layout="sidebar" {...workbenchProps} onClose={() => setWorkbenchOpen(false)} />
          </div>
        )}

        {chat.notices.length > 0 && (
          <div className="pointer-events-none absolute top-3 left-1/2 z-50 flex -translate-x-1/2 flex-col gap-2" aria-live="polite">
            {chat.notices.map((notice) => (
              <Toast key={notice.id} id={notice.id} message={notice.message} onDismiss={chat.dismissNotice} />
            ))}
          </div>
        )}
      </div>

      {sheetOpen && (
        <div className="lg:hidden">
          <Workbench layout="sheet" {...workbenchProps} onClose={() => setSheetOpen(false)} />
        </div>
      )}

      {paletteOpen && <CommandPalette items={paletteItems} onClose={() => setPaletteOpen(false)} />}
    </div>
  );
}

function Toast({ id, message, onDismiss }: { id: number; message: string; onDismiss: (id: number) => void }) {
  useEffect(() => {
    const timer = window.setTimeout(() => onDismiss(id), 6000);
    return () => window.clearTimeout(timer);
  }, [id, onDismiss]);
  return (
    <div className="bg-raised border-subtle shadow-card pointer-events-auto flex items-center gap-3 rounded-lg border px-3 py-2 text-xs">
      <span aria-hidden="true" className="h-4 w-0.5 rounded-full" style={{ background: "var(--color-accent)" }} />
      {message}
      <button type="button" onClick={() => onDismiss(id)} aria-label="Dismiss" className="text-muted hover:text-body cursor-pointer">
        ×
      </button>
    </div>
  );
}
