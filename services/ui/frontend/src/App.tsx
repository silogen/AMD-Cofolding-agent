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
import type { SessionSummary, StructurePanel } from "./types/api";
import { collectArtifacts, runOf, sameFile, structureTitle } from "./lib/artifacts";
import { availableCommands, VIEW_PREFIX } from "./lib/commands";
import { onOpenRequest, sendToViewer } from "./lib/viewerBus";
import { onWorkbenchRequest } from "./lib/workbenchBus";
import { parseViewerCommand, VIEWER_HELP, type ViewerAction } from "./lib/viewerCommands";
import { download, sessionEvents } from "./lib/api";
import { extractTargetFacts, toolEvents, type TargetFacts } from "./lib/ttp";
import { sessionToMarkdown, slug } from "./lib/exportSession";
import { useUserStructures, withUserStructures } from "./lib/userStructures";

/** The conversation in the address, so a reload or a shared link reopens it. */
function sessionFromHash(): string | null {
  const match = /^#\/s\/(s-[0-9-]+-[0-9a-f]+)$/.exec(window.location.hash);
  return match ? match[1] : null;
}

const wide = () => window.matchMedia("(min-width: 1024px)").matches;

/** How the header names the target: "BRD4", else the TTP id, else the sequence length. */
function targetLabel(facts: TargetFacts): string | null {
  if (facts.gene) return facts.gene;
  if (facts.name) return facts.name.length > 24 ? `${facts.name.slice(0, 23)}…` : facts.name;
  if (facts.ttpId) return facts.ttpId;
  return facts.sequenceLength ? `${facts.sequenceLength} aa protein` : null;
}

/** A structure file from the shared volume that no step of this conversation produced. */
function structureAt(path: string): StructurePanel {
  const name = path.split("/").pop() ?? path;
  return { kind: "structure", tool: "assets", name, path, url: `/api/artifacts/${path}`, format: /\.pdb$/i.test(name) ? "pdb" : "cif" };
}

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
  /** Bumped on every request to show a tab, so one the user closed opens again. */
  const [openSeq, setOpenSeq] = useState(0);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [draft, setDraft] = useState("");
  const composerRef = useRef<ComposerHandle>(null);

  const userStructures = useUserStructures(session?.id ?? "new");
  const artifacts = useMemo(() => withUserStructures(collectArtifacts(transcript.turns), userStructures), [transcript.turns, userStructures]);
  const [picked, setPicked] = useState<StructurePanel | null>(null);
  const [selectedSmiles, setSelectedSmiles] = useState<string | null>(null);
  const [hints, setHints] = useState<{ id: number; message: string }[]>([]);
  const activeStructure = picked ?? artifacts.structures[0]?.item ?? null;
  const target = useMemo(() => targetLabel(extractTargetFacts(toolEvents(artifacts.steps))), [artifacts.steps]);

  const hint = useCallback((message: string) => setHints((list) => [...list.slice(-2), { id: Date.now(), message }]), []);
  const dismissHint = useCallback((id: number) => setHints((list) => list.filter((entry) => entry.id !== id)), []);

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
    setOpenSeq((value) => value + 1);
    setWorkbenchOpen(true);
    if (!wide()) setSheetOpen(true);
  }, []);

  useEffect(() => onWorkbenchRequest(showWorkbench), [showWorkbench]);

  // Anything that asks the viewer for something brings it up, on the structure it names.
  useEffect(
    () =>
      onOpenRequest((request) => {
        const target = request.path;
        const found = target ? artifacts.structures.find(({ item }) => sameFile(item.path, target)) : undefined;
        if (found) setPicked(found.item);
        else if (target && /\.(pdb|cif|mmcif)$/i.test(target)) setPicked(structureAt(target));
        else if (!artifacts.structures.length) hint("There is no structure in this conversation yet.");
        showWorkbench("structure");
      }),
    [artifacts.structures, showWorkbench, hint],
  );

  const view = useCallback((action: ViewerAction) => sendToViewer({ action, source: "palette" }), []);

  /** "/view …" runs on the viewer, here; everything else goes to the agent. */
  const { send: sendToAgent } = chat;
  const send = useCallback(
    (text: string) => {
      if (!text.toLowerCase().startsWith(VIEW_PREFIX)) return sendToAgent(text);
      const command = text.slice(VIEW_PREFIX.length).trim();
      const action = parseViewerCommand(command || "help");
      if (!action) {
        hint(`“${command}” is not a viewer command. Try: ${VIEWER_HELP}`);
        return false;
      }
      sendToViewer({ action, source: "composer" });
      return true;
    },
    [sendToAgent, hint],
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
      const key = event.key.toLowerCase();
      // Ctrl+N belongs to the browser, so a new conversation is Alt+N.
      if (event.altKey && !event.ctrlKey && !event.metaKey && key === "n") {
        event.preventDefault();
        open(null);
        return;
      }
      if (!(event.ctrlKey || event.metaKey) || event.altKey || event.shiftKey) return;
      if (key === "k") setPaletteOpen((value) => !value);
      else if (key === "b") setWorkbenchOpen((value) => !value);
      else if (key === "j") toggle();
      else return;
      event.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle, open]);

  const paletteItems = useMemo<PaletteItem[]>(() => {
    const items: PaletteItem[] = [
      { id: "new", group: "Conversation", label: "New conversation", hint: "Alt+N", run: () => open(null) },
      ...(busy ? [{ id: "stop", group: "Conversation", label: "Stop the agent", run: chat.cancel }] : []),
      ...(session
        ? [
            { id: "export-md", group: "Conversation", label: "Export as Markdown", run: () => void exportCurrent("md") },
            { id: "export-json", group: "Conversation", label: "Export as JSON", run: () => void exportCurrent("json") },
          ]
        : []),
      { id: "tab-structure", group: "Workbench", label: "Show the structure", run: () => showWorkbench("structure") },
      { id: "tab-candidates", group: "Workbench", label: "Show candidates", run: () => showWorkbench("candidates") },
      { id: "tab-molecules", group: "Workbench", label: "Show molecules", run: () => showWorkbench("molecules") },
      { id: "tab-properties", group: "Workbench", label: "Show predicted properties", run: () => showWorkbench("properties") },
      { id: "tab-jobs", group: "Workbench", label: "Show REINVENT jobs", hint: "/jobs", run: () => showWorkbench("jobs") },
      { id: "tab-ttp", group: "Workbench", label: "Show the target product profile", run: () => showWorkbench("ttp") },
      { id: "tab-target", group: "Workbench", label: "Show the target", run: () => showWorkbench("target") },
      { id: "tab-assets", group: "Workbench", label: "Show shared assets", hint: "/shared", run: () => showWorkbench("assets") },
      ...artifacts.structures.map(({ item }) => ({
        id: `st-${item.path}`,
        group: "Structures",
        label: structureTitle(item),
        hint: runOf(item.path) ?? item.tool,
        run: () => {
          setPicked(item);
          showWorkbench("structure");
        },
      })),
      ...(artifacts.structures.length
        ? [
            { id: "v-ligand", group: "Viewer", label: "Focus the ligand", hint: "/view ligand", run: () => view({ kind: "ligand" }) },
            { id: "v-pocket", group: "Viewer", label: "Show the binding pocket", hint: "/view pocket", run: () => view({ kind: "pocket", radius: 5 }) },
            { id: "v-hbond", group: "Viewer", label: "Show hydrogen bonds", hint: "/view hbonds", run: () => view({ kind: "contacts", type: "hbond" }) },
            { id: "v-surface", group: "Viewer", label: "Surface view", hint: "/view surface", run: () => view({ kind: "style", style: "surface" }) },
            { id: "v-cartoon", group: "Viewer", label: "Cartoon view", hint: "/view cartoon", run: () => view({ kind: "style", style: "cartoon" }) },
            { id: "v-chain", group: "Viewer", label: "Colour by chain", hint: "/view color by chain", run: () => view({ kind: "color", color: "chain" }) },
            { id: "v-plddt", group: "Viewer", label: "Colour by confidence", hint: "/view color by plddt", run: () => view({ kind: "color", color: "confidence" }) },
            { id: "v-quality", group: "Viewer", label: "Pose quality and affinity", hint: "/view quality", run: () => view({ kind: "panel", tab: "quality" }) },
            { id: "v-shot", group: "Viewer", label: "Save an image", hint: "/view screenshot", run: () => view({ kind: "screenshot" }) },
          ]
        : []),
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
  }, [open, busy, chat.cancel, session, exportCurrent, showWorkbench, artifacts.structures, view, railOpen, workbenchOpen, theme, toggle, agent, ask, sessions.sessions]);

  const workbenchProps = {
    tab,
    openSeq,
    onTabChange: setTab,
    artifacts,
    theme,
    sessionId: session?.id ?? null,
    activeStructure,
    onSelectStructure: setPicked,
    selectedSmiles,
    onSelectSmiles: setSelectedSmiles,
    onAsk: ask,
  };

  const rail = (
    <SessionRail
      sessions={sessions.sessions}
      activeId={session?.id ?? null}
      error={sessions.error}
      agent={agent}
      onSelect={(id) => open(id)}
      onNew={() => open(null)}
      onRename={sessions.rename}
      onDelete={async (id) => {
        await sessions.remove(id);
        if (id === session?.id) open(null);
      }}
      onClose={() => setRailOpen(false)}
    />
  );

  return (
    <div className="bg-page flex h-full flex-col">
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
          <Header
            theme={theme}
            onToggleTheme={toggle}
            connection={connection}
            agent={agent}
            gatewayError={chat.gatewayError}
            title={session?.title ?? ""}
            target={target}
            onShowTarget={() => showWorkbench("target")}
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
          <div className="relative flex min-h-0 flex-1 flex-col">
            <Transcript
              turns={transcript.turns}
              theme={theme}
              openStructurePath={activeStructure?.path ?? null}
              onOpenMolecule={openMolecule}
              onShowProperties={() => showWorkbench("properties")}
              onPickPrompt={ask}
              hasStructure={artifacts.structures.length > 0}
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
            onSend={send}
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

        {(chat.notices.length > 0 || hints.length > 0) && (
          <div className="pointer-events-none absolute top-3 left-1/2 z-50 flex max-w-[min(36rem,90vw)] -translate-x-1/2 flex-col gap-2" aria-live="polite">
            {chat.notices.map((notice) => (
              <Toast key={notice.id} id={notice.id} message={notice.message} onDismiss={chat.dismissNotice} />
            ))}
            {hints.map((entry) => (
              <Toast key={`h${entry.id}`} id={entry.id} message={entry.message} onDismiss={dismissHint} />
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
