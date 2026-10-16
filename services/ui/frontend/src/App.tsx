// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useCallback, useState } from "react";
import { useChat } from "./hooks/useChat";
import { useTheme } from "./hooks/useTheme";
import { Header } from "./components/Header/Header";
import { Transcript } from "./components/Transcript/Transcript";
import { Composer } from "./components/Composer/Composer";
import { WorkspacePanel } from "./components/WorkspacePanel/WorkspacePanel";
import type { ViewStyle } from "./components/StructureViewer/StructureViewer";
import type { StructurePanel } from "./types/api";

/**
 * Chat and structure viewer.
 *
 * The layout is mobile first. One column is the base case: the conversation
 * fills the screen and the viewer opens over it on demand. The side-by-side
 * arrangement is an enhancement that appears at ``lg``, where there is finally
 * room for a viewer wide enough to rotate a protein in.
 *
 * Both halves are driven by the same conversation state: a structure the agent
 * produces appears inline in the transcript and, when opened, fills the viewer.
 * Nothing in the viewer exists without a message that produced it.
 */
export default function App() {
  const { theme, toggle } = useTheme();
  const { turns, connection, busy, model, tools, structures, send, reconnect } = useChat();

  // Only the structure the user picked is state. What the viewer shows is
  // derived: until a pick is made, the newest structure wins, so the first
  // complex appears by itself and later ones do not steal the view from a
  // structure someone is reading.
  const [picked, setPicked] = useState<StructurePanel | null>(null);
  const [style, setStyle] = useState<ViewStyle>("cartoon");

  // Two independent pieces of state, because the two layouts have opposite
  // defaults: the sidebar is open until it is dismissed, while the sheet
  // covers the conversation and so must only ever open on request.
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [sheetOpen, setSheetOpen] = useState(false);

  const newest = structures.find((panel): panel is StructurePanel => panel.kind === "structure") ?? null;
  const active = picked ?? newest;

  const openStructure = useCallback((panel: StructurePanel) => {
    setPicked(panel);
    setSidebarOpen(true);
    setSheetOpen(true);
  }, []);

  const disabled = connection !== "open" || busy;
  const disabledReason =
    connection !== "open" ? "Waiting for the agent…" : "The agent is working on your last request…";

  return (
    <div className="bg-page flex h-full flex-col">
      <Header
        theme={theme}
        onToggleTheme={toggle}
        connection={connection}
        model={model}
        toolCount={tools.length}
        onReconnect={reconnect}
        workspaceOpen={sidebarOpen}
        onToggleWorkspace={() => setSidebarOpen((open) => !open)}
      />

      <div className="flex min-h-0 flex-1">
        <main className="flex min-w-0 flex-1 flex-col">
          {/* The floating button is positioned against the transcript, not
              against the whole column, so it never covers the composer. */}
          <div className="relative flex min-h-0 flex-1 flex-col">
            <Transcript
              turns={turns}
              busy={busy}
              theme={theme}
              onOpenStructure={openStructure}
              openStructurePath={active?.path ?? null}
            />

            {/*
              Below lg the viewer is not on screen, so without this the only way
              to reach a structure is to scroll back to the message that made it.
              The button appears only once there is something to show.
            */}
            {active && !sheetOpen && (
              <button
                type="button"
                onClick={() => setSheetOpen(true)}
                className="absolute right-4 bottom-4 z-30 flex min-h-11 items-center gap-2 rounded-full px-4 text-xs font-semibold text-white shadow-lg lg:hidden"
                style={{ background: "var(--color-accent)" }}
              >
                <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                  <path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3Z" strokeLinejoin="round" />
                  <path d="M12 12l8-4.5M12 12v9M12 12L4 7.5" strokeLinejoin="round" />
                </svg>
                View structure
              </button>
            )}
          </div>

          <Composer onSend={send} disabled={disabled} disabledReason={disabledReason} />
        </main>

        {/* Wide screens: the viewer sits beside the conversation. */}
        {sidebarOpen && (
          <div className="hidden lg:flex">
            <WorkspacePanel
              layout="sidebar"
              structures={structures}
              active={active}
              onSelect={setPicked}
              style={style}
              onStyleChange={setStyle}
              theme={theme}
              onClose={() => setSidebarOpen(false)}
            />
          </div>
        )}
      </div>

      {/* Narrow screens: the viewer covers the conversation while it is open. */}
      {sheetOpen && (
        <div className="lg:hidden">
          <WorkspacePanel
            layout="sheet"
            structures={structures}
            active={active}
            onSelect={setPicked}
            style={style}
            onStyleChange={setStyle}
            theme={theme}
            onClose={() => setSheetOpen(false)}
          />
        </div>
      )}
    </div>
  );
}
