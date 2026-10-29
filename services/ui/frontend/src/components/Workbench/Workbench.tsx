// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { memo, useEffect, useMemo, useState, type CSSProperties, type ReactNode } from "react";
import type { StructurePanel } from "../../types/api";
import type { Theme } from "../../hooks/useTheme";
import type { Artifacts } from "../../lib/artifacts";
import { DOCK_DEFAULT_WIDTH, DOCK_MIN_WIDTH, DOCK_TABS, clampWidth, dockRoom, dockTabOf, loadWidth, saveWidth, type DockTab, type DockView } from "../../lib/dock";
import { sendToViewer } from "../../lib/viewerBus";
import { onMoleculeRequest, type MoleculeContext } from "../../lib/moleculePage";
import { StructureWorkspace } from "../StructureWorkspace/StructureWorkspace";
import { MoleculePage } from "../MoleculePage/MoleculePage";
import { MoleculesPanel } from "../MoleculesPanel/MoleculesPanel";
import { extractTargetFacts, toolEvents } from "../../lib/ttp";
import { candidatesOf } from "../../lib/candidates";
import { JobsPanel } from "../JobsPanel/JobsPanel";
import { RunResults } from "../RunResults/RunResults";
import { RunAnalysis } from "../RunAnalysis/RunAnalysis";
import { TtpPanel } from "../TtpPanel/TtpPanel";
import { TargetPanel } from "../TargetCard/TargetPanel";
import { AssetsPanel } from "../AssetsPanel/AssetsPanel";
import { JobPicker } from "../JobPicker/JobPicker";

/** A tab, or a view inside one: "candidates" and "properties" open Molecules, "results" and "activity" (Analysis) views of Jobs. */
export type WorkbenchTab = DockView;
export type WorkbenchLayout = "sidebar" | "sheet";

interface Props {
  layout: WorkbenchLayout;
  tab: WorkbenchTab;
  onTabChange: (tab: WorkbenchTab) => void;
  artifacts: Artifacts;
  theme: Theme;
  sessionId: string | null;
  activeStructure: StructurePanel | null;
  onSelectStructure: (panel: StructurePanel) => void;
  selectedSmiles: string | null;
  onSelectSmiles: (smiles: string | null) => void;
  /** Put a prompt in the composer. */
  onAsk: (text: string) => void;
  onClose: () => void;
  /** Bump to show `tab` afresh even when it did not change (e.g. back to a list from a molecule page). */
  openSeq?: number;
}

const LABEL: Record<DockTab, string> = {
  structure: "Structure",
  molecules: "Molecules",
  jobs: "Jobs",
  target: "Target",
  ttp: "TTP",
  assets: "Assets",
};
/** Tabs a molecule page opens over; from any other, it opens over Molecules. */
const PAGE_TABS: readonly DockTab[] = ["molecules", "jobs"];

/** A molecule's page, shown over the list of the tab it was opened from. */
interface Page {
  tab: DockTab;
  smiles: string;
  context?: MoleculeContext;
}
type Show = (smiles: string, context?: MoleculeContext) => void;

const JOB_VIEWS = [
  { id: "jobs", label: "Summary" },
  { id: "results", label: "Results" },
  // The view id predates the Analysis view; dock.ts and the workbench bus still call it "activity".
  { id: "activity", label: "Analysis" },
] as const;

/**
 * Everything the conversation produced, in one place: the complexes, the
 * molecules, the runs and the log of tool calls. The chat says
 * what happened; the workbench is where the results are worked with.
 */
export const Workbench = memo(function Workbench(props: Props) {
  const { layout, tab, onTabChange, artifacts, onClose, sessionId, openSeq, selectedSmiles, onSelectSmiles } = props;
  const isSheet = layout === "sheet";

  useEffect(() => {
    if (!isSheet) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = previous;
    };
  }, [isSheet, onClose]);

  const active = dockTabOf(tab);

  /** The view each merged tab showed last, so going back to it lands where the user left. Synced from the `tab` prop during render. */
  const [lastView, setLastView] = useState<Partial<Record<DockTab, WorkbenchTab>>>(() => ({ [active]: tab }));
  // A molecule picked in the chat arrives as `selectedSmiles` with the Molecules tab.
  const [page, setPage] = useState<Page | null>(() => (selectedSmiles && active === "molecules" ? { tab: active, smiles: selectedSmiles } : null));
  const [seen, setSeen] = useState({ sessionId, tab, openSeq, selectedSmiles });
  if (seen.sessionId !== sessionId) {
    setSeen({ sessionId, tab, openSeq, selectedSmiles });
    setPage(null);
  } else if (seen.tab !== tab || seen.openSeq !== openSeq || seen.selectedSmiles !== selectedSmiles) {
    setSeen({ sessionId, tab, openSeq, selectedSmiles });
    if (selectedSmiles && selectedSmiles !== seen.selectedSmiles) setPage({ tab: active, smiles: selectedSmiles });
    // "Open candidates" and the like ask for the list.
    else if (seen.openSeq !== openSeq && page?.tab === active) setPage(null);
    if (seen.tab !== tab) setLastView((views) => ({ ...views, [active]: tab }));
  }
  // Cleared once its page is closed, so picking the same molecule in the chat again opens it again.
  useEffect(() => {
    if (!page && selectedSmiles) onSelectSmiles(null);
  }, [page, selectedSmiles, onSelectSmiles]);

  const viewOf = (id: DockTab): WorkbenchTab => lastView[id] ?? id;
  const show: Show = (smiles, context) => {
    const here = PAGE_TABS.includes(active) ? active : "molecules";
    setPage({ tab: here, smiles, ...(context ? { context } : {}) });
    if (here !== active) onTabChange(viewOf(here));
  };
  useEffect(() => onMoleculeRequest(({ smiles, context }) => show(smiles, context)));
  const shown = page && page.tab === active ? page : null;

  const [width, setWidth] = useState<number | null>(() => loadWidth());
  const [dragging, setDragging] = useState(false);
  const [aside, setAside] = useState<HTMLElement | null>(null);
  const room = useDockRoom(aside, !isSheet);

  const shell = isSheet
    ? "fixed inset-0 z-50 flex flex-col bg-raised"
    : "relative flex shrink-0 flex-col border-l border-subtle bg-raised";
  // Whatever the saved width, the chat keeps CHAT_MIN_WIDTH.
  const style: CSSProperties | undefined = isSheet
    ? undefined
    : {
        width: width ? `${clampWidth(width, window.innerWidth, room)}px` : `min(${DOCK_DEFAULT_WIDTH}, ${room}px)`,
        minWidth: Math.min(DOCK_MIN_WIDTH, room),
        maxWidth: `${room}px`,
      };

  return (
    <aside ref={setAside} className={shell} style={style} role={isSheet ? "dialog" : undefined} aria-modal={isSheet || undefined} aria-label="Workbench">
      {!isSheet && (
        <div
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize the workbench"
          title="Drag to resize"
          onPointerDown={(event) => {
            event.currentTarget.setPointerCapture(event.pointerId);
            setDragging(true);
          }}
          onPointerMove={(event) => {
            if (!dragging) return;
            setWidth(clampWidth(window.innerWidth - event.clientX, window.innerWidth, room));
          }}
          onPointerUp={(event) => {
            if (!dragging) return;
            setDragging(false);
            saveWidth(clampWidth(window.innerWidth - event.clientX, window.innerWidth, room));
          }}
          className={`absolute inset-y-0 -left-[3px] z-20 w-1.5 cursor-col-resize transition-colors hover:bg-[var(--color-accent)] ${
            dragging ? "bg-[var(--color-accent)]" : ""
          }`}
        />
      )}
      <div className="border-subtle bg-sunken flex shrink-0 items-stretch gap-1 border-b pr-1">
        <div role="tablist" aria-label="Workbench" className="flex min-w-0 flex-1 overflow-x-auto">
          {DOCK_TABS.map((id) => {
            const selected = id === active;
            return (
              <button
                key={id}
                type="button"
                role="tab"
                aria-selected={selected}
                onClick={() => onTabChange(viewOf(id))}
                className={`shrink-0 cursor-pointer border-b-2 px-3.5 py-2 text-xs whitespace-nowrap ${
                  selected ? "bg-raised text-body" : "text-muted hover:text-body border-b-transparent"
                }`}
                style={selected ? { borderBottomColor: "var(--color-accent)" } : undefined}
              >
                {LABEL[id]}
              </button>
            );
          })}
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Hide the workbench"
          title="Hide the workbench"
          className="text-muted hover:text-body grid size-8 shrink-0 cursor-pointer place-items-center self-center rounded-lg text-sm touch:size-11"
        >
          ›
        </button>
      </div>

      <div className="flex min-h-0 flex-1 flex-col">
        {shown && (
          <MoleculePage
            key={shown.smiles}
            smiles={shown.smiles}
            context={shown.context}
            artifacts={artifacts}
            theme={props.theme}
            sessionId={sessionId}
            onBack={() => setPage(null)}
            onAsk={props.onAsk}
          />
        )}
        {/* Hidden rather than unmounted under a page, so going back finds the list as it was. */}
        <div className={shown ? "hidden" : "contents"}>
          {active === "structure" && <StructureTab {...props} />}
          {active === "molecules" && <MoleculesPanel artifacts={artifacts} theme={props.theme} sessionId={sessionId} onOpenMolecule={show} onAsk={props.onAsk} />}
          {active === "jobs" && (
            <>
              <JobPicker sessionId={sessionId} />
              <ViewSwitch label="Jobs" views={JOB_VIEWS} value={tab} onChange={onTabChange} />
              {tab === "jobs" && (
                <div className="min-h-0 flex-1">
                  <JobsPanel sessionId={sessionId ?? undefined} theme={props.theme} onAsk={props.onAsk} onOpenMolecule={show} />
                </div>
              )}
              {tab === "results" && (
                <div className="min-h-0 flex-1">
                  <RunResults artifacts={artifacts} theme={props.theme} sessionId={sessionId} onOpenMolecule={show} onAsk={props.onAsk} />
                </div>
              )}
              {tab === "activity" && (
                <div className="min-h-0 flex-1">
                  <RunAnalysis sessionId={sessionId} theme={props.theme} onOpenMolecule={show} />
                </div>
              )}
            </>
          )}
          {(active === "ttp" || active === "target" || active === "assets") && <ContextTab {...props} />}
        </div>
      </div>
    </aside>
  );
});

/** The tabs about the campaign rather than its results: the profile, the target and the files. */
function ContextTab({ tab, artifacts, sessionId, onAsk }: Props) {
  const events = useMemo(() => toolEvents(artifacts.steps), [artifacts.steps]);
  const molecules = useMemo(() => candidatesOf(artifacts).map((c) => ({ smiles: c.smiles, label: c.name })), [artifacts]);
  if (tab === "target") {
    return (
      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        <TargetPanel events={events} onAsk={onAsk} />
      </div>
    );
  }
  if (tab === "ttp") return <TtpPanel sessionId={sessionId ?? "new"} events={events} molecules={molecules} onAsk={onAsk} />;
  return <AssetsPanel events={events} onOpenStructure={(path) => sendToViewer({ path, source: "assets" })} onInsert={onAsk} />;
}

function StructureTab({ artifacts, activeStructure, onSelectStructure, theme, onAsk, sessionId }: Props) {
  const list = artifacts.structures;
  const sequence = useMemo(() => extractTargetFacts(toolEvents(artifacts.steps)).uniprotSequence, [artifacts.steps]);
  if (!activeStructure) {
    return <Empty>No structure yet. When the agent cofolds a complex with Boltz-2, it appears here.</Empty>;
  }
  return (
    <StructureWorkspace
      structures={list}
      active={activeStructure}
      onSelect={onSelectStructure}
      interactions={artifacts.interactions}
      properties={artifacts.properties}
      targetSequence={sequence}
      sessionId={sessionId}
      theme={theme}
      onAsk={onAsk}
    />
  );
}

/** The quiet switch between the views a merged tab holds. */
function ViewSwitch({
  label,
  views,
  value,
  onChange,
}: {
  label: string;
  views: readonly { id: WorkbenchTab; label: string }[];
  value: WorkbenchTab;
  onChange: (view: WorkbenchTab) => void;
}) {
  return (
    <div role="group" aria-label={label} className="border-subtle flex shrink-0 items-center gap-0.5 border-b px-2 py-1.5">
      {views.map((view) => {
        const selected = view.id === value;
        return (
          <button
            key={view.id}
            type="button"
            aria-pressed={selected}
            onClick={() => onChange(view.id)}
            className={`cursor-pointer rounded-md px-2.5 py-1 text-xs touch:min-h-11 ${selected ? "bg-inset text-body" : "text-muted hover:text-body"}`}
          >
            {view.label}
          </button>
        );
      })}
    </div>
  );
}

/** The chat column the dock sits beside: the <main> just before it or one of its wrappers. */
function chatOf(aside: HTMLElement): HTMLElement | null {
  for (let node: HTMLElement | null = aside; node; node = node.parentElement) {
    const before = node.previousElementSibling;
    if (before instanceof HTMLElement && before.tagName === "MAIN") return before;
  }
  return document.querySelector("main");
}

/** The widest the dock can be beside the chat (dockRoom), followed as the window and the session rail change. */
function useDockRoom(aside: HTMLElement | null, on: boolean): number {
  const [room, setRoom] = useState(() => dockRoom(window.innerWidth));
  useEffect(() => {
    if (!aside || !on) return;
    const update = () => setRoom(dockRoom(window.innerWidth, chatOf(aside)?.getBoundingClientRect().left ?? 0));
    update();
    const chat = chatOf(aside);
    const observer = new ResizeObserver(update);
    if (chat) observer.observe(chat);
    window.addEventListener("resize", update);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", update);
    };
  }, [aside, on]);
  return room;
}

function Empty({ children }: { children: ReactNode }) {
  return <p className="text-faint flex-1 px-8 py-12 text-center text-xs leading-relaxed">{children}</p>;
}
