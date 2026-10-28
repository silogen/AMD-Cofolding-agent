// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { memo, useEffect, useMemo, useState, type CSSProperties, type ReactNode } from "react";
import type { StructurePanel } from "../../types/api";
import type { Theme } from "../../hooks/useTheme";
import { structureTitle, type Artifacts } from "../../lib/artifacts";
import { clock } from "../../lib/format";
import {
  DOCK_DEFAULT_WIDTH,
  DOCK_MIN_WIDTH,
  DOCK_TABS,
  clampWidth,
  closeTab,
  dockRoom,
  dockTabOf,
  loadTabs,
  loadWidth,
  openTab,
  saveTabs,
  saveWidth,
  type DockTab,
  type DockView,
} from "../../lib/dock";
import { serviceName } from "../../lib/properties";
import { sendToViewer } from "../../lib/viewerBus";
import { onMoleculeRequest, type MoleculeContext } from "../../lib/moleculePage";
import { StructureWorkspace } from "../StructureWorkspace/StructureWorkspace";
import { PropertiesCard } from "../PropertiesCard/PropertiesCard";
import { MolDepiction } from "../MolDepiction/MolDepiction";
import { MoleculePage } from "../MoleculePage/MoleculePage";
import { extractTargetFacts, toolEvents } from "../../lib/ttp";
import { candidatesOf } from "../../lib/candidates";
import { CandidatesPanel } from "../CandidatesPanel/CandidatesPanel";
import { JobsPanel } from "../JobsPanel/JobsPanel";
import { RunResults } from "../RunResults/RunResults";
import { RunAnalysis } from "../RunAnalysis/RunAnalysis";
import { TtpPanel } from "../TtpPanel/TtpPanel";
import { TargetCard } from "../TargetCard/TargetCard";
import { AssetsPanel } from "../AssetsPanel/AssetsPanel";
import { ShortlistPanel } from "../Shortlist/ShortlistPanel";
import { StarButton } from "../Shortlist/StarButton";
import { useShortlist } from "../../lib/shortlist";

/** A tab, or a view inside one: "candidates" and "properties" open in Molecules, "activity" (Analysis) in Jobs. */
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
  /** Bump to re-open `tab` even when it did not change (e.g. after its dock tab was closed). */
  openSeq?: number;
}

const LABEL: Record<DockTab, string> = {
  structure: "Structure",
  molecules: "Molecules",
  tables: "Tables",
  jobs: "Jobs",
  results: "Results",
  target: "Target",
  ttp: "TTP",
  assets: "Assets",
};
/** The tabs the dock offers; "tables" is gone, though saved tab lists and old links may still name it. */
const TABS: readonly DockTab[] = DOCK_TABS.filter((id) => id !== "tables" && id !== "results");
const openable = (ids: readonly DockTab[], id: DockTab) => (TABS.includes(id) ? openTab(ids, id) : [...ids]);
/** Tabs a molecule page opens over; from any other, it opens over Molecules. */
const PAGE_TABS: readonly DockTab[] = ["molecules", "jobs"];

/** A molecule's page, shown over the list of the tab it was opened from. */
interface Page {
  tab: DockTab;
  smiles: string;
  context?: MoleculeContext;
}
type Show = (smiles: string, context?: MoleculeContext) => void;

const MOLECULE_VIEWS = [
  { id: "candidates", label: "Candidates" },
  { id: "molecules", label: "All" },
  { id: "properties", label: "Properties" },
  { id: "shortlist", label: "Shortlist" },
] as const;
const JOB_VIEWS = [
  { id: "jobs", label: "Jobs" },
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

  const candidateCount = candidatesOf(artifacts).length;
  const starred = useShortlist(sessionId ?? "new").length;
  const counts: Record<WorkbenchTab, number> = {
    structure: artifacts.structures.length,
    candidates: candidateCount,
    molecules: artifacts.molecules.length,
    properties: artifacts.properties.length,
    shortlist: starred,
    jobs: 0,
    results: 0,
    ttp: 0,
    target: 0,
    assets: 0,
    tables: 0,
    activity: 0,
  };

  const labelOf = (id: DockTab) => (id === "structure" && props.activeStructure ? structureTitle(props.activeStructure) : LABEL[id]);
  const current = dockTabOf(tab);

  // Open tabs, editor-style. Synced from the `tab` prop during render (no effect).
  const [openIds, setOpenIds] = useState<DockTab[]>(() => openable(loadTabs(sessionId, TABS) ?? [], current));
  /** The view each merged tab showed last, so going back to it lands where the user left. */
  const [lastView, setLastView] = useState<Partial<Record<DockTab, WorkbenchTab>>>(() => ({ [current]: tab }));
  // A molecule picked in the chat arrives as `selectedSmiles` with the Molecules tab.
  const [page, setPage] = useState<Page | null>(() => (selectedSmiles && current === "molecules" ? { tab: current, smiles: selectedSmiles } : null));
  const [seen, setSeen] = useState({ sessionId, tab, openSeq, selectedSmiles });
  if (seen.sessionId !== sessionId) {
    setSeen({ sessionId, tab, openSeq, selectedSmiles });
    setOpenIds(openable(loadTabs(sessionId, TABS) ?? [], current));
    setPage(null);
  } else if (seen.tab !== tab || seen.openSeq !== openSeq || seen.selectedSmiles !== selectedSmiles) {
    setSeen({ sessionId, tab, openSeq, selectedSmiles });
    if (selectedSmiles && selectedSmiles !== seen.selectedSmiles) setPage({ tab: current, smiles: selectedSmiles });
    // "Open candidates" and the like ask for the list.
    else if (seen.openSeq !== openSeq && page?.tab === current) setPage(null);
    if (seen.tab !== tab || seen.openSeq !== openSeq) {
      setOpenIds((ids) => openable(ids, current));
      setLastView((views) => ({ ...views, [current]: tab }));
    }
  }
  useEffect(() => saveTabs(sessionId, openIds), [sessionId, openIds]);
  // Cleared once its page is closed, so picking the same molecule in the chat again opens it again.
  useEffect(() => {
    if (!page && selectedSmiles) onSelectSmiles(null);
  }, [page, selectedSmiles, onSelectSmiles]);

  const viewOf = (id: DockTab): WorkbenchTab => lastView[id] ?? (id === "molecules" && candidateCount > 0 ? "candidates" : id);
  const active = openIds.includes(current) ? current : null;
  const close = (id: DockTab) => {
    const next = closeTab(openIds, id, active);
    setOpenIds(next.tabs);
    if (next.active && next.active !== current) onTabChange(viewOf(next.active));
  };
  const open = (id: DockTab) => {
    setOpenIds((ids) => openTab(ids, id));
    onTabChange(viewOf(id));
  };
  const show: Show = (smiles, context) => {
    const here = active && PAGE_TABS.includes(active) ? active : "molecules";
    setPage({ tab: here, smiles, ...(context ? { context } : {}) });
    if (here !== active) open(here);
  };
  useEffect(() => onMoleculeRequest(({ smiles, context }) => show(smiles, context)));
  const shown = page && page.tab === active ? page : null;

  const [menu, setMenu] = useState(false);
  const [width, setWidth] = useState<number | null>(() => loadWidth());
  const [expanded, setExpanded] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [aside, setAside] = useState<HTMLElement | null>(null);
  const room = useDockRoom(aside, !isSheet);

  const shell = isSheet
    ? "fixed inset-0 z-50 flex flex-col bg-raised"
    : "relative flex shrink-0 flex-col border-l border-subtle bg-raised";
  // Whatever the saved width or the expanded state, the chat keeps CHAT_MIN_WIDTH.
  const style: CSSProperties | undefined = isSheet
    ? undefined
    : {
        width: expanded ? `${room}px` : width ? `${clampWidth(width, window.innerWidth, room)}px` : `min(${DOCK_DEFAULT_WIDTH}, ${room}px)`,
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
            setExpanded(false);
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
          {openIds.map((id) => {
            const selected = id === active;
            return (
              <div
                key={id}
                className={`group flex shrink-0 items-center border-b-2 ${
                  selected ? "bg-raised text-body" : "text-muted hover:text-body border-b-transparent"
                }`}
                style={selected ? { borderBottomColor: "var(--color-accent)" } : undefined}
              >
                <button
                  type="button"
                  role="tab"
                  aria-selected={selected}
                  onClick={() => onTabChange(viewOf(id))}
                  className="flex max-w-56 cursor-pointer items-center py-2 pr-1 pl-3.5 text-xs whitespace-nowrap"
                >
                  <span className="truncate">{labelOf(id)}</span>
                </button>
                <button
                  type="button"
                  onClick={() => close(id)}
                  aria-label={`Close ${labelOf(id)}`}
                  title="Close tab"
                  className={`text-faint hover:text-body hover:bg-inset mr-1.5 grid size-5 cursor-pointer place-items-center rounded text-sm leading-none touch:opacity-100 ${
                    selected ? "" : "opacity-0 group-hover:opacity-100 focus-visible:opacity-100"
                  }`}
                >
                  ×
                </button>
              </div>
            );
          })}
        </div>
        <div className="relative flex shrink-0 items-center gap-0.5">
          <button
            type="button"
            onClick={() => setMenu((value) => !value)}
            aria-haspopup="menu"
            aria-expanded={menu}
            aria-label="Open a panel"
            title="Open a panel"
            className="text-muted hover:text-body hover:bg-inset grid size-7 cursor-pointer place-items-center rounded-md text-base touch:size-11"
          >
            +
          </button>
          {menu && (
            <>
              <button type="button" aria-label="Close menu" tabIndex={-1} className="fixed inset-0 z-30 cursor-default" onClick={() => setMenu(false)} />
              <ul role="menu" className="bg-raised border-subtle shadow-card absolute top-full right-0 z-40 mt-1 w-48 rounded-lg border py-1">
                {TABS.map((id) => (
                  <li key={id} role="none">
                    <button
                      type="button"
                      role="menuitem"
                      onClick={() => {
                        setMenu(false);
                        open(id);
                      }}
                      className={`hover:bg-sunken flex w-full cursor-pointer items-center gap-2 px-3 py-1.5 text-left text-xs ${
                        openIds.includes(id) ? "text-body" : "text-muted"
                      }`}
                    >
                      <span className="flex-1">{LABEL[id]}</span>
                      {counts[id] > 0 && <span className="text-faint text-[11px] tabular-nums">{counts[id]}</span>}
                    </button>
                  </li>
                ))}
              </ul>
            </>
          )}
          {!isSheet && (
            <button
              type="button"
              onClick={() => setExpanded((value) => !value)}
              aria-pressed={expanded}
              aria-label={expanded ? "Restore the workbench width" : "Expand the workbench"}
              title={expanded ? "Restore" : "Expand"}
              className="text-muted hover:text-body hover:bg-inset grid size-7 cursor-pointer place-items-center rounded-md text-sm"
            >
              {expanded ? "⇥" : "⇤"}
            </button>
          )}
          <button
            type="button"
            onClick={onClose}
            aria-label="Close the workbench"
            title="Close the workbench"
            className="text-muted hover:text-body hover:bg-inset grid size-7 cursor-pointer place-items-center rounded-md text-base touch:size-11"
          >
            ×
          </button>
        </div>
      </div>

      <div className="flex min-h-0 flex-1 flex-col">
        {active === null && <DockEmpty />}
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
          {active === "molecules" && (
            <>
              <ViewSwitch label="Molecules" views={MOLECULE_VIEWS} value={tab} counts={counts} onChange={onTabChange} />
              {tab === "candidates" && (
                <div className="min-h-0 flex-1">
                  <CandidatesPanel artifacts={artifacts} theme={props.theme} onOpenMolecule={show} onAsk={props.onAsk} />
                </div>
              )}
              {tab === "molecules" && <MoleculesTab {...props} onOpen={show} />}
              {tab === "properties" && <PropertiesTab {...props} onOpen={show} />}
              {tab === "shortlist" && <ShortlistPanel artifacts={artifacts} theme={props.theme} sessionId={sessionId} onOpenMolecule={show} onAsk={props.onAsk} />}
            </>
          )}
          {active === "jobs" && (
            <>
              <ViewSwitch label="Jobs" views={JOB_VIEWS} value={tab} counts={counts} onChange={onTabChange} />
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
  const facts = useMemo(() => extractTargetFacts(events), [events]);
  const molecules = useMemo(() => candidatesOf(artifacts).map((c) => ({ smiles: c.smiles, label: c.name })), [artifacts]);
  if (tab === "target") {
    return (
      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        <TargetCard facts={facts} onAsk={onAsk} />
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

function PropertiesTab({ artifacts, theme, onOpen }: Props & { onOpen: Show }) {
  if (artifacts.properties.length === 0) {
    return <Empty>No predictions yet. ADMET and RetroDMTA predictions appear here, molecules by endpoints.</Empty>;
  }
  return (
    <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-3">
      {artifacts.properties.map(({ item, tool, at }, index) => (
        <section key={index}>
          <h3 className="text-muted mb-1 flex gap-2 text-[11px] font-medium">
            <span>{serviceName(item)}</span>
            <span className="text-faint font-mono">{tool}</span>
            <span className="text-faint ml-auto">{clock(at)}</span>
          </h3>
          <PropertiesCard panel={item} theme={theme} onOpenMolecule={onOpen} />
        </section>
      ))}
    </div>
  );
}

function MoleculesTab({ artifacts, theme, sessionId, onOpen }: Props & { onOpen: Show }) {
  const [query, setQuery] = useState("");
  const shown = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return artifacts.molecules;
    return artifacts.molecules.filter(
      ({ item, tool }) => item.smiles.toLowerCase().includes(needle) || (item.label ?? "").toLowerCase().includes(needle) || tool.includes(needle),
    );
  }, [artifacts.molecules, query]);

  if (artifacts.molecules.length === 0) {
    return <Empty>No molecules yet. Candidates, seeds and compounds the agent looks up are collected here.</Empty>;
  }

  return (
    <>
      <div className="border-subtle shrink-0 border-b px-3 py-2">
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Filter by SMILES, name or tool"
          aria-label="Filter molecules"
          className="bg-sunken border-subtle w-full rounded-lg border px-2.5 py-1.5 text-xs outline-none focus:border-strong"
        />
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        <ul className="grid grid-cols-2 gap-2 xl:grid-cols-3">
          {shown.map(({ item, tool }) => (
            <li key={item.smiles} className="relative">
              <StarButton smiles={item.smiles} label={item.label} source={tool} session={sessionId ?? "new"} className="absolute top-1 right-1 z-10" />
              <button
                type="button"
                onClick={() => onOpen(item.smiles)}
                title={item.smiles}
                className="bg-sunken border-subtle hover:border-strong w-full cursor-pointer overflow-hidden rounded-lg border text-left"
              >
                <MolDepiction smiles={item.smiles} theme={theme} width={200} height={140} className="h-28 w-full p-1" />
                <span className="border-subtle block truncate border-t px-2 py-1 text-[10px]">
                  <span className="font-semibold">{item.label ?? "Molecule"}</span>
                  <span className="text-faint"> · {tool}</span>
                </span>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </>
  );
}

/** The quiet switch between the views a merged tab holds. */
function ViewSwitch({
  label,
  views,
  value,
  counts,
  onChange,
}: {
  label: string;
  views: readonly { id: WorkbenchTab; label: string }[];
  value: WorkbenchTab;
  counts: Record<WorkbenchTab, number>;
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
            className={`flex cursor-pointer items-center gap-1.5 rounded-md px-2.5 py-1 text-xs touch:min-h-11 ${
              selected ? "bg-inset text-body" : "text-muted hover:text-body"
            }`}
          >
            {view.label}
            {counts[view.id] > 0 && <span className="text-faint text-[11px] tabular-nums">{counts[view.id]}</span>}
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

function DockEmpty() {
  return (
    <div className="grid flex-1 place-items-center px-10 text-center">
      <p className="text-faint max-w-72 text-xs leading-relaxed">
        Structures, molecules and jobs open here. Click a result in the conversation, or use + to open a panel.
      </p>
    </div>
  );
}

function Empty({ children }: { children: ReactNode }) {
  return <p className="text-faint flex-1 px-8 py-12 text-center text-xs leading-relaxed">{children}</p>;
}
