// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { memo, useEffect, useMemo, useState, type CSSProperties, type ReactNode } from "react";
import type { StructurePanel } from "../../types/api";
import type { Theme } from "../../hooks/useTheme";
import { structureTitle, type Artifacts } from "../../lib/artifacts";
import { clock, duration } from "../../lib/format";
import { descriptors } from "../../lib/rdkit";
import {
  DOCK_DEFAULT_WIDTH,
  DOCK_MAX_FRACTION,
  DOCK_MIN_WIDTH,
  DOCK_TABS,
  clampWidth,
  closeTab,
  dockTabOf,
  loadTabs,
  loadWidth,
  openTab,
  saveTabs,
  saveWidth,
  type DockTab,
  type DockView,
} from "../../lib/dock";
import { endpointLabel, formatValue, serviceName } from "../../lib/properties";
import { sendToViewer } from "../../lib/viewerBus";
import { StructureWorkspace } from "../StructureWorkspace/StructureWorkspace";
import { PropertiesCard } from "../PropertiesCard/PropertiesCard";
import { DataTable } from "../DataTable/DataTable";
import { CopyButton } from "../CopyButton/CopyButton";
import { MolDepiction } from "../MolDepiction/MolDepiction";
import { useRDKit } from "../../hooks/useRDKit";
import { summarise } from "../../lib/format";
import { extractTargetFacts, toolEvents } from "../../lib/ttp";
import { candidatesOf } from "../../lib/candidates";
import { CandidatesPanel } from "../CandidatesPanel/CandidatesPanel";
import { JobsPanel } from "../JobsPanel/JobsPanel";
import { RunResults } from "../RunResults/RunResults";
import { TtpPanel } from "../TtpPanel/TtpPanel";
import { TargetCard } from "../TargetCard/TargetCard";
import { AssetsPanel } from "../AssetsPanel/AssetsPanel";
import { ShortlistPanel } from "../Shortlist/ShortlistPanel";
import { StarButton } from "../Shortlist/StarButton";
import { useShortlist } from "../../lib/shortlist";

/** A tab, or a view inside one: "candidates" and "properties" open in Molecules, "activity" in Jobs. */
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

const MOLECULE_VIEWS = [
  { id: "candidates", label: "Candidates" },
  { id: "molecules", label: "All" },
  { id: "properties", label: "Properties" },
  { id: "shortlist", label: "Shortlist" },
] as const;
const JOB_VIEWS = [
  { id: "jobs", label: "Jobs" },
  { id: "activity", label: "Activity" },
] as const;

/**
 * Everything the conversation produced, in one place: the complexes, the
 * molecules, the result tables and the log of tool calls. The chat says
 * what happened; the workbench is where the results are worked with.
 */
export const Workbench = memo(function Workbench(props: Props) {
  const { layout, tab, onTabChange, artifacts, onClose, sessionId, openSeq } = props;
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
    tables: artifacts.tables.length,
    activity: artifacts.steps.length,
  };

  const labelOf = (id: DockTab) => (id === "structure" && props.activeStructure ? structureTitle(props.activeStructure) : LABEL[id]);
  const current = dockTabOf(tab);

  // Open tabs, editor-style. Synced from the `tab` prop during render (no effect).
  const [openIds, setOpenIds] = useState<DockTab[]>(() => openTab(loadTabs(sessionId, DOCK_TABS) ?? [], current));
  /** The view each merged tab showed last, so going back to it lands where the user left. */
  const [lastView, setLastView] = useState<Partial<Record<DockTab, WorkbenchTab>>>(() => ({ [current]: tab }));
  const [seen, setSeen] = useState({ sessionId, tab, openSeq });
  if (seen.sessionId !== sessionId) {
    setSeen({ sessionId, tab, openSeq });
    setOpenIds(openTab(loadTabs(sessionId, DOCK_TABS) ?? [], current));
  } else if (seen.tab !== tab || seen.openSeq !== openSeq) {
    setSeen({ sessionId, tab, openSeq });
    setOpenIds((ids) => openTab(ids, current));
    setLastView((views) => ({ ...views, [current]: tab }));
  }
  useEffect(() => saveTabs(sessionId, openIds), [sessionId, openIds]);

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

  const [menu, setMenu] = useState(false);
  const [width, setWidth] = useState<number | null>(() => loadWidth());
  const [expanded, setExpanded] = useState(false);
  const [dragging, setDragging] = useState(false);

  const shell = isSheet
    ? "fixed inset-0 z-50 flex flex-col bg-raised"
    : "relative flex shrink-0 flex-col border-l border-subtle bg-raised";
  const style: CSSProperties | undefined = isSheet
    ? undefined
    : {
        width: expanded ? `${DOCK_MAX_FRACTION * 100}vw` : width ? `${width}px` : DOCK_DEFAULT_WIDTH,
        minWidth: DOCK_MIN_WIDTH,
        maxWidth: `${DOCK_MAX_FRACTION * 100}vw`,
      };

  return (
    <aside className={shell} style={style} role={isSheet ? "dialog" : undefined} aria-modal={isSheet || undefined} aria-label="Workbench">
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
            setWidth(clampWidth(window.innerWidth - event.clientX, window.innerWidth));
          }}
          onPointerUp={(event) => {
            if (!dragging) return;
            setDragging(false);
            saveWidth(clampWidth(window.innerWidth - event.clientX, window.innerWidth));
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
                {DOCK_TABS.map((id) => (
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
        {active === "structure" && <StructureTab {...props} />}
        {active === "molecules" && (
          <>
            <ViewSwitch label="Molecules" views={MOLECULE_VIEWS} value={tab} counts={counts} onChange={onTabChange} />
            {tab === "candidates" && <CandidatesTab {...props} />}
            {tab === "molecules" && <MoleculesTab {...props} />}
            {tab === "properties" && <PropertiesTab {...props} />}
            {tab === "shortlist" && (
              <ShortlistPanel
                artifacts={props.artifacts}
                theme={props.theme}
                sessionId={props.sessionId}
                onOpenMolecule={(smiles) => openMolecule(props, smiles)}
                onAsk={props.onAsk}
              />
            )}
          </>
        )}
        {active === "jobs" && (
          <>
            <ViewSwitch label="Jobs" views={JOB_VIEWS} value={tab} counts={counts} onChange={onTabChange} />
            {tab === "jobs" && (
              <div className="min-h-0 flex-1">
                <JobsPanel sessionId={props.sessionId ?? undefined} theme={props.theme} onAsk={props.onAsk} onOpenMolecule={(smiles) => openMolecule(props, smiles)} />
              </div>
            )}
            {tab === "activity" && <ActivityTab {...props} />}
          </>
        )}
        {active === "results" && (
          <div className="min-h-0 flex-1">
            <RunResults artifacts={props.artifacts} theme={props.theme} sessionId={props.sessionId} onOpenMolecule={(smiles) => openMolecule(props, smiles)} onAsk={props.onAsk} />
          </div>
        )}
        {(active === "ttp" || active === "target" || active === "assets") && <ContextTab {...props} />}
        {active === "tables" && <TablesTab {...props} />}
      </div>
    </aside>
  );
});

function openMolecule({ onSelectSmiles, onTabChange }: Props, smiles: string) {
  onSelectSmiles(smiles);
  onTabChange("molecules");
}

function CandidatesTab(props: Props) {
  const { artifacts, theme, onSelectStructure, onTabChange, onAsk } = props;
  return (
    <div className="min-h-0 flex-1">
      <CandidatesPanel
        artifacts={artifacts}
        theme={theme}
        onOpenMolecule={(smiles) => openMolecule(props, smiles)}
        onOpenStructure={(path) => {
          const hit = artifacts.structures.find(({ item }) => item.path === path);
          if (!hit) return;
          onSelectStructure(hit.item);
          onTabChange("structure");
        }}
        onAsk={onAsk}
      />
    </div>
  );
}

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
      targetSequence={sequence}
      sessionId={sessionId}
      theme={theme}
      onAsk={onAsk}
    />
  );
}

function PropertiesTab({ artifacts, theme, onSelectSmiles, onTabChange }: Props) {
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
          <PropertiesCard
            panel={item}
            theme={theme}
            onOpenMolecule={(smiles) => {
              onSelectSmiles(smiles);
              onTabChange("molecules");
            }}
          />
        </section>
      ))}
    </div>
  );
}

function MoleculesTab({ artifacts, theme, selectedSmiles, onSelectSmiles, onAsk, sessionId }: Props) {
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
      {selectedSmiles && (
        <MoleculeDetail smiles={selectedSmiles} artifacts={artifacts} theme={theme} onClose={() => onSelectSmiles(null)} onAsk={onAsk} />
      )}
      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        <ul className="grid grid-cols-2 gap-2 xl:grid-cols-3">
          {shown.map(({ item, tool }) => (
            <li key={item.smiles} className="relative">
              <StarButton smiles={item.smiles} label={item.label} source={tool} session={sessionId ?? "new"} className="absolute top-1 right-1 z-10" />
              <button
                type="button"
                onClick={() => onSelectSmiles(item.smiles)}
                aria-pressed={item.smiles === selectedSmiles}
                title={item.smiles}
                className={`bg-sunken border-subtle w-full cursor-pointer overflow-hidden rounded-lg border text-left ${
                  item.smiles === selectedSmiles ? "border-strong ring-2 ring-[var(--color-accent)]" : "hover:border-strong"
                }`}
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

/** Rows of every result table that mention this SMILES. */
function rowsFor(smiles: string, artifacts: Artifacts) {
  const found: { tool: string; row: Record<string, unknown> }[] = [];
  for (const { item, tool } of artifacts.tables) {
    for (const row of item.rows) {
      if (Object.values(row).some((value) => value === smiles)) found.push({ tool, row });
    }
  }
  return found;
}

function MoleculeDetail({
  smiles,
  artifacts,
  theme,
  onClose,
  onAsk,
}: {
  smiles: string;
  artifacts: Artifacts;
  theme: Theme;
  onClose: () => void;
  onAsk: (text: string) => void;
}) {
  const { rdkit } = useRDKit();
  const props = useMemo(() => (rdkit ? descriptors(rdkit, smiles) : null), [rdkit, smiles]);
  const rows = useMemo(() => rowsFor(smiles, artifacts), [smiles, artifacts]);
  const complex = artifacts.structures.find(({ item }) => item.smiles === smiles)?.item;
  const predicted = artifacts.properties.flatMap(({ item }) =>
    item.molecules.filter((m) => m.ok && (m.smiles === smiles || m.standardized === smiles)).map((m) => ({ panel: item, molecule: m })),
  )[0];

  const facts: [string, string][] = props
    ? [
        ["MW", props.mw.toFixed(1)],
        ["cLogP", props.clogp.toFixed(2)],
        ["TPSA", props.tpsa.toFixed(1)],
        ["HBD / HBA", `${props.hbd} / ${props.hba}`],
        ["Rot. bonds", String(props.rotb)],
        ["Heavy atoms", String(props.heavyAtoms)],
      ]
    : [];

  const asks: [string, string][] = [
    ["Cofold", `Cofold ${smiles} into {TARGET} with Boltz-2 and check the pose with PoseBusters.`],
    ["ADMET", `Predict ADMET properties for ${smiles}.`],
    ["Analogues", `Generate analogues of ${smiles} with Mol2Mol and score them against {TARGET}.`],
  ];

  return (
    <section aria-label="Selected molecule" className="border-subtle bg-sunken max-h-[55%] shrink-0 overflow-y-auto border-b p-3">
      <div className="flex gap-3">
        <div className="bg-raised border-subtle w-44 shrink-0 rounded-lg border p-1">
          <MolDepiction smiles={smiles} theme={theme} width={240} height={180} className="h-32 w-full" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-start gap-2">
            <code className="min-w-0 flex-1 font-mono text-[11px] break-all">{smiles}</code>
            <CopyButton value={smiles} label="Copy SMILES" />
            <button type="button" onClick={onClose} aria-label="Close" className="text-muted hover:text-body cursor-pointer text-base leading-none">
              ×
            </button>
          </div>
          {facts.length > 0 && (
            <dl className="mt-2 grid grid-cols-3 gap-x-3 gap-y-1 text-[11px]">
              {facts.map(([label, value]) => (
                <div key={label}>
                  <dt className="text-faint">{label}</dt>
                  <dd className="font-mono tabular-nums">{value}</dd>
                </div>
              ))}
            </dl>
          )}
          <div className="mt-2 flex flex-wrap gap-1">
            {asks.map(([label, prompt]) => (
              <button
                key={label}
                type="button"
                onClick={() => onAsk(prompt)}
                className="bg-raised border-subtle hover:border-strong cursor-pointer rounded-full border px-2.5 py-0.5 text-[11px]"
              >
                {label}…
              </button>
            ))}
            {complex && (
              <button
                type="button"
                onClick={() => sendToViewer({ path: complex.path, action: { kind: "ligand" }, source: "molecule" })}
                className="cursor-pointer rounded-full px-2.5 py-0.5 text-[11px] font-medium text-white"
                style={{ background: "var(--color-accent)" }}
              >
                Show complex
              </button>
            )}
          </div>
        </div>
      </div>
      {predicted && (
        <dl className="bg-raised border-subtle mt-3 grid grid-cols-3 gap-x-3 gap-y-1 rounded-lg border p-2 text-[11px] sm:grid-cols-4">
          {predicted.panel.endpoints.map((endpoint) => {
            const value = predicted.molecule.values[endpoint];
            return (
              <div key={endpoint} className="min-w-0">
                <dt className="text-faint truncate">{endpointLabel(endpoint)}</dt>
                <dd className="truncate font-mono tabular-nums" title={value?.unit}>
                  {formatValue(value)}
                </dd>
              </div>
            );
          })}
        </dl>
      )}
      {rows.length > 0 && (
        <div className="mt-3 space-y-2">
          {rows.slice(0, 4).map(({ tool, row }, index) => (
            <div key={index} className="bg-raised border-subtle rounded-lg border p-2">
              <p className="text-faint mb-1 text-[10px] font-semibold uppercase">{tool}</p>
              <dl className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[11px] sm:grid-cols-3">
                {Object.entries(row)
                  .filter(([, value]) => value !== smiles && value !== null && value !== "")
                  .map(([key, value]) => (
                    <div key={key} className="min-w-0">
                      <dt className="text-faint truncate">{key}</dt>
                      <dd className="truncate font-mono tabular-nums">
                        {typeof value === "number" && !Number.isInteger(value) ? value.toFixed(3) : String(value)}
                      </dd>
                    </div>
                  ))}
              </dl>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function TablesTab({ artifacts }: Props) {
  if (artifacts.tables.length === 0) {
    return <Empty>No tables yet. Scores, top candidates and predictions appear here.</Empty>;
  }
  return (
    <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-3">
      {artifacts.tables.map(({ item, tool, at }, index) => (
        <section key={index}>
          <h3 className="text-muted mb-1 flex gap-2 text-[11px] font-medium">
            <span className="font-mono">{tool}</span>
            <span className="text-faint ml-auto">{clock(at)}</span>
          </h3>
          <DataTable panel={item} />
        </section>
      ))}
    </div>
  );
}

function ActivityTab({ artifacts }: Props) {
  const [filter, setFilter] = useState<"all" | "failed">("all");
  const steps = filter === "failed" ? artifacts.steps.filter((step) => step.result?.isError) : artifacts.steps;
  const total = artifacts.steps.reduce((sum, step) => sum + (step.result?.elapsedMs ?? 0), 0);
  const failed = artifacts.steps.filter((step) => step.result?.isError).length;

  if (artifacts.steps.length === 0) {
    return <Empty>No tool calls yet. Every call the agent makes is logged here with its duration.</Empty>;
  }

  return (
    <>
      <div className="text-faint flex shrink-0 items-center gap-3 px-3 py-2 text-[11px]">
        <span>
          {artifacts.steps.length} calls · {duration(total)} in tools
        </span>
        {failed > 0 && (
          <div role="group" aria-label="Filter" className="ml-auto flex gap-0.5">
            {(["all", "failed"] as const).map((value) => (
              <button
                key={value}
                type="button"
                aria-pressed={filter === value}
                onClick={() => setFilter(value)}
                className={`cursor-pointer rounded-md px-2 py-0.5 ${filter === value ? "bg-inset text-body" : "hover:text-body"}`}
              >
                {value === "all" ? "All" : `Failed ${failed}`}
              </button>
            ))}
          </div>
        )}
      </div>
      <ol className="border-subtle min-h-0 flex-1 overflow-y-auto border-t">
        {steps.map((step) => {
          const state = !step.result ? "running" : step.result.isError ? "failed" : "ok";
          return (
            <li key={step.id} className="border-subtle flex items-center gap-2 border-b px-3 py-1.5 text-[11px]">
              <span
                aria-label={state}
                className={`size-1.5 shrink-0 rounded-full ${state === "running" ? "animate-pulse" : ""}`}
                style={{ background: state === "ok" ? "var(--border-strong)" : state === "failed" ? "var(--color-accent)" : "var(--color-warn)" }}
              />
              <span className="text-faint shrink-0 font-mono">{clock(step.at)}</span>
              <span className="shrink-0 font-mono font-semibold">{step.name}</span>
              <span className="text-faint min-w-0 flex-1 truncate font-mono" title={summarise(step.arguments)}>
                {summarise(step.arguments)}
              </span>
              <span className="text-faint shrink-0 font-mono">
                {step.result?.elapsedMs != null ? duration(step.result.elapsedMs) : state === "running" ? "…" : ""}
              </span>
            </li>
          );
        })}
      </ol>
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
