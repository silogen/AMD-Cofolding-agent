// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useRef, useState, type FormEvent, type ReactNode } from "react";
import type { Contact, InteractionsPanel, StructurePanel } from "../../types/api";
import type { Theme } from "../../hooks/useTheme";
import type { Sourced } from "../../lib/artifacts";
import { contactsFor, runOf, sameFile, structureTitle } from "../../lib/artifacts";
import {
  BACKGROUNDS,
  COLOR_THEMES,
  StructureView,
  VIEW_STYLES,
  type BackgroundName,
  type ColorName,
  type Picked,
  type ViewStyleName,
  pickedName,
} from "../../lib/molstar";
import { parseViewerCommand, VIEWER_EXAMPLES, VIEWER_HELP, type ViewerAction } from "../../lib/viewerCommands";
import { onViewerRequest } from "../../lib/viewerBus";
import { contactColor, contactLabel, residuesOf, sortedTypes } from "../../lib/contacts";
import { affinityConcentration, affinityText, gate, pIC50, type Gate } from "../../lib/gate";
import { clock } from "../../lib/format";
import { MolDepiction } from "../MolDepiction/MolDepiction";
import { CopyButton } from "../CopyButton/CopyButton";

interface Props {
  structures: Sourced<StructurePanel>[];
  active: StructurePanel;
  onSelect: (panel: StructurePanel) => void;
  interactions: Sourced<InteractionsPanel>[];
  theme: Theme;
  /** Put a prompt in the composer. */
  onAsk: (text: string) => void;
}

type DrawerTab = "contacts" | "quality" | "files";
interface Message {
  id: number;
  text: string;
  tone: "ok" | "bad";
  /** A command the viewer did not understand, offered to the agent instead. */
  ask?: string;
}

const COLOR_LABELS: Record<ColorName, string> = {
  chain: "Chain",
  element: "Element",
  residue: "Residue type",
  hydrophobicity: "Hydrophobicity",
  confidence: "Confidence (pLDDT)",
  secondary: "Secondary structure",
};

/**
 * The 3D structure workspace: one Mol* viewport for every complex in the
 * conversation, with the confidence gate above it, quick views and a command
 * line on it, and the contacts and quality details under it.
 */
export function StructureWorkspace({ structures, active, onSelect, interactions, theme, onAsk }: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const viewRef = useRef<StructureView | null>(null);
  const pendingRef = useRef<ViewerAction | null>(null);

  const [viewReady, setViewReady] = useState(false);
  const [loadedPath, setLoadedPath] = useState<string | null>(null);
  const [error, setError] = useState<{ path: string | null; text: string } | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [message, setMessage] = useState<Message | null>(null);
  const [picked, setPicked] = useState<(Picked & { path: string }) | null>(null);
  const [selected, setSelected] = useState<{ path: string; index: number } | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [drawerTab, setDrawerTab] = useState<DrawerTab>("contacts");
  const [typeFilter, setTypeFilter] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [controls, setControls] = useState(false);
  const [style, setStyle] = useState<ViewStyleName>("cartoon");
  const [color, setColor] = useState<ColorName | null>(null);
  const [background, setBackground] = useState<BackgroundName | null>(null);
  const [spinning, setSpinning] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  const profile = useMemo(() => contactsFor(active, interactions), [active, interactions]);
  const verdict = useMemo(() => gate(active.metrics), [active.metrics]);
  const index = structures.findIndex(({ item }) => item.path === active.path);
  const title = structureTitle(active);
  const run = runOf(active.path);
  const loaded = loadedPath === active.path;
  const failure = error && (error.path === null || error.path === active.path) ? error.text : null;
  const backgroundValue = BACKGROUNDS[background ?? (theme === "dark" ? "black" : "white")];
  const lightCanvas = backgroundValue === BACKGROUNDS.white || backgroundValue === BACKGROUNDS.light;

  const pathRef = useRef(active.path);
  useEffect(() => {
    pathRef.current = active.path;
  }, [active.path]);

  // One viewer for the life of the workspace; structures load into it.
  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    let cancelled = false;
    let created: StructureView | null = null;
    StructureView.create(host, BACKGROUNDS.black)
      .then((view) => {
        if (cancelled) {
          view.dispose();
          return;
        }
        created = view;
        viewRef.current = view;
        view.onPick((pick) => setPicked(pick && pick.resnum !== null ? { ...pick, path: pathRef.current } : null));
        setViewReady(true);
      })
      .catch((reason: unknown) => {
        if (!cancelled) setError({ path: null, text: reason instanceof Error ? reason.message : String(reason) });
      });
    const observer = new ResizeObserver(() => viewRef.current?.resize());
    observer.observe(host);
    return () => {
      cancelled = true;
      observer.disconnect();
      created?.dispose();
      viewRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (viewReady) viewRef.current?.setBackground(backgroundValue);
  }, [viewReady, backgroundValue]);

  // The drawer and the Mol* panel change the viewport size.
  useEffect(() => {
    viewRef.current?.resize();
  }, [detailsOpen, controls]);

  useEffect(() => {
    const view = viewRef.current;
    if (!viewReady || !view) return;
    let stale = false;
    view
      .load(active.url, active.format, title)
      .then(async (ok) => {
        if (!ok || stale) return;
        if (color) await view.setColor(color);
        setLoadedPath(active.path);
        setError(null);
      })
      .catch((reason: unknown) => {
        if (!stale) setError({ path: active.path, text: reason instanceof Error ? reason.message : String(reason) });
      });
    return () => {
      stale = true;
    };
    // color is applied on load and by its own command; reloading on a colour change is not wanted.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewReady, active.path, active.url, active.format, title, attempt]);

  useEffect(() => {
    if (!message) return;
    const timer = window.setTimeout(() => setMessage(null), message.ask ? 9000 : 4500);
    return () => window.clearTimeout(timer);
  }, [message]);

  const say = (text: string, tone: "ok" | "bad" = "ok", ask?: string) => setMessage({ id: Date.now(), text, tone, ask });

  const openDetails = (tab: DrawerTab) => {
    setDrawerTab(tab);
    setDetailsOpen(true);
  };

  const showContacts = (contacts: Contact[], what: string) => {
    const residues = residuesOf(contacts);
    if (!residues.length) {
      say(`No ${what} in this structure.`, "bad");
      return;
    }
    viewRef.current?.focusResidues(residues);
    const names = [...new Set(contacts.map((c) => c.label))];
    say(`${contacts.length} ${what}: ${names.slice(0, 6).join(", ")}${names.length > 6 ? "…" : ""}`);
  };

  /** Run one viewer action and say what happened. */
  function execute(action: ViewerAction) {
    const view = viewRef.current;
    if (!view || !loaded) {
      pendingRef.current = action;
      return;
    }
    try {
      apply(view, action);
    } catch (reason) {
      say(`That did not work: ${reason instanceof Error ? reason.message : String(reason)}`, "bad");
    }
  }

  function apply(view: StructureView, action: ViewerAction) {
    switch (action.kind) {
      case "help":
        say(VIEWER_HELP);
        break;
      case "fit":
        view.fit();
        say("Whole complex in view.");
        break;
      case "ligand":
        view.focusLigand();
        say("Focused the ligand and its contacts.");
        break;
      case "pocket":
        view.focusPocket(action.radius);
        say(`Residues within ${action.radius} Å of the ligand.`);
        break;
      case "residues": {
        view.focusResidues(action.residues, action.mode);
        const names = action.residues
          .map((r) => {
            const resname = profile?.contacts.find((c) => c.resnum === r.resnum && (!r.chain || c.chain === r.chain))?.resname;
            return `${resname ?? ""}${r.resnum}${r.chain ? ` (chain ${r.chain})` : ""}`;
          })
          .join(", ");
        say(`${action.mode === "highlight" ? "Highlighted" : "Focused"} ${names}.`);
        break;
      }
      case "contacts": {
        if (!profile) {
          say("No interaction profile for this structure yet.", "bad", "Profile the interactions of this complex.");
          break;
        }
        const list = action.type ? profile.contacts.filter((c) => c.type === action.type) : profile.contacts;
        setTypeFilter(action.type);
        openDetails("contacts");
        showContacts(list, action.type ? `${contactLabel(action.type).toLowerCase()} contacts` : "contacts");
        break;
      }
      case "style":
        setStyle(action.style);
        void view.setStyle(action.style).then(() => color && view.setColor(color));
        say(`${VIEW_STYLES[action.style].label} view.`);
        break;
      case "color":
        setColor(action.color);
        void view.setColor(action.color);
        say(`Coloured by ${COLOR_LABELS[action.color].toLowerCase()}.`);
        break;
      case "background":
        setBackground(action.background);
        say(`Background ${action.background}.`);
        break;
      case "spin": {
        const on = view.spin(action.on ?? !spinning);
        setSpinning(on);
        say(on ? "Spinning. Type stop to stop." : "Stopped.");
        break;
      }
      case "clear":
        view.clearSelection();
        setSelected(null);
        setPicked(null);
        say("Selection cleared.");
        break;
      case "controls":
        view.showControls(action.on);
        setControls(action.on);
        say(action.on ? "Mol* controls shown." : "Mol* controls hidden.");
        break;
      case "screenshot":
        void view.screenshot(`${title}.png`).then((ok) => say(ok ? "Image saved." : "Screenshots are not available.", ok ? "ok" : "bad"));
        break;
      case "download": {
        const link = document.createElement("a");
        link.href = active.url;
        link.download = active.name;
        link.click();
        say(`Downloading ${active.name}.`);
        break;
      }
      case "panel":
        openDetails(action.tab);
        break;
      case "structure": {
        if (structures.length < 2) {
          say("This is the only structure in the conversation.", "bad");
          break;
        }
        const next = structures[(Math.max(0, index) + action.step + structures.length) % structures.length];
        onSelect(next.item);
        say(`Showing ${next.item.label || next.item.name}.`);
        break;
      }
    }
  }

  const executeRef = useRef(execute);
  useEffect(() => {
    executeRef.current = execute;
  });

  // Actions asked for before the structure was on screen.
  useEffect(() => {
    if (!loaded || !pendingRef.current) return;
    const action = pendingRef.current;
    pendingRef.current = null;
    executeRef.current(action);
  }, [loaded, loadedPath]);

  useEffect(
    () =>
      onViewerRequest((request) => {
        if (!request.action) return;
        // The app switches structure first; run once that one is loaded.
        const target = request.path;
        if (target && !sameFile(target, pathRef.current) && structures.some(({ item }) => sameFile(item.path, target))) {
          pendingRef.current = request.action;
          return;
        }
        executeRef.current(request.action);
      }),
    [structures],
  );

  const pickedHere = picked && picked.path === active.path ? picked : null;
  const pickedContacts = pickedHere && profile ? profile.contacts.filter((c) => c.resnum === pickedHere.resnum) : [];
  const selectedHere = selected && selected.path === active.path && profile ? profile.contacts[selected.index] : null;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <header className="border-subtle flex shrink-0 items-center gap-2 border-b px-3 py-2">
        <div className="flex min-w-0 flex-1 items-center gap-2">
          <h3 className="truncate text-sm font-semibold" title={active.path}>
            {title}
          </h3>
          {run && <span className="text-faint shrink-0 font-mono text-[11px]">{run}</span>}
          {active.metrics && <Badge tone="structure">Predicted</Badge>}
          <GatePill verdict={verdict} />
        </div>
        {structures.length > 1 && (
          <div className="border-subtle flex shrink-0 items-center rounded-md border">
            <HeadButton label="‹" title="Previous structure" onClick={() => execute({ kind: "structure", step: -1 })} />
            <span className="text-faint min-w-9 text-center font-mono text-[11px]">
              {index + 1}/{structures.length}
            </span>
            <HeadButton label="›" title="Next structure" onClick={() => execute({ kind: "structure", step: 1 })} />
          </div>
        )}
        <a href={active.url} download={active.name} className="text-muted hover:text-body hover:bg-inset rounded-md px-2 py-1 text-[11px]">
          Download
        </a>
        <HeadButton
          label="Display"
          title="Show or hide the Mol* controls"
          pressed={controls}
          onClick={() => execute({ kind: "controls", on: !controls })}
        />
      </header>

      <GateStrip verdict={verdict} panel={active} />

      <div className="relative min-h-[280px] flex-1 overflow-hidden" style={{ background: `#${backgroundValue.toString(16).padStart(6, "0")}` }}>
        <div ref={hostRef} className="molstar-host absolute inset-0" />

        {(!loaded || failure) && (
          <div className="absolute inset-0 z-10 grid place-items-center text-center text-xs" style={{ color: lightCanvas ? "#555" : "#aaa" }}>
            {failure ? (
              <div className="max-w-xs space-y-2 px-4">
                <p className="font-semibold" style={{ color: "var(--color-accent)" }}>
                  The structure could not be shown
                </p>
                <p className="break-words">{failure}</p>
                <button
                  type="button"
                  onClick={() => {
                    setError(null);
                    setAttempt((value) => value + 1);
                  }}
                  className="border-subtle bg-raised text-body cursor-pointer rounded-md border px-2.5 py-1 text-[11px]"
                >
                  Try again
                </button>
              </div>
            ) : (
              <span className="flex items-center gap-2">
                <span className="size-3 animate-spin rounded-full border-2 border-current border-t-transparent" />
                {viewReady ? "Loading structure…" : "Starting Mol*…"}
              </span>
            )}
          </div>
        )}

        <div className="absolute top-2 left-2 z-20 flex flex-wrap gap-1">
          <OverlayButton label="Fit" title="The whole complex" onClick={() => execute({ kind: "fit" })} />
          <OverlayButton label="Ligand" title="Focus the ligand and its contacts" onClick={() => execute({ kind: "ligand" })} />
          <OverlayButton label="Pocket" title="Residues within 5 Å of the ligand" onClick={() => execute({ kind: "pocket", radius: 5 })} />
          <OverlayButton label="Details" title="Contacts, quality and files" pressed={detailsOpen} onClick={() => setDetailsOpen((open) => !open)} />
          <div className="relative">
            <OverlayButton label="Style ▾" title="Representation, colours and background" pressed={menuOpen} onClick={() => setMenuOpen((open) => !open)} />
            {menuOpen && (
              <StyleMenu
                style={style}
                color={color}
                background={background}
                spinning={spinning}
                onAction={(action) => {
                  execute(action);
                  setMenuOpen(false);
                }}
              />
            )}
          </div>
        </div>

        {pickedHere && (
          <div className="bg-raised border-subtle shadow-card absolute bottom-12 left-2 z-20 flex max-w-[calc(100%-1rem)] flex-wrap items-center gap-1.5 rounded-lg border px-2 py-1.5 text-[11px]">
            <span className="font-mono font-semibold">{pickedHere.label}</span>
            {pickedHere.chain && <span className="text-faint">chain {pickedHere.chain}</span>}
            {pickedContacts.length > 0 && (
              <span className="flex items-center gap-1">
                {[...new Set(pickedContacts.map((c) => c.type))].map((type) => (
                  <span key={type} className="size-2 rounded-full" title={contactLabel(type)} style={{ background: contactColor(type) }} />
                ))}
                <span className="text-muted">{pickedContacts.length} contact{pickedContacts.length === 1 ? "" : "s"}</span>
              </span>
            )}
            <ChipButton
              onClick={() =>
                pickedHere.resnum !== null && execute({ kind: "residues", residues: [{ chain: pickedHere.chain, resnum: pickedHere.resnum }], mode: "focus" })
              }
            >
              Focus
            </ChipButton>
            <ChipButton
              primary
              onClick={() =>
                onAsk(
                  pickedName(pickedHere) === "the ligand"
                    ? `In ${title}, how is the ligand held in the pocket? Name its key contacts from the interaction profile.`
                    : `In ${title}, what role does ${pickedHere.label}${pickedHere.chain ? ` (chain ${pickedHere.chain})` : ""} play in binding the ligand? Use the interaction profile.`,
                )
              }
            >
              Ask the agent about {pickedName(pickedHere)}
            </ChipButton>
            <button type="button" aria-label="Dismiss" onClick={() => setPicked(null)} className="text-muted hover:text-body cursor-pointer px-1">
              ×
            </button>
          </div>
        )}

        <CommandBar onRun={execute} onUnknown={(text) => say(`“${text}” is not a viewer command.`, "bad", text)} />

        {message && (
          <div
            role="status"
            className="bg-raised border-subtle shadow-card absolute right-2 bottom-12 left-2 z-30 mx-auto flex max-w-xl items-start gap-2 rounded-lg border px-3 py-1.5 text-[11px]"
          >
            <span aria-hidden="true" className="mt-1 size-1.5 shrink-0 rounded-full" style={{ background: message.tone === "ok" ? "var(--color-ok)" : "var(--color-accent)" }} />
            <span className="min-w-0 flex-1">{message.text}</span>
            {message.ask && (
              <ChipButton
                primary
                onClick={() => {
                  onAsk(`About the structure ${title}: ${message.ask}`);
                  setMessage(null);
                }}
              >
                Ask the agent
              </ChipButton>
            )}
          </div>
        )}
      </div>

      <section aria-label="Structure details" className={`border-subtle bg-raised flex shrink-0 flex-col border-t ${detailsOpen ? "h-[min(42%,340px)]" : ""}`}>
        <div className="flex h-9 shrink-0 items-center gap-2 px-3">
          <button type="button" onClick={() => setDetailsOpen((open) => !open)} className="text-muted hover:text-body cursor-pointer text-[11px] font-semibold">
            {detailsOpen ? "Hide details" : "Show details"}
          </button>
          <span className="text-faint text-[11px]">
            {profile ? `${profile.totalContacts} contact${profile.totalContacts === 1 ? "" : "s"}` : "no contact profile"}
          </span>
          {detailsOpen && (
            <nav role="tablist" className="ml-auto flex gap-0.5">
              {(["contacts", "quality", "files"] as const).map((tab) => (
                <button
                  key={tab}
                  type="button"
                  role="tab"
                  aria-selected={drawerTab === tab}
                  onClick={() => setDrawerTab(tab)}
                  className={`-mb-px cursor-pointer border-b-2 px-2 py-1 text-[11px] capitalize ${drawerTab === tab ? "text-body" : "text-muted hover:text-body border-transparent"}`}
                  style={drawerTab === tab ? { borderColor: "var(--color-accent)" } : undefined}
                >
                  {tab}
                </button>
              ))}
            </nav>
          )}
        </div>
        {detailsOpen && (
          <div className="border-subtle min-h-0 flex-1 overflow-y-auto border-t px-3 py-2">
            {drawerTab === "contacts" && (
              <ContactsTab
                profile={profile}
                smiles={active.smiles ?? profile?.smiles ?? null}
                theme={theme}
                typeFilter={typeFilter}
                onTypeFilter={(type) => {
                  setTypeFilter(type);
                  if (type && profile) showContacts(profile.contacts.filter((c) => c.type === type), `${contactLabel(type).toLowerCase()} contacts`);
                }}
                query={query}
                onQuery={setQuery}
                selected={selectedHere}
                onSelect={(contact) => {
                  if (!profile) return;
                  setSelected({ path: active.path, index: profile.contacts.indexOf(contact) });
                  if (contact.resnum !== null) viewRef.current?.focusResidues([{ chain: contact.chain, resnum: contact.resnum }]);
                }}
                onProfile={() => onAsk(`Profile the protein-ligand interactions of ${active.path}.`)}
              />
            )}
            {drawerTab === "quality" && <QualityTab panel={active} verdict={verdict} />}
            {drawerTab === "files" && <FilesTab panel={active} at={structures[index]?.at} tool={structures[index]?.tool ?? active.tool} />}
          </div>
        )}
      </section>
    </div>
  );
}

// ---------------------------------------------------------------------------

function Badge({ tone, children }: { tone: "structure" | "ok" | "bad" | "muted"; children: ReactNode }) {
  const colour = { structure: "var(--color-structure)", ok: "var(--color-ok)", bad: "var(--color-accent)", muted: "var(--text-faint)" }[tone];
  return (
    <span className="shrink-0 rounded-full border px-1.5 py-px text-[10px] font-semibold whitespace-nowrap" style={{ color: colour, borderColor: colour }}>
      {children}
    </span>
  );
}

export function GatePill({ verdict }: { verdict: Gate }) {
  if (verdict.status === "pass") return <Badge tone="ok">Gate passed</Badge>;
  if (verdict.status === "fail") return <Badge tone="bad">Gate failed</Badge>;
  return null;
}

function GateStrip({ verdict, panel }: { verdict: Gate; panel: StructurePanel }) {
  const m = panel.metrics;
  if (!m || verdict.status === "unknown") return null;
  const values = verdict.checks.filter((check) => check.value !== "—").map((check) => `${check.label.toLowerCase()} ${check.value}`);
  if (verdict.status === "fail") {
    return (
      <div className="border-subtle flex shrink-0 items-baseline gap-2 overflow-hidden border-b px-3 py-1.5 text-[11px]" style={{ background: "var(--color-accent-soft)" }}>
        <strong className="shrink-0" style={{ color: "var(--color-accent)" }}>
          Affinity withheld
        </strong>
        <span className="truncate" title={verdict.reasons.join(" · ")}>
          {verdict.reasons.join(" · ")}
        </span>
      </div>
    );
  }
  const affinity = affinityText(m);
  return (
    <div className="border-subtle flex shrink-0 items-baseline gap-2 overflow-hidden border-b px-3 py-1.5 text-[11px]" style={{ background: "color-mix(in oklch, var(--color-ok) 8%, transparent)" }}>
      <span className="text-body truncate">
        {values.join(" · ")}
        {affinity && ` · affinity ${affinity} log₁₀(IC₅₀/µM)`}
      </span>
      <span className="text-faint shrink-0">screening estimate</span>
    </div>
  );
}

function HeadButton({ label, title, onClick, pressed }: { label: string; title: string; onClick: () => void; pressed?: boolean }) {
  return (
    <button
      type="button"
      title={title}
      aria-pressed={pressed}
      onClick={onClick}
      className={`cursor-pointer rounded-md px-2 py-1 text-[11px] ${pressed ? "bg-inset text-body" : "text-muted hover:text-body hover:bg-inset"}`}
    >
      {label}
    </button>
  );
}

function OverlayButton({ label, title, onClick, pressed }: { label: string; title: string; onClick: () => void; pressed?: boolean }) {
  return (
    <button
      type="button"
      title={title}
      aria-pressed={pressed}
      onClick={onClick}
      className={`bg-raised cursor-pointer rounded border px-2 py-0.5 text-[11px] backdrop-blur-sm ${
        pressed ? "text-body border-[var(--color-accent)]" : "text-muted border-subtle hover:text-body hover:border-[var(--color-accent)]"
      }`}
    >
      {label}
    </button>
  );
}

function ChipButton({ children, onClick, primary }: { children: ReactNode; onClick: () => void; primary?: boolean }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`cursor-pointer rounded-full px-2 py-0.5 text-[11px] font-medium whitespace-nowrap ${primary ? "text-white" : "bg-inset text-body"}`}
      style={primary ? { background: "var(--color-accent)" } : undefined}
    >
      {children}
    </button>
  );
}

function StyleMenu({
  style,
  color,
  background,
  spinning,
  onAction,
}: {
  style: ViewStyleName;
  color: ColorName | null;
  background: BackgroundName | null;
  spinning: boolean;
  onAction: (action: ViewerAction) => void;
}) {
  const item = (label: string, active: boolean, action: ViewerAction, hint?: string) => (
    <button
      key={label}
      type="button"
      title={hint}
      onClick={() => onAction(action)}
      className={`flex w-full cursor-pointer items-center gap-2 rounded px-2 py-1 text-left text-[11px] ${active ? "bg-inset font-semibold" : "hover:bg-sunken"}`}
    >
      <span className="w-2" aria-hidden="true">
        {active ? "•" : ""}
      </span>
      {label}
    </button>
  );
  return (
    <div className="bg-raised border-subtle shadow-card absolute top-full left-0 z-40 mt-1 grid w-64 grid-cols-2 gap-x-2 rounded-lg border p-2">
      <div>
        <p className="text-faint px-2 pb-1 text-[10px] font-semibold uppercase">Show as</p>
        {(Object.keys(VIEW_STYLES) as ViewStyleName[]).map((name) =>
          item(VIEW_STYLES[name].label, style === name, { kind: "style", style: name }, VIEW_STYLES[name].hint),
        )}
        <p className="text-faint px-2 pt-2 pb-1 text-[10px] font-semibold uppercase">Background</p>
        {(Object.keys(BACKGROUNDS) as BackgroundName[]).map((name) => item(name[0].toUpperCase() + name.slice(1), background === name, { kind: "background", background: name }))}
      </div>
      <div>
        <p className="text-faint px-2 pb-1 text-[10px] font-semibold uppercase">Colour by</p>
        {(Object.keys(COLOR_THEMES) as ColorName[]).map((name) => item(COLOR_LABELS[name], color === name, { kind: "color", color: name }))}
        <p className="text-faint px-2 pt-2 pb-1 text-[10px] font-semibold uppercase">More</p>
        {item(spinning ? "Stop spinning" : "Spin", spinning, { kind: "spin", on: !spinning })}
        {item("Save image", false, { kind: "screenshot" })}
        {item("Clear selection", false, { kind: "clear" })}
      </div>
    </div>
  );
}

function CommandBar({ onRun, onUnknown }: { onRun: (action: ViewerAction) => void; onUnknown: (text: string) => void }) {
  const [value, setValue] = useState("");
  const [focused, setFocused] = useState(false);
  const [history, setHistory] = useState<string[]>([]);
  const [cursor, setCursor] = useState(-1);
  const needle = value.trim().toLowerCase();
  const suggestions = focused ? VIEWER_EXAMPLES.filter((example) => !needle || example.includes(needle)).slice(0, 4) : [];

  const run = (text: string) => {
    const command = text.trim();
    if (!command) return;
    const action = parseViewerCommand(command);
    if (action) onRun(action);
    else onUnknown(command);
    setHistory((previous) => [command, ...previous.filter((entry) => entry !== command)].slice(0, 20));
    setCursor(-1);
    setValue("");
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    run(value);
  };

  return (
    <div className="pointer-events-none absolute right-2 bottom-2 left-2 z-20">
      <form
        onSubmit={submit}
        className="bg-sunken border-subtle pointer-events-auto mx-auto flex h-7 max-w-[520px] items-center border focus-within:border-[var(--color-accent)]"
      >
        <span aria-hidden="true" className="w-[25px] text-center text-sm" style={{ color: "var(--color-accent)" }}>
          ⌘
        </span>
        <input
          value={value}
          onChange={(event) => setValue(event.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          onKeyDown={(event) => {
            if (event.key === "Escape") event.currentTarget.blur();
            if (event.key === "ArrowUp" && history.length) {
              event.preventDefault();
              const next = Math.min(cursor + 1, history.length - 1);
              setCursor(next);
              setValue(history[next]);
            }
            if (event.key === "ArrowDown" && cursor >= 0) {
              event.preventDefault();
              const next = cursor - 1;
              setCursor(next);
              setValue(next < 0 ? "" : history[next]);
            }
          }}
          spellCheck={false}
          autoComplete="off"
          aria-label="Viewer command"
          placeholder="Ask the viewer… fit, focus ligand, pocket, focus ASN140, show hbonds"
          className="text-body placeholder:text-muted h-full min-w-0 flex-1 bg-transparent px-1 text-[11px] outline-none"
        />
        <button
          type="submit"
          className="bg-inset text-muted mr-[3px] h-[22px] cursor-pointer rounded px-[7px] text-[10px] hover:bg-[var(--color-accent)] hover:text-white"
        >
          Run
        </button>
      </form>
      {suggestions.length > 0 && (
        <div className="pointer-events-auto mx-auto flex max-w-[520px] flex-wrap justify-center gap-[3px] pt-[3px]">
          {suggestions.map((suggestion) => (
            <button
              key={suggestion}
              type="button"
              onMouseDown={(event) => {
                event.preventDefault();
                run(suggestion);
              }}
              className="bg-sunken border-subtle text-muted hover:text-body cursor-pointer border px-1.5 py-0.5 font-mono text-[10px] hover:border-[var(--color-accent)]"
            >
              {suggestion}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function ContactsTab({
  profile,
  smiles,
  theme,
  typeFilter,
  onTypeFilter,
  query,
  onQuery,
  selected,
  onSelect,
  onProfile,
}: {
  profile: InteractionsPanel | null;
  smiles: string | null;
  theme: Theme;
  typeFilter: string | null;
  onTypeFilter: (type: string | null) => void;
  query: string;
  onQuery: (text: string) => void;
  selected: Contact | null;
  onSelect: (contact: Contact) => void;
  onProfile: () => void;
}) {
  if (!profile) {
    return (
      <div className="text-muted space-y-2 py-4 text-center text-xs">
        <p>No interaction profile for this structure.</p>
        <button type="button" onClick={onProfile} className="border-subtle hover:border-strong cursor-pointer rounded-full border px-3 py-1 text-[11px]">
          Ask the agent to profile it
        </button>
      </div>
    );
  }
  if (profile.error) return <p className="py-4 text-center text-xs" style={{ color: "var(--color-accent)" }}>{profile.error}</p>;

  const needle = query.trim().toLowerCase();
  const rows = profile.contacts.filter(
    (contact) =>
      (!typeFilter || contact.type === typeFilter) &&
      (!needle || [contact.type, contact.label, contact.fragment, contact.ligandAtom, contact.receptorAtom].join(" ").toLowerCase().includes(needle)),
  );
  const highlight = selected?.ligandAtomIndex != null && selected.ligandAtomIndex >= 0 ? [selected.ligandAtomIndex] : undefined;

  return (
    <div className="flex gap-3">
      <div className="min-w-0 flex-1">
        <div className="mb-2 flex flex-wrap items-center gap-1.5">
          <input
            type="search"
            value={query}
            onChange={(event) => onQuery(event.target.value)}
            placeholder="Filter contacts"
            aria-label="Filter contacts"
            className="bg-sunken border-subtle w-32 rounded border px-2 py-0.5 text-[11px] outline-none focus:border-[var(--color-accent)]"
          />
          {sortedTypes(profile.counts).map((type) => (
            <button
              key={type}
              type="button"
              aria-pressed={typeFilter === type}
              onClick={() => onTypeFilter(typeFilter === type ? null : type)}
              title={`Show ${contactLabel(type).toLowerCase()} contacts on the structure`}
              className={`flex cursor-pointer items-center gap-1 rounded-full border px-1.5 py-px text-[10px] ${typeFilter === type ? "border-strong bg-inset text-body" : "border-subtle text-muted hover:text-body"}`}
            >
              <span className="size-1.5 rounded-full" style={{ background: contactColor(type) }} />
              {contactLabel(type)}
              <b className="font-semibold">{profile.counts[type]}</b>
            </button>
          ))}
        </div>
        <ul className="space-y-px">
          {rows.map((contact, index) => {
            const isSelected = selected === contact;
            return (
              <li key={index}>
                <button
                  type="button"
                  onClick={() => onSelect(contact)}
                  className={`grid w-full cursor-pointer grid-cols-[minmax(7rem,1fr)_4.5rem_minmax(6rem,1.3fr)] items-center gap-2 rounded border px-1.5 py-1 text-left text-[11px] ${
                    isSelected ? "border-[var(--color-accent)] bg-[var(--color-accent-soft)]" : "hover:bg-sunken border-transparent"
                  }`}
                >
                  <span className="flex min-w-0 items-center gap-1.5 truncate">
                    <span className="size-1.5 shrink-0 rounded-full" style={{ background: contactColor(contact.type) }} />
                    {contactLabel(contact.type)}
                  </span>
                  <strong className="font-mono">
                    {contact.label}
                    {contact.chain && <span className="text-faint font-normal">.{contact.chain}</span>}
                  </strong>
                  <span className="text-muted truncate font-mono">
                    {contact.distance != null ? `${contact.distance.toFixed(2)} Å` : "—"}
                    {contact.fragment ? ` · ${contact.fragment}` : ""}
                    {contact.receptorAtom ? ` · ${contact.receptorAtom}` : ""}
                  </span>
                </button>
              </li>
            );
          })}
          {rows.length === 0 && <li className="text-muted py-3 text-center text-[11px]">No matching contacts.</li>}
        </ul>
        {profile.totalContacts > profile.contacts.length && (
          <p className="text-faint mt-2 text-[10px]">
            Showing {profile.contacts.length} of {profile.totalContacts} contacts.
          </p>
        )}
      </div>
      {smiles && (
        <div className="hidden w-36 shrink-0 sm:block">
          <div className="bg-sunken border-subtle rounded-lg border p-1">
            <MolDepiction smiles={smiles} theme={theme} width={200} height={160} atoms={highlight} className="h-28 w-full" />
          </div>
          <p className="text-faint mt-1 text-center text-[10px]">
            {selected ? `${selected.ligandAtom ?? "ligand atom"} → ${selected.label}` : "Pick a contact to see the ligand atom"}
          </p>
        </div>
      )}
    </div>
  );
}

function QualityTab({ panel, verdict }: { panel: StructurePanel; verdict: Gate }) {
  const m = panel.metrics;
  if (!m) return <p className="text-muted py-4 text-center text-xs">No confidence values were reported for this structure.</p>;
  const affinity = affinityText(m);
  const extra: [string, string][] = [];
  if (m.iptm !== undefined) extra.push(["Interface ipTM", m.iptm.toFixed(2)]);
  if (m.plddt !== undefined) extra.push(["Complex pLDDT", m.plddt.toFixed(2)]);

  return (
    <div className="grid gap-4 text-[11px] sm:grid-cols-2">
      <div>
        <p className="text-faint mb-1 text-[10px] font-semibold uppercase">Confidence gate</p>
        <table className="w-full">
          <tbody>
            {verdict.checks.map((check) => (
              <tr key={check.label} className="border-subtle border-b last:border-b-0">
                <td className="text-muted py-1">{check.label}</td>
                <td className="py-1 text-right font-mono tabular-nums">{check.value}</td>
                <td className="text-faint py-1 pl-2">{check.rule}</td>
                <td className="py-1 pl-2 text-right">
                  {check.ok === null ? (
                    <span className="text-faint">not run</span>
                  ) : (
                    <span style={{ color: check.ok ? "var(--color-ok)" : "var(--color-accent)" }}>{check.ok ? "✓" : "✕"}</span>
                  )}
                </td>
              </tr>
            ))}
            {extra.map(([label, value]) => (
              <tr key={label} className="border-subtle border-b last:border-b-0">
                <td className="text-muted py-1">{label}</td>
                <td className="py-1 text-right font-mono tabular-nums">{value}</td>
                <td colSpan={2} />
              </tr>
            ))}
          </tbody>
        </table>
        {m.failedChecks && m.failedChecks.length > 0 && (
          <p className="mt-2" style={{ color: "var(--color-accent)" }}>
            Failed PoseBusters checks: {m.failedChecks.join(", ")}
          </p>
        )}
      </div>
      <div>
        <p className="text-faint mb-1 text-[10px] font-semibold uppercase">Affinity</p>
        {verdict.status === "pass" && affinity && m.affinity !== undefined ? (
          <div className="space-y-1">
            <p>
              <span className="font-mono text-sm font-semibold tabular-nums">{affinity}</span> log₁₀(IC₅₀/1 µM)
            </p>
            <p className="text-muted">
              {affinityConcentration(m.affinity)} · pIC₅₀ ≈ {pIC50(m.affinity).toFixed(1)}
            </p>
            {m.binderProbability !== undefined && <p className="text-muted">Binder probability {m.binderProbability.toFixed(2)}</p>}
            <p className="text-faint">
              A ranking signal, not a measured constant. The unit is six log units away from pIC₅₀.
              {affinity.includes("to") && " The two ensemble members disagree by more than one log unit, so this is a range."}
            </p>
          </div>
        ) : (
          <div className="space-y-1">
            <p className="font-semibold" style={{ color: verdict.status === "fail" ? "var(--color-accent)" : undefined }}>
              {verdict.status === "fail" ? "Withheld: the pose failed the gate" : m.affinity === undefined ? "Not predicted" : "Withheld until the gate is known"}
            </p>
            {verdict.reasons.length > 0 && (
              <ul className="text-muted list-disc pl-4">
                {verdict.reasons.map((reason) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ul>
            )}
            {m.binderProbability !== undefined && <p className="text-muted">Binder probability {m.binderProbability.toFixed(2)} may still serve as a rough guide.</p>}
          </div>
        )}
      </div>
    </div>
  );
}

function FilesTab({ panel, tool, at }: { panel: StructurePanel; tool: string; at?: number }) {
  const rows: [string, ReactNode][] = [
    ["File", panel.name],
    ["Format", panel.format.toUpperCase()],
    [
      "Path",
      <span key="path" className="flex items-center gap-1">
        <span className="truncate font-mono">{panel.path}</span>
        <CopyButton value={panel.path} label="Copy path" />
      </span>,
    ],
    ["From", tool],
  ];
  if (at) rows.push(["At", clock(at)]);
  if (panel.smiles)
    rows.push([
      "Ligand",
      <span key="smiles" className="flex items-center gap-1">
        <span className="truncate font-mono">{panel.smiles}</span>
        <CopyButton value={panel.smiles} label="Copy SMILES" />
      </span>,
    ]);
  return (
    <dl className="grid grid-cols-[5rem_1fr] gap-x-3 gap-y-1 text-[11px]">
      {rows.map(([label, value]) => (
        <div key={label} className="contents">
          <dt className="text-muted">{label}</dt>
          <dd className="min-w-0 truncate">{value}</dd>
        </div>
      ))}
      <div className="contents">
        <dt />
        <dd>
          <a href={panel.url} download={panel.name} className="font-medium" style={{ color: "var(--color-accent)" }}>
            Download {panel.format.toUpperCase()}
          </a>
        </dd>
      </div>
    </dl>
  );
}
