// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useRef, useState, type FormEvent, type ReactNode } from "react";
import type { Contact, InteractionsPanel, PropertiesPanel, StructureMetrics, StructurePanel } from "../../types/api";
import type { Theme } from "../../hooks/useTheme";
import type { Sourced } from "../../lib/artifacts";
import { contactsFor, runOf, sameFile, structureTitle } from "../../lib/artifacts";
import {
  BACKGROUNDS,
  COLOR_THEMES,
  StructureView,
  VIEW_STYLES,
  cssColor,
  type BackgroundName,
  type ColorName,
  type Comparison,
  type Picked,
  type Residue,
  type ViewStyleName,
} from "../../lib/molstar";
import { VIEWER_HELP, type ViewerAction } from "../../lib/viewerCommands";
import { onViewerRequest } from "../../lib/viewerBus";
import { contactColor, contactLabel, contactLines, matchesTypes, residuesOf, sortedTypes, toggleType } from "../../lib/contacts";
import { STRUCTURE_KINDS, structureKind, structureOption } from "../../lib/poses";
import { buildNumbering, displayNumber, rememberNumbering, residueLabel, type FileResidue, type Numbering } from "../../lib/numbering";
import { locateResidue } from "../../lib/residueTarget";
import { profileError } from "../../lib/profileError";
import { matchStructure } from "../../lib/residueLinks";
import { boltzOutput, chainSequences, fastaFile, inputFiles, parseBoltzInput, smilesFile, type BoltzInput, type RelatedFile } from "../../lib/structureFiles";
import { metricsFromValues, predictionGroups, propertyFacts, type Fact, type FactGroup } from "../../lib/moleculeFacts";
import { canonical, descriptors } from "../../lib/rdkit";
import { useRDKit } from "../../hooks/useRDKit";
import { fetchPdbEntry, isPdbId } from "../../lib/rcsb";
import { MolDepiction } from "../MolDepiction/MolDepiction";
import { DetailsColumns } from "./DetailsColumns";
import { cofoldBars, predictionBars, propertyBars } from "./detailFacts";
import { pickKind, pickQuestion, pickTitle, structureRef, type PickedResidue } from "./pickQuestion";

interface Props {
  structures: Sourced<StructurePanel>[];
  active: StructurePanel;
  onSelect: (panel: StructurePanel) => void;
  interactions: Sourced<InteractionsPanel>[];
  /** The conversation's ADMET and RetroDMTA predictions, for the ligand's in Details. */
  properties?: Sourced<PropertiesPanel>[];
  /** The target's canonical UniProt sequence, when the conversation looked it up. */
  targetSequence?: string | null;
  /** The conversation, for structures the user brings in; null before it has an id. */
  sessionId?: string | null;
  theme: Theme;
  /** Put a prompt in the composer. */
  onAsk: (text: string) => void;
}

type DrawerTab = "contacts" | "details";
interface Message {
  id: number;
  text: string;
  tone: "ok" | "bad";
  /** A request the viewer cannot serve, offered to the agent instead. */
  ask?: string;
}
/** Which contacts are drawn as dashes: every one (ligand or pocket focus) or those of some residues. */
interface DashScope {
  path: string;
  residues: { chain: string | null; resnum: number }[] | null;
}

const COLOR_LABELS: Record<ColorName, string> = {
  chain: "Chain",
  element: "Element",
  residue: "Residue type",
  hydrophobicity: "Hydrophobicity",
  confidence: "Confidence (pLDDT)",
  secondary: "Secondary structure",
};

const OPEN_PDB = "\u0000open-pdb";
type Side = "primary" | "other";

const sameResidue = (r: { chain?: string | null; resnum: number }, c: Contact) =>
  c.resnum === r.resnum && (!r.chain || !c.chain || c.chain === r.chain);
const residueKey = (chain: string | null | undefined, resnum: number | null) => `${chain ?? ""}:${resnum}`;

/**
 * The 3D structure workspace: one Mol* viewport for every complex in the
 * conversation, with quick views on it, and its contacts, details and files
 * under it.
 */
export function StructureWorkspace({ structures, active, onSelect, interactions, properties, targetSequence, sessionId, theme, onAsk }: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const viewRef = useRef<StructureView | null>(null);
  const pendingRef = useRef<ViewerAction | null>(null);

  const [viewReady, setViewReady] = useState(false);
  const [loadedPath, setLoadedPath] = useState<string | null>(null);
  const [loads, setLoads] = useState(0);
  const [error, setError] = useState<{ path: string | null; text: string } | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [message, setMessage] = useState<Message | null>(null);
  const [picked, setPicked] = useState<(Picked & { path: string }) | null>(null);
  const [selected, setSelected] = useState<{ path: string; index: number } | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [drawerTab, setDrawerTab] = useState<DrawerTab>("contacts");
  const [types, setTypes] = useState<string[]>([]);
  const [query, setQuery] = useState("");
  const [controls, setControls] = useState(false);
  const [style, setStyle] = useState<ViewStyleName>("cartoon");
  const [color, setColor] = useState<ColorName | null>(null);
  const [background, setBackground] = useState<BackgroundName | null>(null);
  const [spinning, setSpinning] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [fileResidues, setFileResidues] = useState<{ path: string; list: FileResidue[] } | null>(null);
  const [dashScope, setDashScope] = useState<DashScope | null>(null);
  const [flash, setFlash] = useState<{ keys: string[]; at: number } | null>(null);
  const [pdbForm, setPdbForm] = useState<"open" | "compare" | null>(null);
  /** The structure overlaid on *primary*; it goes when another structure is shown. */
  const [compare, setCompare] = useState<{ primary: string; other: StructurePanel } | null>(null);
  const [comparison, setComparison] = useState<{ key: string; result: Comparison | null; error?: string } | null>(null);
  const [hiddenSides, setHiddenSides] = useState<{ key: string; sides: Side[] } | null>(null);

  const profile = useMemo(() => contactsFor(active, interactions), [active, interactions]);
  const numbering = useMemo(
    () => (fileResidues && fileResidues.path === active.path ? buildNumbering(fileResidues.list, targetSequence) : null),
    [fileResidues, active.path, targetSequence],
  );
  const index = structures.findIndex(({ item }) => item.path === active.path);
  const title = structureTitle(active);
  const loaded = loadedPath === active.path;
  const failure = error && (error.path === null || error.path === active.path) ? error.text : null;
  const backgroundValue = BACKGROUNDS[background ?? (theme === "dark" ? "black" : "white")];
  const lightCanvas = backgroundValue === BACKGROUNDS.white || backgroundValue === BACKGROUNDS.light;
  const kind = structureKind(active.path, Boolean(active.metrics), active.tool);

  /** "LEU94" in UniProt numbering when known, with the file's number when it differs. */
  const nameOf = (resname: string, chain: string | null | undefined, resnum: number) => residueLabel(numbering, resname, chain, resnum);
  const nameInFile = (r: { chain?: string | null; resnum: number }) =>
    profile?.contacts.find((c) => sameResidue(r, c))?.resname ??
    numbering?.names.get(residueKey(r.chain, r.resnum)) ??
    fileResidues?.list.find((f) => f.resnum === r.resnum && (!r.chain || f.chain === r.chain))?.resname ??
    "";

  const pathRef = useRef(active.path);
  const profileRef = useRef(profile);
  useEffect(() => {
    pathRef.current = active.path;
    profileRef.current = profile;
  }, [active.path, profile]);

  useEffect(() => {
    if (numbering || fileResidues?.path === active.path) rememberNumbering(active.path, numbering);
  }, [numbering, fileResidues, active.path]);

  /** A residue clicked in 3D: its contacts in the drawer and as dashes. */
  const onResiduePicked = (pick: (Picked & { path: string }) | null) => {
    setPicked(pick);
    if (!pick || pick.resnum === null) {
      setDashScope(null);
      return;
    }
    const residue = { chain: pick.chain, resnum: pick.resnum };
    setDashScope({ path: pick.path, residues: [residue] });
    const current = profileRef.current;
    const first = current ? current.contacts.findIndex((c) => sameResidue(residue, c)) : -1;
    if (first < 0) return;
    setSelected({ path: pick.path, index: first });
    setFlash({ keys: [residueKey(pick.chain, pick.resnum)], at: Date.now() });
    setDrawerTab("contacts");
    setDetailsOpen(true);
  };
  const onResiduePickedRef = useRef(onResiduePicked);
  useEffect(() => {
    onResiduePickedRef.current = onResiduePicked;
  });

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
        view.onPick((pick) => onResiduePickedRef.current(pick && pick.resnum !== null ? { ...pick, path: pathRef.current } : null));
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
        setFileResidues({ path: active.path, list: view.residues() });
        setLoadedPath(active.path);
        setLoads((n) => n + 1);
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

  // The profiler's contacts as dashes, for what is in focus and the chosen types.
  const dashed = useMemo(() => {
    if (!profile || !dashScope || dashScope.path !== active.path) return [];
    const residues = dashScope.residues;
    return profile.contacts.filter((c) => matchesTypes(c, types) && (!residues || residues.some((r) => sameResidue(r, c))));
  }, [profile, dashScope, active.path, types]);
  const canvasTheme = lightCanvas ? "light" : "dark";
  useEffect(() => {
    if (loaded) void viewRef.current?.setContacts(contactLines(dashed, canvasTheme));
  }, [loaded, dashed, canvasTheme, backgroundValue]);

  const comparing = compare && compare.primary === active.path ? compare.other : null;
  const compareKey = comparing ? `${active.path}\u0000${comparing.path}` : null;
  const compared = comparison && comparison.key === compareKey ? comparison : null;
  const hidden = hiddenSides && hiddenSides.key === compareKey ? hiddenSides.sides : [];
  const compareColors = () => {
    const host = hostRef.current ?? document.documentElement;
    return { primary: cssColor(host, "--color-molecule"), other: cssColor(host, "--color-structure"), ghost: cssColor(host, "--text-muted") };
  };

  // A new load clears the viewport, overlay included, so it is drawn again after each one.
  useEffect(() => {
    const view = viewRef.current;
    if (!view || !loaded) return;
    if (!comparing || !compareKey) {
      if (view.comparing) void view.endCompare();
      return;
    }
    let stale = false;
    view
      .compare(comparing.url, comparing.format, structureTitle(comparing), compareColors())
      .then((result) => {
        if (!stale && result) setComparison({ key: compareKey, result });
      })
      .catch((reason: unknown) => {
        if (!stale) setComparison({ key: compareKey, result: null, error: reason instanceof Error ? reason.message : String(reason) });
      });
    return () => {
      stale = true;
    };
  }, [loaded, loads, comparing, compareKey]);

  // The app switches the theme class after this renders.
  const overlaid = Boolean(compared?.result);
  useEffect(() => {
    if (!overlaid) return;
    const frame = requestAnimationFrame(() => void viewRef.current?.setCompareColors(compareColors()));
    return () => cancelAnimationFrame(frame);
  }, [theme, overlaid]);

  const toggleSide = (side: Side) => {
    if (!compareKey) return;
    const sides = hidden.includes(side) ? hidden.filter((s) => s !== side) : [...hidden, side];
    setHiddenSides({ key: compareKey, sides });
    viewRef.current?.setCompareVisible(side, !sides.includes(side));
  };

  const compareWith = (panel: StructurePanel) => {
    setCompare({ primary: active.path, other: panel });
    setHiddenSides(null);
  };

  const comparePdb = async (id: string) => {
    if (!isPdbId(id)) {
      say(`“${id}” is not a PDB id: four characters starting with a digit, like 3MXF.`, "bad");
      return;
    }
    try {
      compareWith(await fetchPdbEntry(id, sessionId ?? "new"));
    } catch (reason) {
      say(reason instanceof Error ? reason.message : String(reason), "bad");
    }
  };

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

  /** Focus residues (file numbering), dash their contacts and flash their rows. */
  const focusOn = (view: StructureView, residues: { chain: string | null; resnum: number }[], mode: "focus" | "highlight" = "focus") => {
    if (!view.focusResidues(residues, mode)) return false;
    if (mode === "focus") setDashScope({ path: active.path, residues });
    setFlash({ keys: residues.map((r) => residueKey(r.chain, r.resnum)), at: Date.now() });
    return true;
  };

  const focusLigand = (view: StructureView, radius?: number) => {
    const ok = radius === undefined ? view.focusLigand() : view.focusPocket(radius);
    if (ok) setDashScope({ path: active.path, residues: null });
    return ok;
  };

  const openPdb = async (id: string) => {
    if (!isPdbId(id)) {
      say(`“${id}” is not a PDB id: four characters starting with a digit, like 3MXF.`, "bad");
      return;
    }
    say(`Fetching ${id.toUpperCase()} from the PDB…`);
    try {
      const panel = await fetchPdbEntry(id, sessionId ?? "new");
      onSelect(panel);
      say(`${id.toUpperCase()} from the RCSB PDB.`);
    } catch (reason) {
      say(reason instanceof Error ? reason.message : String(reason), "bad");
    }
  };

  /** Run one viewer action and say what happened. */
  function execute(action: ViewerAction) {
    if (action.kind === "pdb") {
      void openPdb(action.id);
      return;
    }
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
        if (focusLigand(view)) say(profile ? "The ligand, its neighbours as sticks, and its contacts from the profile." : "The ligand and the residues within 5 Å.");
        else say("This structure has no ligand.", "bad");
        break;
      case "pocket":
        if (focusLigand(view, action.radius)) say(`The pocket: residues within ${action.radius} Å of the ligand.`);
        else say("This structure has no ligand.", "bad");
        break;
      case "residues": {
        const inFile = fileResidues?.path === active.path ? fileResidues.list : [];
        const located = action.residues.map((r: Residue) => locateResidue(inFile, numbering, r));
        const residues = located.flatMap((at) => at ?? []);
        const asked = (r: Residue) => (r.resname ? residueLabel(null, r.resname, null, r.resnum).label : String(r.resnum)) + (r.insCode ?? "");
        if (!residues.length || !focusOn(view, residues, action.mode)) {
          say(`${action.residues.map(asked).join(", ")} is not in ${title}.`, "bad");
          break;
        }
        const names = [...new Set(residues.map((r) => nameOf(nameInFile(r) || action.residues[0].resname || "", r.chain, r.resnum).label + (r.insCode ?? "")))].join(", ");
        const missing = action.residues.filter((_, i) => !located[i]).map(asked);
        const count = profile ? profile.contacts.filter((c) => residues.some((r) => sameResidue(r, c))).length : 0;
        say(
          `${names}${action.mode !== "highlight" && count ? ` · ${count} contact${count === 1 ? "" : "s"}` : ""}${missing.length ? ` · ${missing.join(", ")} not in ${title}` : ""}`,
          missing.length ? "bad" : "ok",
        );
        break;
      }
      case "contacts": {
        if (!profile) {
          say("No interaction profile for this structure yet.", "bad", "Profile the interactions of this complex.");
          break;
        }
        const list = action.type ? profile.contacts.filter((c) => c.type === action.type) : profile.contacts;
        setTypes(action.type ? [action.type] : []);
        openDetails("contacts");
        if (!list.length || !focusLigand(view)) {
          say(`No ${action.type ? `${contactLabel(action.type).toLowerCase()} ` : ""}contacts in this structure.`, "bad");
          break;
        }
        const names = residuesOf(list).map((r) => nameOf(nameInFile(r), r.chain, r.resnum).label);
        say(`${list.length} ${action.type ? `${contactLabel(action.type).toLowerCase()} ` : ""}contacts: ${names.slice(0, 6).join(", ")}${names.length > 6 ? "…" : ""}`);
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
        setDashScope(null);
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
        openDetails(action.tab === "contacts" ? "contacts" : "details");
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
        const target = request.path ?? (request.ref ? (matchStructure(request.ref, structures.map(({ item }) => item.path), request.near) ?? request.near ?? null) : null);
        if (target && !sameFile(target, pathRef.current)) {
          const known = structures.find(({ item }) => sameFile(item.path, target));
          // A path is opened by the app, a name here; either way the action runs once that structure is loaded.
          if (known || (request.path && /\.(pdb|cif|mmcif)$/i.test(target))) {
            pendingRef.current = request.action;
            if (!request.path && known) onSelect(known.item);
            return;
          }
        }
        executeRef.current(request.action);
      }),
    [structures, onSelect],
  );

  /** A chip: show only that type, add it, or drop it; with nothing in focus, focus the ligand so the dashes show. */
  const onToggleType = (type: string) => {
    setTypes((current) => toggleType(current, type));
    const view = viewRef.current;
    if (view && loaded && (!dashScope || dashScope.path !== active.path)) focusLigand(view);
  };

  const pickedHere = picked && picked.path === active.path ? picked : null;
  const pickedContacts = pickedHere && profile ? profile.contacts.filter((c) => c.resnum === pickedHere.resnum) : [];
  const pickedShown = pickedHere?.resname && pickedHere.resnum !== null ? nameOf(pickedHere.resname, pickedHere.chain, pickedHere.resnum) : null;
  const pickedInPolymer =
    pickedHere?.resname && fileResidues?.path === active.path
      ? fileResidues.list.some(
          (r) => r.resnum === pickedHere.resnum && (!pickedHere.chain || r.chain === pickedHere.chain) && r.resname.toUpperCase() === pickedHere.resname?.toUpperCase(),
        )
      : null;
  const pickedResidue: PickedResidue | null =
    pickedHere?.resname && pickedShown
      ? { kind: pickKind(pickedHere.resname, pickedInPolymer), resname: pickedHere.resname, label: pickedShown.label, chain: pickedHere.chain, file: pickedShown.file }
      : null;
  const pickedTitle = pickedResidue ? pickTitle(pickedResidue) : (pickedHere?.label ?? "");
  const selectedHere = selected && selected.path === active.path && profile ? profile.contacts[selected.index] : null;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <header className="border-subtle flex shrink-0 items-center gap-2 border-b px-3 py-1.5">
        {pdbForm ? (
          <PdbForm
            action={pdbForm === "compare" ? "Compare" : "Open"}
            onOpen={(id) => {
              setPdbForm(null);
              void (pdbForm === "compare" ? comparePdb(id) : openPdb(id));
            }}
            onCancel={() => setPdbForm(null)}
          />
        ) : (
          <select
            value={active.path}
            onChange={(event) => {
              if (event.target.value === OPEN_PDB) {
                setPdbForm("open");
                return;
              }
              const next = structures.find(({ item }) => item.path === event.target.value);
              if (next) onSelect(next.item);
            }}
            aria-label="Structure"
            title={active.path}
            className="hover:bg-inset text-body min-w-0 flex-1 cursor-pointer truncate rounded-md bg-transparent px-1 py-0.5 text-[13px] font-semibold outline-none"
          >
            {index < 0 && <option value={active.path}>{structureOption(title, active.path, runOf(active.path))}</option>}
            {structures.map(({ item }, i) => (
              <option key={item.path} value={item.path}>
                {structures.length > 1 ? `${i + 1}/${structures.length} · ` : ""}
                {structureOption(structureTitle(item), item.path, runOf(item.path))}
              </option>
            ))}
            <option value={OPEN_PDB}>Open PDB entry…</option>
          </select>
        )}
        {kind && (
          <span className="bg-inset text-muted shrink-0 rounded-full px-1.5 py-px text-[10px] whitespace-nowrap" title={STRUCTURE_KINDS[kind].hint}>
            {STRUCTURE_KINDS[kind].label}
          </span>
        )}
        {!pdbForm && (
          <select
            value=""
            onChange={(event) => {
              if (event.target.value === OPEN_PDB) {
                setPdbForm("compare");
                return;
              }
              const other = structures.find(({ item }) => item.path === event.target.value);
              if (other) compareWith(other.item);
            }}
            aria-label="Compare with"
            title="Overlay another structure, superposed by the protein"
            className={`w-[4.6rem] shrink-0 cursor-pointer rounded-md px-1 py-1 text-[11px] outline-none ${comparing ? "bg-inset text-body" : "text-muted hover:text-body hover:bg-inset bg-transparent"}`}
          >
            <option value="" disabled hidden>
              Compare
            </option>
            {structures
              .filter(({ item }) => item.path !== active.path)
              .map(({ item }) => (
                <option key={item.path} value={item.path}>
                  {structureOption(structureTitle(item), item.path, runOf(item.path))}
                </option>
              ))}
            <option value={OPEN_PDB}>PDB entry…</option>
          </select>
        )}
        <a href={active.url} download={active.name} className="text-muted hover:text-body hover:bg-inset shrink-0 rounded-md px-2 py-1 text-[11px]">
          Download
        </a>
        <HeadButton
          label="Display"
          title="Show or hide the Mol* controls"
          pressed={controls}
          onClick={() => execute({ kind: "controls", on: !controls })}
        />
      </header>

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
          <OverlayButton label="Details" title="Contacts and details" pressed={detailsOpen} onClick={() => setDetailsOpen((open) => !open)} />
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

        {comparing && loaded && (
          <CompareLegend
            names={{ primary: structureOption(title, active.path, runOf(active.path)), other: structureOption(structureTitle(comparing), comparing.path, runOf(comparing.path)) }}
            paths={{ primary: active.path, other: comparing.path }}
            result={compared?.result ?? null}
            error={compared?.error ?? null}
            hidden={hidden}
            onToggle={toggleSide}
            onClose={() => setCompare(null)}
          />
        )}

        {pickedHere && (
          <div className="bg-raised border-subtle shadow-card absolute bottom-2 left-2 z-20 flex max-w-[calc(100%-1rem)] flex-wrap items-center gap-1.5 rounded-lg border px-2 py-1.5 text-[11px]">
            <span className="font-mono font-semibold">{pickedShown?.label ?? pickedHere.label}</span>
            {pickedShown?.file != null && <span className="text-faint font-mono text-[10px]">file {pickedShown.file}</span>}
            {pickedHere.chain && <span className="text-faint">chain {pickedHere.chain}</span>}
            {pickedContacts.length > 0 && (
              <span className="flex items-center gap-1">
                {[...new Set(pickedContacts.map((c) => c.type))].map((type) => (
                  <span key={type} className="size-2 rounded-full" title={contactLabel(type)} style={{ background: contactColor(type, theme) }} />
                ))}
                <span className="text-muted">{pickedContacts.length} contact{pickedContacts.length === 1 ? "" : "s"}</span>
              </span>
            )}
            <ChipButton
              onClick={() =>
                pickedHere.resnum !== null &&
                execute({ kind: "residues", residues: [{ chain: pickedHere.chain, resnum: pickedHere.resnum, resname: pickedHere.resname }], mode: "focus" })
              }
            >
              Focus
            </ChipButton>
            <ChipButton
              primary
              onClick={() =>
                onAsk(
                  pickedResidue
                    ? pickQuestion(title, active.path, pickedResidue)
                    : `In ${structureRef(title, active.path)}, what role does ${pickedTitle}${pickedHere.chain ? ` (chain ${pickedHere.chain})` : ""} play in binding the ligand? Use the interaction profile.`,
                )
              }
            >
              Ask the agent about {pickedTitle}
            </ChipButton>
            <button type="button" aria-label="Dismiss" onClick={() => setPicked(null)} className="text-muted hover:text-body cursor-pointer px-1">
              ×
            </button>
          </div>
        )}

        {message && (
          <div
            role="status"
            className={`bg-raised border-subtle shadow-card absolute right-2 left-2 z-30 mx-auto flex max-w-xl items-start gap-2 rounded-lg border px-3 py-1.5 text-[11px] ${pickedHere ? "bottom-12" : "bottom-2"}`}
          >
            <span aria-hidden="true" className="mt-1 size-1.5 shrink-0 rounded-full" style={{ background: message.tone === "ok" ? "var(--color-ok)" : "var(--color-accent)" }} />
            <span className="min-w-0 flex-1">{message.text}</span>
            {message.ask && (
              <ChipButton
                primary
                onClick={() => {
                  onAsk(`About the structure ${structureRef(title, active.path)}: ${message.ask}`);
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
        <div className="flex h-8 shrink-0 items-center gap-2 px-3">
          <button type="button" onClick={() => setDetailsOpen((open) => !open)} className="text-muted hover:text-body cursor-pointer text-[11px] font-semibold">
            {detailsOpen ? "Hide details" : "Show details"}
          </button>
          {profile && !profile.error && (
            <span className="text-faint text-[11px]">
              {profile.totalContacts} contact{profile.totalContacts === 1 ? "" : "s"}
            </span>
          )}
          {detailsOpen && (
            <nav role="tablist" className="ml-auto flex gap-0.5">
              {(["contacts", "details"] as const).map((tab) => (
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
                numbering={numbering}
                smiles={active.smiles ?? profile?.smiles ?? null}
                theme={theme}
                types={types}
                onToggleType={onToggleType}
                onAllTypes={() => setTypes([])}
                query={query}
                onQuery={setQuery}
                selected={selectedHere}
                picked={pickedHere && pickedHere.resnum !== null ? { chain: pickedHere.chain, resnum: pickedHere.resnum } : null}
                flash={flash}
                onSelect={(contact) => {
                  if (!profile) return;
                  setSelected({ path: active.path, index: profile.contacts.indexOf(contact) });
                  const view = viewRef.current;
                  if (view && contact.resnum !== null) focusOn(view, [{ chain: contact.chain, resnum: contact.resnum }]);
                }}
                onProfile={() => onAsk(`Profile the protein-ligand interactions of ${active.path}.`)}
              />
            )}
            {drawerTab === "details" && (
              <DetailsTab
                panel={active}
                title={title}
                residues={fileResidues?.path === active.path ? fileResidues.list : null}
                smiles={active.smiles ?? profile?.smiles ?? null}
                properties={properties ?? []}
                theme={theme}
              />
            )}
          </div>
        )}
      </section>
    </div>
  );
}

// ---------------------------------------------------------------------------

const SIDE_COLORS: Record<Side, string> = { primary: "var(--color-molecule)", other: "var(--color-structure)" };

/** The two structures of an overlay, each a toggle, and how far apart they are. */
function CompareLegend({
  names,
  paths,
  result,
  error,
  hidden,
  onToggle,
  onClose,
}: {
  names: Record<Side, string>;
  paths: Record<Side, string>;
  result: Comparison | null;
  error: string | null;
  hidden: Side[];
  onToggle: (side: Side) => void;
  onClose: () => void;
}) {
  return (
    <div className="bg-raised border-subtle shadow-card absolute top-9 left-2 z-20 max-w-[min(18rem,calc(100%-1rem))] rounded-lg border py-1 pr-1 pl-2 text-[11px]">
      <div className="flex items-start gap-1">
        <div className="min-w-0 flex-1">
          {(["primary", "other"] as const).map((side) => {
            const off = hidden.includes(side);
            return (
              <button
                key={side}
                type="button"
                aria-pressed={!off}
                onClick={() => onToggle(side)}
                title={`${off ? "Show" : "Hide"} ${paths[side]}`}
                className="flex w-full min-w-0 cursor-pointer items-center gap-1.5 py-px text-left"
              >
                <span
                  aria-hidden="true"
                  className="size-2 shrink-0 rounded-full"
                  style={off ? { boxShadow: `inset 0 0 0 1.5px ${SIDE_COLORS[side]}` } : { background: SIDE_COLORS[side] }}
                />
                <span className={`truncate ${off ? "text-faint" : "text-body"}`}>{names[side]}</span>
              </button>
            );
          })}
        </div>
        <button type="button" aria-label="Stop comparing" title="Back to one structure" onClick={onClose} className="text-muted hover:text-body cursor-pointer px-1 leading-none">
          ×
        </button>
      </div>
      <p className="text-faint pt-0.5 text-[10px] tabular-nums">
        {error ? (
          <span style={{ color: "var(--color-accent)" }}>{error}</span>
        ) : !result ? (
          "Superposing…"
        ) : !result.superposed ? (
          "Not superposed: the proteins share too little sequence"
        ) : (
          <>
            {result.ligandRmsd !== null && (
              <span title="Ligand heavy atoms, symmetry-corrected, after superposing the proteins">
                ligand RMSD <span className="text-body font-mono">{result.ligandRmsd.toFixed(2)} Å</span>
                {" · "}
              </span>
            )}
            <span title={`CA atoms of ${result.core} of ${result.aligned} residues aligned by sequence`}>
              CA <span className="font-mono">{result.caRmsd?.toFixed(2)} Å</span>
            </span>
          </>
        )}
      </p>
    </div>
  );
}

function PdbForm({ action, onOpen, onCancel }: { action: string; onOpen: (id: string) => void; onCancel: () => void }) {
  const [value, setValue] = useState("");
  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (value.trim()) onOpen(value.trim());
  };
  return (
    <form onSubmit={submit} className="flex min-w-0 flex-1 items-center gap-1">
      <input
        autoFocus
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={(event) => event.key === "Escape" && onCancel()}
        placeholder="PDB id, e.g. 3MXF"
        aria-label="PDB id"
        maxLength={4}
        spellCheck={false}
        className="bg-sunken border-subtle w-32 rounded-md border px-2 py-0.5 font-mono text-[11px] uppercase outline-none focus:border-[var(--color-accent)]"
      />
      <button type="submit" disabled={!isPdbId(value)} className="text-body hover:bg-inset cursor-pointer rounded-md px-2 py-0.5 text-[11px] font-semibold disabled:cursor-default disabled:opacity-50">
        {action}
      </button>
      <button type="button" onClick={onCancel} className="text-muted hover:text-body cursor-pointer rounded-md px-2 py-0.5 text-[11px]">
        Cancel
      </button>
    </form>
  );
}

function HeadButton({ label, title, onClick, pressed }: { label: string; title: string; onClick: () => void; pressed?: boolean }) {
  return (
    <button
      type="button"
      title={title}
      aria-pressed={pressed}
      onClick={onClick}
      className={`shrink-0 cursor-pointer rounded-md px-2 py-1 text-[11px] ${pressed ? "bg-inset text-body" : "text-muted hover:text-body hover:bg-inset"}`}
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
      className={`bg-raised cursor-pointer rounded px-2 py-0.5 text-[11px] shadow-sm ${pressed ? "text-body" : "text-muted hover:text-body"}`}
      style={pressed ? { boxShadow: "inset 0 -2px 0 var(--color-accent)" } : undefined}
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

function ContactsTab({
  profile,
  numbering,
  smiles,
  theme,
  types,
  onToggleType,
  onAllTypes,
  query,
  onQuery,
  selected,
  picked,
  flash,
  onSelect,
  onProfile,
}: {
  profile: InteractionsPanel | null;
  numbering: Numbering | null;
  smiles: string | null;
  theme: Theme;
  /** Types shown; empty means all. */
  types: string[];
  onToggleType: (type: string) => void;
  onAllTypes: () => void;
  query: string;
  onQuery: (text: string) => void;
  selected: Contact | null;
  /** The residue clicked in 3D; its contacts are highlighted and scrolled to. */
  picked: { chain: string | null; resnum: number } | null;
  /** Residues just clicked ("chain:resnum"), whose rows flash. */
  flash: { keys: string[]; at: number } | null;
  onSelect: (contact: Contact) => void;
  onProfile: () => void;
}) {
  const listRef = useRef<HTMLUListElement>(null);
  const pickedKey = picked ? `${picked.chain}:${picked.resnum}` : null;
  useEffect(() => {
    if (!pickedKey) return;
    listRef.current?.querySelector('[data-picked="true"]')?.scrollIntoView({ block: "nearest" });
  }, [pickedKey, selected]);

  useEffect(() => {
    const list = listRef.current;
    if (!flash || !list) return;
    const rows = flash.keys.flatMap((key) => [...list.querySelectorAll<HTMLElement>(`[data-residue="${CSS.escape(key)}"] > button`)]);
    rows[0]?.scrollIntoView({ block: "nearest" });
    for (const row of rows) {
      row.animate([{ backgroundColor: "color-mix(in oklch, var(--color-structure) 28%, transparent)" }, { backgroundColor: "transparent" }], {
        duration: 1400,
        easing: "ease-out",
      });
    }
  }, [flash]);

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
  if (profile.error) {
    const error = profileError(profile.error);
    return (
      <p className={`py-4 text-center text-xs ${error.quiet ? "text-muted" : ""}`} style={error.quiet ? undefined : { color: "var(--color-accent)" }}>
        {error.text}
      </p>
    );
  }

  const needle = query.trim().toLowerCase();
  const rows = profile.contacts.filter((contact) => {
    if (!matchesTypes(contact, types)) return false;
    if (!needle) return true;
    const shown = contact.resnum !== null ? `${contact.resname}${displayNumber(numbering, contact.chain, contact.resnum)}` : "";
    return [contact.type, contactLabel(contact.type), contact.label, shown, contact.fragment, contact.ligandAtom, contact.receptorAtom].join(" ").toLowerCase().includes(needle);
  });
  const highlight = selected?.ligandAtomIndex != null && selected.ligandAtomIndex >= 0 ? [selected.ligandAtomIndex] : undefined;
  const selectedName = selected?.resnum != null ? residueLabel(numbering, selected.resname, selected.chain, selected.resnum).label : selected?.label;

  return (
    <div className="flex gap-3">
      <div className="min-w-0 flex-1">
        <div className="mb-2 flex flex-wrap items-center gap-1">
          <input
            type="search"
            value={query}
            onChange={(event) => onQuery(event.target.value)}
            placeholder="Filter contacts"
            aria-label="Filter contacts"
            className="bg-sunken border-subtle mr-1 w-28 rounded border px-2 py-0.5 text-[11px] outline-none focus:border-[var(--color-accent)]"
          />
          <div role="group" aria-label="Contact types" className="flex flex-wrap items-center gap-1">
            {sortedTypes(profile.counts).map((type) => {
              const on = types.includes(type);
              const dimmed = types.length > 0 && !on;
              return (
                <button
                  key={type}
                  type="button"
                  aria-pressed={on}
                  onClick={() => onToggleType(type)}
                  title={on ? `Stop showing ${contactLabel(type).toLowerCase()} contacts` : types.length ? `Also show ${contactLabel(type).toLowerCase()} contacts` : `Show only ${contactLabel(type).toLowerCase()} contacts`}
                  className={`flex cursor-pointer items-center gap-1 rounded-full px-1.5 py-px text-[10px] transition-opacity ${on ? "text-body" : "text-muted hover:text-body"} ${dimmed ? "opacity-50 hover:opacity-100" : ""}`}
                  style={on ? { background: `color-mix(in oklch, ${contactColor(type, theme)} 18%, transparent)`, boxShadow: `inset 0 0 0 1px ${contactColor(type, theme)}` } : undefined}
                >
                  <span className="size-1.5 rounded-full" style={{ background: contactColor(type, theme) }} />
                  {contactLabel(type)}
                  <b className="font-semibold">{profile.counts[type]}</b>
                </button>
              );
            })}
            {types.length > 0 && (
              <button type="button" onClick={onAllTypes} className="text-faint hover:text-body cursor-pointer px-1 text-[10px]">
                Show all
              </button>
            )}
          </div>
        </div>
        <ul ref={listRef} className="space-y-px">
          {rows.map((contact, index) => {
            const isSelected = selected === contact;
            const isPicked = picked !== null && sameResidue(picked, contact);
            const name = contact.resnum !== null ? residueLabel(numbering, contact.resname, contact.chain, contact.resnum) : { label: contact.label, file: null };
            return (
              <li key={index} data-picked={isPicked || undefined} data-residue={residueKey(contact.chain, contact.resnum)}>
                <button
                  type="button"
                  onClick={() => onSelect(contact)}
                  aria-current={isSelected || undefined}
                  className={`grid w-full cursor-pointer grid-cols-[minmax(7rem,1fr)_minmax(5rem,auto)_minmax(6rem,1.3fr)] items-center gap-2 rounded px-1.5 py-1 text-left text-[11px] ${
                    isSelected || isPicked ? "" : "hover:bg-sunken"
                  }`}
                  style={
                    isSelected || isPicked
                      ? {
                          background: `color-mix(in oklch, var(--color-structure) ${isSelected ? 12 : 6}%, transparent)`,
                          boxShadow: isSelected ? "inset 2px 0 0 var(--color-structure)" : undefined,
                        }
                      : undefined
                  }
                >
                  <span className="flex min-w-0 items-center gap-1.5 truncate">
                    <span className="size-1.5 shrink-0 rounded-full" style={{ background: contactColor(contact.type, theme) }} />
                    {contactLabel(contact.type)}
                  </span>
                  <span className="font-mono whitespace-nowrap">
                    <strong>{name.label}</strong>
                    {contact.chain && <span className="text-faint">.{contact.chain}</span>}
                    {name.file !== null && (
                      <span className="text-faint ml-1 text-[10px]" title="Number in the structure file">
                        {name.file}
                      </span>
                    )}
                  </span>
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
          <div className="bg-sunken rounded-lg p-1">
            <MolDepiction smiles={smiles} theme={theme} width={200} height={160} atoms={highlight} className="h-28 w-full" />
          </div>
          {selected && <p className="text-faint mt-1 text-center text-[10px]">{`${selected.ligandAtom ?? "ligand atom"} → ${selectedName}`}</p>}
        </div>
      )}
    </div>
  );
}

/** The structure's files, then pose quality, affinity and the ligand's predictions next to its properties, and the ligand in 2D. */
function DetailsTab({
  panel,
  title,
  residues,
  smiles,
  properties,
  theme,
}: {
  panel: StructurePanel;
  title: string;
  residues: FileResidue[] | null;
  smiles: string | null;
  properties: Sourced<PropertiesPanel>[];
  theme: Theme;
}) {
  const { rdkit } = useRDKit(Boolean(smiles));
  const boltz = useBoltzFiles(panel);
  const cofold = panel.metrics ?? boltz.metrics;
  const files = structureFiles(panel, title, residues, smiles, boltz);
  let predictions: FactGroup[] = [];
  let ligand: Fact[] = [];
  if (smiles) {
    predictions = predictionGroups(smiles, properties);
    if (!predictions.length && rdkit) {
      const key = (value: string) => canonical(rdkit, value) ?? value;
      const canonicalPanels = properties.map((entry) => ({
        ...entry,
        item: {
          ...entry.item,
          molecules: entry.item.molecules.map((m) => ({ ...m, smiles: m.smiles && key(m.smiles), standardized: m.standardized && key(m.standardized) })),
        },
      }));
      predictions = predictionGroups(key(smiles), canonicalPanels);
    }
    ligand = propertyFacts({}, rdkit ? descriptors(rdkit, smiles) : null);
  }

  return (
    <div className="@container flex gap-3">
      <div className="min-w-0 flex-1">
        <DetailsColumns files={files} cofold={cofold ? cofoldBars(cofold) : []} predictions={predictionBars(predictions)} properties={propertyBars(ligand)} />
      </div>
      {smiles && (
        <div className="bg-sunken hidden w-36 shrink-0 self-start rounded-lg p-1 @3xl:block">
          <MolDepiction smiles={smiles} theme={theme} width={200} height={160} className="h-28 w-full" />
        </div>
      )}
    </div>
  );
}

interface BoltzFiles {
  files: RelatedFile[];
  input: BoltzInput | null;
  /** Confidence and affinity read from its JSON files, for a prediction that came without them. */
  metrics?: StructureMetrics;
}

/** The Boltz files next to *panel* that exist, its input YAML read for the MSA, templates and sequences, and its metrics. */
function useBoltzFiles(panel: StructurePanel): BoltzFiles {
  const [found, setFound] = useState<(BoltzFiles & { path: string }) | null>(null);
  useEffect(() => {
    const boltz = boltzOutput(panel.path);
    if (!boltz) return;
    const controller = new AbortController();
    const read = (file: RelatedFile) =>
      fetch(file.url, { signal: controller.signal })
        .then((response) => (response.ok ? response.text() : null))
        .catch(() => null);
    void Promise.all([Promise.all(boltz.candidates.map(read)), read(boltz.input)]).then(([texts, input]) => {
      if (controller.signal.aborted) return;
      const files = boltz.candidates.filter((_, i) => texts[i] !== null);
      if (input !== null) files.push(boltz.input);
      const values: Record<string, number> = {};
      boltz.candidates.forEach((file, i) => {
        const text = texts[i];
        if (file.type !== "JSON" || text === null) return;
        try {
          for (const [name, value] of Object.entries(JSON.parse(text) as Record<string, unknown>)) if (typeof value === "number") values[name] = value;
        } catch {
          // Not JSON after all: a gateway error page.
        }
      });
      setFound({ path: panel.path, files, input: input !== null ? parseBoltzInput(input) : null, metrics: metricsFromValues(values) });
    });
    return () => controller.abort();
  }, [panel.path]);
  return found && found.path === panel.path ? found : { files: [], input: null };
}

/** Download links: the structure, its ligand, sequence, MSA and the files the prediction wrote. */
function structureFiles(panel: StructurePanel, title: string, residues: FileResidue[] | null, smiles: string | null, boltz: BoltzFiles): RelatedFile[] {
  const stem = panel.name.replace(/\.(cif|mmcif|pdb)$/i, "");
  const sdf = boltz.files.find((file) => file.type === "SDF");
  const proteins = boltz.input?.proteins.length ? boltz.input.proteins : residues ? chainSequences(residues) : [];
  return [
    { label: "Structure", type: panel.format.toUpperCase(), url: panel.url, name: panel.name },
    sdf ?? (smiles ? smilesFile(title, smiles) : null),
    fastaFile(stem, proteins),
    ...(boltz.input ? inputFiles(boltz.input) : []),
    ...boltz.files.filter((file) => file !== sdf),
  ].filter((file): file is RelatedFile => file !== null);
}
