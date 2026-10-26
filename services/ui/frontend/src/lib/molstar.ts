// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// The prebuilt viewer bundle and its stylesheet, fetched on first use: Mol*
// is 5 MB and only needed once there is a structure to look at.
import scriptUrl from "molstar/build/viewer/molstar.js?url";
import styleUrl from "molstar/build/viewer/molstar.css?url";

/*
 * The small part of the Mol* viewer API used here. The bundle is a classic
 * script that defines window.molstar, so these types describe it rather than
 * import it.
 */
type Expression = unknown;
interface QueryBuilder {
  struct: {
    generator: { atomGroups: (params: Record<string, Expression>) => Expression };
    modifier: { includeSurroundings: (params: Record<string | number, unknown>) => Expression };
  };
  core: { rel: { eq: (args: unknown[]) => Expression } };
  /** Macromolecular atom property, e.g. "entityType". */
  ammp: (property: string) => Expression;
}
interface SchemaItem {
  auth_asym_id?: string;
  auth_seq_id?: number;
  auth_atom_id?: string;
  atom_index?: number;
}
interface Observable<T> {
  subscribe: (next: (value: T) => void) => { unsubscribe: () => void };
}
interface StructureRef {
  cell: { obj?: { data?: unknown }; transform: { ref: string } };
  components: unknown[];
}
type Loci = unknown;
interface StateBuilderNode {
  ref: string;
  apply: (transformer: unknown, params?: unknown, options?: { tags?: string; dependsOn?: string[] }) => StateBuilderNode;
}
interface StateBuilder {
  toRoot: () => StateBuilderNode;
  delete: (ref: string) => StateBuilder;
  commit: (options?: { doNotLogTiming?: boolean }) => Promise<unknown>;
}
interface Plugin {
  canvas3d?: { setProps: (props: Record<string, unknown>) => void; requestCameraReset?: (options?: object) => void };
  state: {
    data: { build: () => StateBuilder };
    updateBehavior: (behavior: { id: string }, params: (old: Record<string, unknown>) => Record<string, unknown>) => Promise<unknown>;
  };
  managers: {
    camera: {
      reset: (snapshot?: unknown, durationMs?: number) => void;
      focusLoci: (loci: Loci, options?: { extraRadius?: number; durationMs?: number }) => void;
    };
    interactivity: {
      lociSelects: { deselectAll: () => void; selectOnly: (loci: { loci: Loci }) => void };
      lociHighlights: { clearHighlights: () => void; highlight: (loci: { loci: Loci }) => void };
    };
    structure: {
      /** Shows the target and its surroundings as sticks. */
      focus: { setFromLoci: (loci: Loci) => void; clear: () => void };
      hierarchy: { current: { structures: StructureRef[] } };
      component: {
        clear: (structures: StructureRef[]) => Promise<void>;
        updateRepresentationsTheme: (components: unknown[], params: { color: string }) => Promise<void>;
      };
    };
  };
  builders: { structure: { representation: { applyPreset: (parent: unknown, preset: string) => Promise<unknown> } } };
  behaviors: {
    interaction: { click: Observable<{ current: { loci: ClickLoci } }> };
    labels?: { highlight: { value: { labels: ReadonlyArray<string> } } };
  };
  helpers: { viewportScreenshot?: { download: (name?: string) => Promise<void> | void } };
  config: { set: (key: unknown, value: unknown) => void };
  layout: { setProps: (props: Record<string, unknown>) => void };
  clear: () => Promise<void> | void;
}
interface MolstarViewer {
  plugin: Plugin;
  loadStructureFromUrl: (url: string, format: string, isBinary?: boolean, options?: { label?: string }) => Promise<void>;
  handleResize: () => void;
  dispose: () => void;
}
type Location = unknown;
interface BondUnit {
  elements: ArrayLike<number>;
}
interface ClickLoci {
  kind: string;
  structure?: unknown;
  bonds?: { aUnit: BondUnit; aIndex: number }[];
}
type Property<T> = (location: Location) => T;
interface StructureLib {
  Structure: { eachAtomicHierarchyElement: (structure: unknown, visit: { residue?: (location: Location) => void }) => void };
  StructureElement: {
    Loci: {
      is: (loci: unknown) => boolean;
      isEmpty: (loci: unknown) => boolean;
      getFirstLocation: (loci: unknown) => Location | undefined;
      forEachLocation: (loci: unknown, visit: (location: Location) => void) => void;
      fromExpression: (structure: unknown, expression: (Q: QueryBuilder) => Expression) => Loci;
      fromSchema: (structure: unknown, schema: { items: SchemaItem[] }) => Loci;
    };
    Location: { create: (structure: unknown, unit: BondUnit, element: number) => Location };
    Bundle: { fromLoci: (loci: Loci) => unknown };
  };
  StructureProperties: {
    atom: { auth_atom_id: Property<string>; sourceIndex: Property<number> };
    residue: { auth_comp_id: Property<string>; auth_seq_id: Property<number> };
    chain: { auth_asym_id: Property<string> };
    entity: { type: Property<string> };
  };
}
interface PluginLib {
  PluginConfig: { Viewport: { ShowIllumination: unknown; ShowXR: unknown } };
  StateTransforms: {
    Misc: { CreateGroup: unknown };
    Model: { MultiStructureSelectionFromBundle: unknown };
    Representation: { StructureSelectionsDistance3D: unknown };
  };
}
interface MolstarGlobal {
  Viewer: { create: (element: HTMLElement, options: Record<string, unknown>) => Promise<MolstarViewer> };
  lib?: { structure?: StructureLib; plugin?: PluginLib };
}

declare global {
  interface Window {
    molstar?: MolstarGlobal;
  }
}

let loading: Promise<MolstarGlobal> | null = null;

export function loadMolstar(): Promise<MolstarGlobal> {
  if (loading) return loading;
  loading = new Promise<MolstarGlobal>((resolve, reject) => {
    if (window.molstar) {
      resolve(window.molstar);
      return;
    }
    if (!document.querySelector(`link[href="${styleUrl}"]`)) {
      const link = document.createElement("link");
      link.rel = "stylesheet";
      link.href = styleUrl;
      document.head.appendChild(link);
    }
    const script = document.createElement("script");
    script.src = scriptUrl;
    script.async = true;
    script.onload = () => (window.molstar ? resolve(window.molstar) : reject(new Error("Mol* did not initialise")));
    script.onerror = () => reject(new Error("Could not load Mol*"));
    document.head.appendChild(script);
  });
  loading.catch(() => {
    loading = null;
  });
  return loading;
}

/** Display styles, each a Mol* representation preset. */
export const VIEW_STYLES = {
  cartoon: { preset: "polymer-and-ligand", label: "Cartoon", hint: "Protein cartoon, ligand and nearby residues" },
  surface: { preset: "molecular-surface", label: "Surface", hint: "Molecular surface of the pocket" },
  sticks: { preset: "atomic-detail", label: "Atoms", hint: "Every atom as ball and stick" },
  illustrative: { preset: "illustrative", label: "Illustrative", hint: "Flat, outlined illustration" },
} as const;
export type ViewStyleName = keyof typeof VIEW_STYLES;

/** Colour schemes, each a Mol* colour theme. */
export const COLOR_THEMES = {
  chain: "chain-id",
  element: "element-symbol",
  residue: "residue-name",
  hydrophobicity: "hydrophobicity",
  confidence: "uncertainty",
  secondary: "secondary-structure",
} as const;
export type ColorName = keyof typeof COLOR_THEMES;

export const BACKGROUNDS = { dark: 0x0b0d12, black: 0x000000, light: 0xf4f5f7, white: 0xffffff } as const;
export type BackgroundName = keyof typeof BACKGROUNDS;

export interface Residue {
  chain?: string | null;
  resnum: number;
  /** Three-letter code when the reference named it ("ASN140"); tells file and UniProt numbering apart. */
  resname?: string | null;
}

/** One protein-ligand contact drawn as a dashed line. */
export interface ContactLine {
  chain: string | null;
  resnum: number;
  /** Atom name, or several for a ring ("CG CD1 CD2"); the whole residue when null. */
  receptorAtom: string | null;
  ligandAtom: string | null;
  /** 0-based among the ligand's atoms, used when the name is not found. */
  ligandAtomIndex: number | null;
  /** "#rrggbb". */
  color: string;
  /** Specific interactions get a thicker line than hydrophobic contacts. */
  strong: boolean;
  /** Text at the middle of the line, or null for none. */
  label: string | null;
}

/** A residue picked by clicking in the viewport. */
export interface Picked {
  label: string;
  resname: string | null;
  resnum: number | null;
  chain: string | null;
}

const ligand = (Q: QueryBuilder) =>
  Q.struct.generator.atomGroups({ "entity-test": Q.core.rel.eq([Q.ammp("entityType"), "non-polymer"]) });

const FOCUS_BEHAVIOR = "ms-plugin.create-structure-focus-representation";
const hexColor = (color: string) => Number.parseInt(color.replace("#", ""), 16) || 0x888888;
const atomNames = (text: string | null) => (text ?? "").split(/[\s,;+/]+/).filter(Boolean);

/**
 * One Mol* viewport. Wraps the few viewer calls the UI needs, so a Mol*
 * upgrade touches this class and nothing else.
 */
export class StructureView {
  private viewer: MolstarViewer | null = null;
  private disposed = false;
  private loadToken = 0;
  private spinning = false;
  private subscriptions: { unsubscribe: () => void }[] = [];
  /** The state node holding the contact dashes, while there are any. */
  private contactsRef: string | null = null;
  /** Contact redraws run one after another, so a quick second focus does not leave the first one's dashes. */
  private contactsQueue: Promise<void> = Promise.resolve();
  private background = 0x000000;
  style: ViewStyleName = "cartoon";

  static async create(host: HTMLElement, background: number): Promise<StructureView> {
    const molstar = await loadMolstar();
    const view = new StructureView();
    view.background = background;
    view.viewer = await molstar.Viewer.create(host, {
      layoutIsExpanded: false,
      layoutShowControls: false,
      layoutShowSequence: false,
      layoutShowLog: false,
      layoutShowLeftPanel: false,
      layoutShowRemoteState: false,
      collapseLeftPanel: true,
      collapseRightPanel: true,
      viewportShowControls: false,
      viewportShowReset: false,
      viewportShowExpand: false,
      viewportShowScreenshotControls: true,
      viewportShowSettings: true,
      viewportShowSelectionMode: false,
      viewportShowAnimation: false,
      viewportShowTrajectoryControls: false,
      viewportShowToggleFullscreen: false,
      viewportBackgroundColor: `#${background.toString(16).padStart(6, "0")}`,
      pdbProvider: "rcsb",
    });
    // Focus shows the target and its neighbours as sticks; the contacts drawn
    // are the profiler's (setContacts), not Mol*'s own, which include
    // protein-protein ones.
    const plugin = view.viewer.plugin;
    await plugin.state.updateBehavior({ id: FOCUS_BEHAVIOR }, (old) => ({ ...old, components: ["target", "surroundings"] })).catch(() => undefined);
    // The viewer options cannot hide these two; the screenshot and settings buttons stay.
    const config = molstar.lib?.plugin?.PluginConfig.Viewport;
    if (config) {
      plugin.config.set(config.ShowIllumination, false);
      plugin.config.set(config.ShowXR, "never");
      plugin.layout.setProps({});
    }
    return view;
  }

  private get plugin(): Plugin | null {
    return this.disposed ? null : (this.viewer?.plugin ?? null);
  }

  private get structureRef(): StructureRef | null {
    return this.plugin?.managers.structure.hierarchy.current.structures[0] ?? null;
  }

  /** Load a file, replacing what is shown. False when a newer load won. */
  async load(url: string, format: "cif" | "pdb", label: string): Promise<boolean> {
    const token = ++this.loadToken;
    const viewer = this.viewer;
    if (!viewer) return false;
    await viewer.plugin.clear();
    this.contactsRef = null;
    if (token !== this.loadToken || this.disposed) return false;
    await viewer.loadStructureFromUrl(url, format === "cif" ? "mmcif" : "pdb", false, { label });
    if (token !== this.loadToken || this.disposed) return false;
    if (this.style !== "cartoon") await this.setStyle(this.style);
    return true;
  }

  fit(): void {
    this.plugin?.managers.camera.reset(undefined, 400);
  }

  /** The atoms an expression or a residue list picks in the loaded structure, or null if none. */
  private lociOf(query: { expression: (Q: QueryBuilder) => Expression } | { items: SchemaItem[] }): Loci | null {
    const lib = window.molstar?.lib?.structure;
    const structure = this.structureRef?.cell.obj?.data;
    if (!lib || !structure) return null;
    const loci =
      "expression" in query ? lib.StructureElement.Loci.fromExpression(structure, query.expression) : lib.StructureElement.Loci.fromSchema(structure, query);
    return lib.StructureElement.Loci.isEmpty(loci) ? null : loci;
  }

  /** The polymer residues of the loaded structure, in file order, for renumbering. */
  residues(): { chain: string | null; resnum: number; resname: string }[] {
    const lib = window.molstar?.lib?.structure;
    const structure = this.structureRef?.cell.obj?.data;
    if (!lib || !structure) return [];
    const seen = new Set<string>();
    const out: { chain: string | null; resnum: number; resname: string }[] = [];
    const P = lib.StructureProperties;
    lib.Structure.eachAtomicHierarchyElement(structure, {
      residue: (location) => {
        if (P.entity.type(location) !== "polymer") return;
        const chain = P.chain.auth_asym_id(location) || null;
        const resnum = P.residue.auth_seq_id(location);
        const id = `${chain}:${resnum}`;
        if (seen.has(id)) return;
        seen.add(id);
        out.push({ chain, resnum, resname: P.residue.auth_comp_id(location) });
      },
    });
    return out;
  }

  /**
   * Mol*'s focus: the target and every residue within 5 Å as sticks; then
   * the camera frames it. A camera move alone shows a cartoon.
   */
  private focus(loci: Loci, extraRadius: number): void {
    const plugin = this.plugin;
    if (!plugin) return;
    plugin.managers.interactivity.lociSelects.deselectAll();
    plugin.managers.structure.focus.setFromLoci(loci);
    plugin.managers.camera.focusLoci(loci, { extraRadius, durationMs: 400 });
  }

  /** Replace the contact dashes with *lines*; an empty list removes them. */
  setContacts(lines: ContactLine[]): Promise<void> {
    this.contactsQueue = this.contactsQueue.then(() => this.drawContacts(lines)).catch(() => undefined);
    return this.contactsQueue;
  }

  private async drawContacts(lines: ContactLine[]): Promise<void> {
    const plugin = this.plugin;
    if (!plugin) return;
    if (this.contactsRef) {
      const ref = this.contactsRef;
      this.contactsRef = null;
      await plugin.state.data.build().delete(ref).commit({ doNotLogTiming: true });
    }
    const lib = window.molstar?.lib;
    const cell = this.structureRef?.cell;
    if (!lines.length || !lib?.structure || !lib.plugin || !cell?.obj?.data) return;
    const { StructureElement, StructureProperties: P } = lib.structure;
    const T = lib.plugin.StateTransforms;

    const ligandAtoms: { name: string; index: number }[] = [];
    const ligandLoci = this.lociOf({ expression: ligand });
    if (ligandLoci) {
      StructureElement.Loci.forEachLocation(ligandLoci, (location) =>
        ligandAtoms.push({ name: P.atom.auth_atom_id(location), index: P.atom.sourceIndex(location) }),
      );
    }
    const ligandSide = (line: ContactLine): Loci | null => {
      const names = atomNames(line.ligandAtom);
      const byName = names.map((name) => ligandAtoms.find((atom) => atom.name === name)).filter((atom) => atom !== undefined);
      const atoms =
        byName.length > 0
          ? byName
          : line.ligandAtomIndex !== null && line.ligandAtomIndex >= 0 && ligandAtoms[line.ligandAtomIndex]
            ? [ligandAtoms[line.ligandAtomIndex]]
            : [];
      return atoms.length ? this.lociOf({ items: atoms.map((atom) => ({ atom_index: atom.index })) }) : null;
    };
    const receptorSide = (line: ContactLine): Loci | null => {
      const residue: SchemaItem = line.chain ? { auth_asym_id: line.chain, auth_seq_id: line.resnum } : { auth_seq_id: line.resnum };
      const names = atomNames(line.receptorAtom);
      return (names.length ? this.lociOf({ items: names.map((name) => ({ ...residue, auth_atom_id: name })) }) : null) ?? this.lociOf({ items: [residue] });
    };

    const builder = plugin.state.data.build();
    const group = builder.toRoot().apply(T.Misc.CreateGroup, { label: "Contacts" });
    const parent = cell.transform.ref;
    let drawn = 0;
    for (const line of lines) {
      const a = receptorSide(line);
      const b = ligandSide(line);
      if (!a || !b) continue;
      const color = hexColor(line.color);
      group
        .apply(
          T.Model.MultiStructureSelectionFromBundle,
          {
            selections: [
              { key: "a", groupId: "a", ref: parent, bundle: StructureElement.Bundle.fromLoci(a) },
              { key: "b", groupId: "b", ref: parent, bundle: StructureElement.Bundle.fromLoci(b) },
            ],
            isTransitive: true,
            label: "Contact",
          },
          { dependsOn: [parent] },
        )
        .apply(T.Representation.StructureSelectionsDistance3D, {
          unitLabel: "Å",
          linesColor: color,
          linesSize: line.strong ? 0.07 : 0.04,
          dashLength: 0.12,
          alpha: line.strong ? 1 : 0.8,
          visuals: line.label ? ["lines", "text"] : ["lines"],
          customText: line.label ?? "",
          textColor: color,
          textSize: 0.32,
          borderWidth: 0.25,
          borderColor: this.background,
        });
      drawn++;
    }
    if (!drawn) return;
    await builder.commit({ doNotLogTiming: true });
    this.contactsRef = group.ref;
  }

  /** A soft glow on residues, to show which ones a click picked. */
  markResidues(residues: Residue[]): void {
    const items = residues.map((r) => (r.chain ? { auth_asym_id: r.chain, auth_seq_id: r.resnum } : { auth_seq_id: r.resnum }));
    const loci = items.length ? this.lociOf({ items }) : null;
    if (loci) this.plugin?.managers.interactivity.lociSelects.selectOnly({ loci });
  }

  /** False when the structure has no ligand. */
  focusLigand(): boolean {
    const loci = this.lociOf({ expression: ligand });
    if (loci) this.focus(loci, 4);
    return loci !== null;
  }

  /** The ligand's contacts with a wider frame, so the whole pocket is in view. */
  focusPocket(radius = 5): boolean {
    const loci = this.lociOf({ expression: ligand });
    if (loci) this.focus(loci, radius + 4);
    return loci !== null;
  }

  /** Focus residues with their neighbours, or only light them up. False when none is in the structure. */
  focusResidues(residues: Residue[], action: "focus" | "highlight" = "focus"): boolean {
    if (!residues.length) return false;
    const items = residues.map((r) => (r.chain ? { auth_asym_id: r.chain, auth_seq_id: r.resnum } : { auth_seq_id: r.resnum }));
    const loci = this.lociOf({ items });
    if (!loci) return false;
    if (action === "focus") this.focus(loci, 3);
    this.plugin?.managers.interactivity.lociSelects.selectOnly({ loci });
    return true;
  }

  clearSelection(): void {
    const plugin = this.plugin;
    if (!plugin) return;
    plugin.managers.structure.focus.clear();
    plugin.managers.interactivity.lociSelects.deselectAll();
    plugin.managers.interactivity.lociHighlights.clearHighlights();
    void this.setContacts([]);
  }

  async setStyle(style: ViewStyleName): Promise<void> {
    const plugin = this.plugin;
    this.style = style;
    if (!plugin) return;
    const structures = plugin.managers.structure.hierarchy.current.structures;
    await plugin.managers.structure.component.clear(structures);
    for (const structure of structures) {
      await plugin.builders.structure.representation.applyPreset(structure.cell, VIEW_STYLES[style].preset);
    }
  }

  async setColor(color: ColorName): Promise<void> {
    const plugin = this.plugin;
    if (!plugin) return;
    for (const structure of plugin.managers.structure.hierarchy.current.structures) {
      await plugin.managers.structure.component.updateRepresentationsTheme(structure.components, { color: COLOR_THEMES[color] });
    }
  }

  setBackground(color: number): void {
    this.background = color;
    this.plugin?.canvas3d?.setProps({ renderer: { backgroundColor: color } });
  }

  /** Toggle a slow turn around the vertical axis. Returns the new state. */
  spin(on = !this.spinning): boolean {
    this.spinning = on;
    this.plugin?.canvas3d?.setProps({
      trackball: { animate: on ? { name: "spin", params: { speed: 0.6 } } : { name: "off", params: {} } },
    });
    return on;
  }

  showControls(on: boolean): void {
    this.plugin?.layout.setProps({ showControls: on, regionState: { left: on ? "full" : "hidden", right: "hidden", top: "hidden", bottom: "hidden" } });
  }

  async screenshot(name: string): Promise<boolean> {
    const helper = this.plugin?.helpers.viewportScreenshot;
    if (!helper) return false;
    await helper.download(name);
    return true;
  }

  /** Report the residue the user clicks, or null for a click on nothing. */
  onPick(listener: (picked: Picked | null) => void): void {
    const plugin = this.plugin;
    if (!plugin) return;
    const lib = window.molstar?.lib?.structure;
    this.subscriptions.push(
      plugin.behaviors.interaction.click.subscribe((event) => {
        const loci = event.current.loci;
        if (loci.kind === "empty-loci") {
          listener(null);
          return;
        }
        let location: Location | undefined;
        if (lib?.StructureElement.Loci.is(loci)) {
          location = lib.StructureElement.Loci.getFirstLocation(loci);
        } else if (lib && loci.kind === "bond-loci" && loci.bonds?.length) {
          // A click on a stick: the residue of the bond's first atom.
          const bond = loci.bonds[0];
          location = lib.StructureElement.Location.create(loci.structure, bond.aUnit, bond.aUnit.elements[bond.aIndex]);
        }
        if (lib && location) {
          const resname = lib.StructureProperties.residue.auth_comp_id(location);
          const resnum = lib.StructureProperties.residue.auth_seq_id(location);
          const chain = lib.StructureProperties.chain.auth_asym_id(location) || null;
          listener({ label: `${resname}${resnum}`, resname, resnum, chain });
          return;
        }
        // Older bundles without lib: read the hover label ("ASN 140 | A ...").
        const text = (plugin.behaviors.labels?.highlight.value.labels[0] ?? "").replace(/<[^>]+>/g, " ");
        listener(parsePickLabel(text));
      }),
    );
  }

  resize(): void {
    try {
      this.viewer?.handleResize();
    } catch {
      // A hidden tab has no size; the next resize fixes it.
    }
  }

  dispose(): void {
    this.disposed = true;
    this.subscriptions.forEach((s) => s.unsubscribe());
    try {
      this.viewer?.dispose();
    } catch {
      // Already gone with its WebGL context.
    }
    this.viewer = null;
  }
}

/** Parse a Mol* element label such as "ASN 140 | A" or "A | ASN 140 | ND2". */
export function parsePickLabel(text: string): Picked | null {
  const clean = text.replace(/\s+/g, " ").trim();
  if (!clean) return null;
  const residue = /\b([A-Z]{3}) (-?\d+)\b/.exec(clean);
  const parts = clean.split("|").map((part) => part.trim());
  const chain = parts.find((part) => /^[A-Za-z0-9]{1,2}$/.test(part)) ?? null;
  if (!residue) return { label: clean.slice(0, 80), resname: null, resnum: null, chain };
  return { label: `${residue[1]}${residue[2]}`, resname: residue[1], resnum: Number(residue[2]), chain };
}

/** Residue names Boltz and common pipelines give a docked or cofolded ligand. */
const LIGAND_RESNAMES = new Set(["LIG", "LIG1", "UNL", "UNK", "MOL", "INH", "DRG"]);

/** How to name a picked residue in prose: "the ligand" rather than "LIG1". */
export function pickedName(picked: Picked): string {
  return picked.resname && LIGAND_RESNAMES.has(picked.resname.toUpperCase()) ? "the ligand" : picked.label;
}
