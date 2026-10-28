// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { JSMol, RDKitModule } from "@rdkit/rdkit";
// Both files are emitted as assets and fetched on first use, so the 7 MB
// WebAssembly binary never delays the first paint.
import scriptUrl from "@rdkit/rdkit/dist/RDKit_minimal.js?url";
import wasmUrl from "@rdkit/rdkit/dist/RDKit_minimal.wasm?url";

let loading: Promise<RDKitModule> | null = null;
let loaded: RDKitModule | null = null;

/** The module if it has already loaded, without starting a download. */
export function loadedRDKit(): RDKitModule | null {
  return loaded;
}

/**
 * The RDKit module, loaded once for the whole app.
 *
 * RDKit_minimal.js is a classic script that defines window.initRDKitModule,
 * not an ES module, so it is added as a script tag rather than imported.
 */
export function loadRDKit(): Promise<RDKitModule> {
  if (loading) return loading;
  loading = new Promise<RDKitModule>((resolve, reject) => {
    const start = () =>
      window
        .initRDKitModule({ locateFile: () => wasmUrl })
        .then((module) => {
          module.prefer_coordgen(true);
          loaded = module;
          resolve(module);
        }, reject);
    if (typeof window.initRDKitModule === "function") {
      start();
      return;
    }
    const script = document.createElement("script");
    script.src = scriptUrl;
    script.async = true;
    script.onload = start;
    script.onerror = () => reject(new Error("Could not load RDKit"));
    document.head.appendChild(script);
  });
  // A failed load can be retried, e.g. after the network comes back.
  loading.catch(() => {
    loading = null;
  });
  return loading;
}

/** A map that forgets its least recently used entry beyond *size* entries. */
class Lru<V> {
  private readonly entries = new Map<string, V>();
  private readonly size: number;

  constructor(size: number) {
    this.size = size;
  }

  get(key: string, make: () => V): V {
    if (this.entries.has(key)) {
      const value = this.entries.get(key) as V;
      this.entries.delete(key);
      this.entries.set(key, value);
      return value;
    }
    const value = make();
    this.entries.set(key, value);
    if (this.entries.size > this.size) this.entries.delete(this.entries.keys().next().value as string);
    return value;
  }
}

/** Drawings are 5 to 20 kB of SVG each, enough for the molecules on screen in every panel. */
const drawings = new Lru<string | null>(200);
const computed = new Lru<Descriptors | null>(2000);

/** Run *read* on a parsed molecule and free it afterwards. Null for invalid SMILES. */
export function withMol<T>(rdkit: RDKitModule, smiles: string, read: (mol: JSMol) => T): T | null {
  let mol: JSMol | null = null;
  try {
    mol = rdkit.get_mol(smiles);
    if (!mol || !mol.is_valid()) return null;
    return read(mol);
  } catch {
    return null;
  } finally {
    mol?.delete();
  }
}

export interface DepictOptions {
  width: number;
  height: number;
  dark: boolean;
  /** Atom indices to highlight. */
  atoms?: number[];
}

/**
 * Draw a molecule as SVG markup.
 *
 * The palette is set per theme: RDKit draws on white by default, which is a
 * white rectangle in the dark theme. A transparent background lets the card
 * surface show through instead.
 */
export function depict(rdkit: RDKitModule, smiles: string, options: DepictOptions): string | null {
  const key = `${smiles} ${options.width}x${options.height} ${options.dark ? "d" : "l"} ${options.atoms?.join(",") ?? ""}`;
  return drawings.get(key, () => draw(rdkit, smiles, options));
}

function draw(rdkit: RDKitModule, smiles: string, options: DepictOptions): string | null {
  return withMol(rdkit, smiles, (mol) => {
    const details: Record<string, unknown> = {
      width: options.width,
      height: options.height,
      bondLineWidth: 1.2,
      addStereoAnnotation: true,
      clearBackground: false,
      backgroundColour: [0, 0, 0, 0],
      padding: 0.08,
    };
    if (options.dark) {
      details.symbolColour = [0.92, 0.93, 0.95];
      details.atomColourPalette = {
        "-1": [0.92, 0.93, 0.95],
        "0": [0.92, 0.93, 0.95],
        "6": [0.92, 0.93, 0.95],
        "7": [0.55, 0.65, 1.0],
        "8": [1.0, 0.45, 0.4],
        "9": [0.5, 0.9, 0.5],
        "15": [1.0, 0.6, 0.2],
        "16": [0.95, 0.85, 0.3],
        "17": [0.4, 0.9, 0.4],
        "35": [0.85, 0.45, 0.35],
      };
    }
    if (options.atoms?.length) {
      details.atoms = options.atoms;
      details.highlightColour = [0.86, 0.25, 0.2, 0.5];
    }
    return mol.get_svg_with_highlights(JSON.stringify(details));
  });
}

/** Common descriptors, rounded for display. */
export interface Descriptors {
  mw: number;
  clogp: number;
  tpsa: number;
  hbd: number;
  hba: number;
  rotb: number;
  rings: number;
  heavyAtoms: number;
}

/** The descriptors of *smiles*, remembered per SMILES; callers must not modify the result. */
export function descriptors(rdkit: RDKitModule, smiles: string): Descriptors | null {
  return computed.get(smiles, () => compute(rdkit, smiles));
}

function compute(rdkit: RDKitModule, smiles: string): Descriptors | null {
  return withMol(rdkit, smiles, (mol) => {
    const all = JSON.parse(mol.get_descriptors()) as Record<string, number>;
    return {
      mw: all.amw,
      clogp: all.CrippenClogP,
      tpsa: all.tpsa,
      hbd: all.lipinskiHBD,
      hba: all.lipinskiHBA,
      rotb: all.NumRotatableBonds,
      rings: all.NumRings,
      heavyAtoms: all.NumHeavyAtoms,
    };
  });
}

/** Canonical SMILES, or null when RDKit cannot parse the input. */
export function canonical(rdkit: RDKitModule, smiles: string): string | null {
  return withMol(rdkit, smiles, (mol) => mol.get_smiles());
}
