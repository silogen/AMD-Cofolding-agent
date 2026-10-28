// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { RDKitModule } from "@rdkit/rdkit";
import { packBytes } from "./umap";
import { withMol } from "./rdkit";

/** Morgan radius 2 on 1024 bits: 32 words a molecule. */
export const FP_BITS = 1024;
export const FP_WORDS = FP_BITS / 32;
const OPTIONS = JSON.stringify({ radius: 2, nBits: FP_BITS });
/** Fingerprints kept between maps: 128 bytes each. */
const CACHE = 20_000;
/** Work this long between yields to the page, in ms. */
const SLICE_MS = 12;

const cache = new Map<string, Uint32Array | null>();

function fingerprint(rdkit: RDKitModule, smiles: string): Uint32Array | null {
  const hit = cache.get(smiles);
  if (hit !== undefined) return hit;
  const bytes = withMol(rdkit, smiles, (mol) => mol.get_morgan_fp_as_uint8array(OPTIONS));
  const fp = bytes ? packBytes(bytes, FP_WORDS) : null;
  cache.set(smiles, fp);
  if (cache.size > CACHE) cache.delete(cache.keys().next().value as string);
  return fp;
}

const pause = () => new Promise<void>((resolve) => window.setTimeout(resolve, 0));

/**
 * The fingerprints of *smiles* as one packed array, computed a slice at a
 * time so the page stays responsive. SMILES RDKit cannot parse are left out:
 * `kept` holds the input positions of the rows.
 */
export async function fingerprints(rdkit: RDKitModule, smiles: readonly string[], cancelled: () => boolean): Promise<{ fps: Uint32Array; kept: number[] } | null> {
  const parts: Uint32Array[] = [];
  const kept: number[] = [];
  let since = performance.now();
  for (let i = 0; i < smiles.length; i++) {
    const fp = fingerprint(rdkit, smiles[i]);
    if (fp) {
      parts.push(fp);
      kept.push(i);
    }
    if (performance.now() - since > SLICE_MS) {
      await pause();
      if (cancelled()) return null;
      since = performance.now();
    }
  }
  const fps = new Uint32Array(parts.length * FP_WORDS);
  parts.forEach((fp, i) => fps.set(fp, i * FP_WORDS));
  return { fps, kept };
}
