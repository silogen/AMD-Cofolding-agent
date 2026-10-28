// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useRef, useState } from "react";
import { loadRDKit } from "../../lib/rdkit";
import { sampleForMap, shouldRecompute, type Molecules } from "../../lib/runAnalysis";
import { FP_WORDS, fingerprints } from "../../lib/umapFingerprints";
import type { UmapRequest, UmapResponse } from "../../workers/umap.worker";

/** A map that takes longer than this is abandoned; the previous one stays. */
const TIMEOUT_MS = 45_000;

export interface MapLayout {
  smiles: string[];
  /** n × 2. */
  coords: Float32Array;
  /** The run's last step when it was computed. */
  step: number;
  /** Molecules it was computed for (before invalid SMILES were dropped). */
  points: number;
  ms: number;
  failed?: string;
}

/** Per run, so going back to the view, or to a run, does not compute again. */
const layouts = new Map<string, MapLayout>();
const remember = (jobId: string, layout: MapLayout) => {
  layouts.delete(jobId);
  layouts.set(jobId, layout);
  if (layouts.size > 4) layouts.delete(layouts.keys().next().value as string);
};

interface Run {
  jobId: string;
  worker: Worker | null;
  timer: number;
}

/**
 * The run's molecules on a 2D map, from their Morgan fingerprints: computed
 * once, and again when a live run has moved on by two steps (ten if the last
 * map was slow) and the previous computation is done.
 */
export function useUmap(jobId: string | null, molecules: Molecules | null): { layout: MapLayout | null; working: boolean } {
  const run = useRef<Run | null>(null);
  const [version, setVersion] = useState(0);
  const [working, setWorking] = useState<string | null>(null);

  useEffect(
    () => () => {
      if (run.current) {
        run.current.worker?.terminate();
        window.clearTimeout(run.current.timer);
      }
      run.current = null;
    },
    [],
  );

  useEffect(() => {
    if (!jobId || !molecules || run.current) return;
    const previous = layouts.get(jobId) ?? null;
    const sample = sampleForMap(molecules);
    const due = shouldRecompute({
      busy: false,
      computedStep: previous?.step ?? null,
      latestStep: molecules.lastStep,
      lastMs: previous?.ms ?? null,
      points: sample.length,
      computedPoints: previous?.points ?? null,
    });
    if (!due) return;
    const token: Run = { jobId, worker: null, timer: 0 };
    run.current = token;
    const live = () => run.current === token;
    const finish = (layout: MapLayout | null) => {
      if (layout) remember(jobId, layout);
      if (!live()) return;
      window.clearTimeout(token.timer);
      token.worker?.terminate();
      run.current = null;
      setWorking(null);
      setVersion((v) => v + 1);
    };
    const fail = (message: string) => finish({ ...(previous ?? { smiles: [], coords: new Float32Array(0) }), step: molecules.lastStep, points: sample.length, ms: TIMEOUT_MS, failed: message });

    void (async () => {
      const rdkit = await loadRDKit().catch(() => null);
      if (!live()) return;
      if (!rdkit) return fail("RDKit did not load");
      setWorking(jobId);
      const smiles = sample.map((i) => molecules.smiles[i]);
      const packed = await fingerprints(rdkit, smiles, () => !live());
      if (!packed || !live()) return;
      const kept = packed.kept.map((i) => smiles[i]);
      const n = kept.length;
      let init: Float32Array | null = null;
      if (previous?.smiles.length) {
        const at = new Map(previous.smiles.map((s, i) => [s, i]));
        init = new Float32Array(n * 2).fill(NaN);
        kept.forEach((s, i) => {
          const j = at.get(s);
          if (j !== undefined) {
            init![2 * i] = previous.coords[2 * j];
            init![2 * i + 1] = previous.coords[2 * j + 1];
          }
        });
      }
      const worker = new Worker(new URL("../../workers/umap.worker.ts", import.meta.url), { type: "module" });
      token.worker = worker;
      token.timer = window.setTimeout(() => fail("The map took too long"), TIMEOUT_MS);
      worker.onmessage = ({ data }: MessageEvent<UmapResponse>) => {
        if ("error" in data) fail(data.error);
        else finish({ smiles: kept, coords: data.coords, step: molecules.lastStep, points: sample.length, ms: data.ms });
      };
      worker.onerror = () => fail("The map could not be computed");
      const request: UmapRequest = { id: 1, fps: packed.fps, n, words: FP_WORDS, init };
      worker.postMessage(request, init ? [packed.fps.buffer, init.buffer] : [packed.fps.buffer]);
    })();
  }, [jobId, molecules, version]);

  return { layout: jobId ? (layouts.get(jobId) ?? null) : null, working: working !== null && working === jobId };
}
