// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { umap, type UmapInput } from "../lib/umap";

export interface UmapRequest extends UmapInput {
  id: number;
}

export type UmapResponse = { id: number; coords: Float32Array; ms: number } | { id: number; error: string };

const scope = self as unknown as {
  onmessage: ((event: MessageEvent<UmapRequest>) => void) | null;
  postMessage: (message: UmapResponse, transfer?: Transferable[]) => void;
};

scope.onmessage = ({ data }) => {
  const start = performance.now();
  try {
    const coords = umap(data);
    scope.postMessage({ id: data.id, coords, ms: performance.now() - start }, [coords.buffer]);
  } catch (error) {
    scope.postMessage({ id: data.id, error: error instanceof Error ? error.message : String(error) });
  }
};
