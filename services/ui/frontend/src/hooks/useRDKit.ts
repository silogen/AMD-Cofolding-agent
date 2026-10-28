// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useState } from "react";
import type { RDKitModule } from "@rdkit/rdkit";
import { loadedRDKit, loadRDKit } from "../lib/rdkit";

/**
 * The RDKit module once it has loaded; null until then or if it failed.
 *
 * Nothing is downloaded while *needed* is false: the WebAssembly binary is
 * 7 MB, so callers only ask for it once there is a molecule to read.
 */
export function useRDKit(needed = true): { rdkit: RDKitModule | null; failed: boolean } {
  const [rdkit, setRDKit] = useState<RDKitModule | null>(loadedRDKit);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    if (!needed || rdkit) return;
    let cancelled = false;
    loadRDKit().then(
      (module) => !cancelled && setRDKit(module),
      () => !cancelled && setFailed(true),
    );
    return () => {
      cancelled = true;
    };
  }, [needed, rdkit]);
  return { rdkit, failed };
}
