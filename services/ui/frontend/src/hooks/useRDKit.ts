// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useState } from "react";
import type { RDKitModule } from "@rdkit/rdkit";
import { loadRDKit } from "../lib/rdkit";

/** The RDKit module once it has loaded; null until then or if it failed. */
export function useRDKit(): { rdkit: RDKitModule | null; failed: boolean } {
  const [rdkit, setRDKit] = useState<RDKitModule | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let cancelled = false;
    loadRDKit().then(
      (module) => !cancelled && setRDKit(module),
      () => !cancelled && setFailed(true),
    );
    return () => {
      cancelled = true;
    };
  }, []);
  return { rdkit, failed };
}
