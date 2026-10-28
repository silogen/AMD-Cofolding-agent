// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { StructureMetrics } from "../../types/api";
import { cofoldFacts, type Fact } from "../../lib/moleculeFacts";

/** What a card says of a pose: affinity as an IC50, confidence, PoseBusters. Ligand ipTM is in the verdict and the details. */
export function structureFacts(metrics: StructureMetrics | undefined): Fact[] {
  return cofoldFacts(metrics ?? {}).filter((fact) => fact.label !== "Ligand ipTM");
}

/** The verdict as one word, or null when it is not known. */
export function verdictLabel(status: "pass" | "fail" | "unknown"): string | null {
  return status === "pass" ? "Valid" : status === "fail" ? "Invalid" : null;
}
