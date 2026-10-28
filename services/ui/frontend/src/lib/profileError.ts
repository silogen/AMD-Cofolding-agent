// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/** A contact profiler error as a short line: a structure without a ligand is a fact, not a failure. */
export function profileError(error: string): { text: string; quiet: boolean } {
  if (/no ligand in this structure|no small-molecule ligand detected/i.test(error)) return { text: "No ligand in this structure", quiet: true };
  return { text: error.replace(/^\w+Error:\s*/, ""), quiet: false };
}
