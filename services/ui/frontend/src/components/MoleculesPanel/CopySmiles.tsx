// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { CopyButton } from "../CopyButton/CopyButton";

/** The square copy icon beside a molecule's drawing; its click does not reach the row. */
export function CopySmiles({ smiles, className }: { smiles: string; className?: string }) {
  return <CopyButton value={smiles} label="Copy SMILES" icon className={className} />;
}
