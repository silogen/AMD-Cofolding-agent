// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useMemo } from "react";
import { depict } from "../../lib/rdkit";
import type { Theme } from "../../hooks/useTheme";
import { useRDKit } from "../../hooks/useRDKit";

interface Props {
  smiles: string;
  theme: Theme;
  width?: number;
  height?: number;
  atoms?: number[];
  label?: string;
  className?: string;
}

/**
 * A 2D drawing of one molecule.
 *
 * An invalid SMILES is a normal output of a generative model, so it gets a
 * quiet placeholder that keeps the size it was given, with the reason in its
 * tooltip; table rows and grid cells keep their shape either way.
 */
export function MolDepiction({ smiles, theme, width = 260, height = 170, atoms, label, className = "" }: Props) {
  const { rdkit, failed } = useRDKit();
  const atomKey = atoms?.join(",") ?? "";
  const svg = useMemo(
    () => (rdkit ? depict(rdkit, smiles, { width, height, dark: theme === "dark", atoms }) : null),
    // atoms is compared by value through atomKey.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [rdkit, smiles, width, height, theme, atomKey],
  );

  if (!rdkit && !failed) {
    return (
      <div className={`overflow-hidden ${className}`} style={{ aspectRatio: `${width} / ${height}` }}>
        <span className="bg-inset block size-full animate-pulse rounded-lg" />
      </div>
    );
  }

  if (!svg) {
    return (
      <Undrawable
        className={className}
        short={failed ? "No drawing" : "Not drawable"}
        reason={failed ? "2D drawing is unavailable: RDKit did not load." : `RDKit could not parse this SMILES, so it cannot be drawn:\n${smiles}`}
      />
    );
  }

  return (
    <div
      role="img"
      aria-label={`2D structure of ${label ?? smiles}`}
      className={`overflow-hidden [&>svg]:h-full [&>svg]:w-full ${className}`}
      // RDKit's SVG is generated locally from a SMILES string and contains
      // no user markup.
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}

/** A struck-through ring; the words only appear when the box is wide enough for them. */
function Undrawable({ className, short, reason }: { className: string; short: string; reason: string }) {
  return (
    <div
      role="img"
      aria-label={reason}
      title={reason}
      className={`@container text-faint flex flex-col items-center justify-center gap-1 overflow-hidden ${className}`}
    >
      <svg viewBox="0 0 24 24" className="size-4 shrink-0" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
        <path d="M12 3.5 19.4 7.75v8.5L12 20.5l-7.4-4.25v-8.5Z" strokeLinejoin="round" />
        <path d="M5 19 19 5" strokeLinecap="round" />
      </svg>
      <span className="hidden text-[10px] leading-none whitespace-nowrap @min-[6rem]:block">{short}</span>
    </div>
  );
}
