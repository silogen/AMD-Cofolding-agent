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
 * short message in place of the drawing rather than an error.
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

  if (!rdkit) {
    return (
      <div
        className={`text-faint grid place-items-center text-[11px] ${className}`}
        style={{ aspectRatio: `${width} / ${height}` }}
      >
        {failed ? "Drawing unavailable" : <span className="bg-inset size-full animate-pulse rounded-lg" />}
      </div>
    );
  }

  if (!svg) {
    return (
      <p className={`text-muted grid place-items-center px-3 text-center text-xs ${className}`}>
        This SMILES could not be drawn.
      </p>
    );
  }

  return (
    <div
      role="img"
      aria-label={`2D structure of ${label ?? smiles}`}
      className={`[&>svg]:h-full [&>svg]:w-full ${className}`}
      // RDKit's SVG is generated locally from a SMILES string and contains
      // no user markup.
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}
