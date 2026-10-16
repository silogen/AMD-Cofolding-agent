// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useRef, useState } from "react";
// The ESM build exposes only a default export, so the drawer classes are
// reached through the namespace rather than as named imports.
import SmilesDrawer from "smiles-drawer";
import type { MoleculePanel } from "../../types/api";
import type { Theme } from "../../hooks/useTheme";
import { CopyButton } from "../CopyButton/CopyButton";

/**
 * One drawer instance is enough for the whole app: draw() is synchronous per
 * call and holds no state between molecules, so rebuilding it per card would
 * only cost time.
 */
const drawer = new SmilesDrawer.SmiDrawer({ padding: 12, bondThickness: 1.1, fontSizeLarge: 6 });

interface Props {
  panel: MoleculePanel;
  theme: Theme;
}

/**
 * A ligand drawn as a 2D graph.
 *
 * REINVENT designs molecules as SMILES, which no chemist reads fluently as
 * text. The card shows the structure and keeps the SMILES available to copy
 * into another tool.
 */
export function MoleculeCard({ panel, theme }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;

    let cancelled = false;
    // A redraw must start from an empty element: the drawer appends to the
    // SVG rather than replacing what is already there.
    svg.replaceChildren();

    // Report the outcome after the effect body, not inside it, so drawing
    // stays a one-way update of the DOM and the result arrives as a normal
    // state change. An invalid SMILES is a normal result from a generative
    // model, not a crash, so a failure only changes what the card shows.
    const report = (didFail: boolean) => {
      queueMicrotask(() => {
        if (!cancelled) setFailed(didFail);
      });
    };

    try {
      drawer.draw(
        panel.smiles,
        svg,
        theme === "dark" ? "dark" : "light",
        () => report(false),
        () => report(true),
      );
    } catch {
      report(true);
    }

    return () => {
      cancelled = true;
    };
  }, [panel.smiles, theme]);

  return (
    <figure className="bg-raised border-subtle shadow-card m-0 overflow-hidden rounded-xl border">
      <figcaption className="border-subtle flex items-center gap-2 border-b px-3 py-2">
        <span
          aria-hidden="true"
          className="size-2 rounded-full"
          style={{ background: "var(--color-molecule)" }}
        />
        <span className="truncate text-xs font-semibold">{panel.label ?? "Ligand"}</span>
        <span className="text-faint ml-auto text-[11px] whitespace-nowrap">2D structure</span>
      </figcaption>

      {/* The SVG stays mounted even when drawing fails, so the ref survives
          and the next molecule can still be drawn into it. */}
      <div className="bg-sunken relative grid h-44 place-items-center p-2">
        <svg
          ref={svgRef}
          role="img"
          aria-label={`2D structure of ${panel.label ?? panel.smiles}`}
          className={`h-full w-full ${failed ? "invisible" : ""}`}
        />
        {failed && (
          <p className="text-muted absolute inset-0 grid place-items-center px-3 text-center text-xs">
            This SMILES could not be drawn. The text below is what the tool returned.
          </p>
        )}
      </div>

      <div className="border-subtle flex items-center gap-2 border-t px-3 py-2">
        <code className="text-muted min-w-0 flex-1 truncate font-mono text-[11px]" title={panel.smiles}>
          {panel.smiles}
        </code>
        <CopyButton value={panel.smiles} label="Copy SMILES" />
      </div>
    </figure>
  );
}
