// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { lazy, Suspense, useState } from "react";
import { createPortal } from "react-dom";
import type { Ketcher } from "ketcher-core";
import type { Theme } from "../../hooks/useTheme";

// Ketcher and its WebAssembly chemistry engine are several megabytes: they
// load the first time the sketcher opens, never with the app.
const KetcherEditor = lazy(() => import("./KetcherEditor"));

interface Props {
  /** The molecule to start from; an empty canvas when unset. */
  smiles?: string;
  theme: Theme;
  onUse: (smiles: string) => void;
  onClose: () => void;
}

/** A molecule editor in a dialog; "Use" hands back what was drawn, as SMILES. */
export function Sketcher({ smiles, theme, onUse, onClose }: Props) {
  const [ketcher, setKetcher] = useState<Ketcher | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [reading, setReading] = useState(false);

  async function use() {
    if (!ketcher) return;
    setReading(true);
    setProblem(null);
    try {
      const drawn = (await ketcher.getSmiles()).trim();
      if (drawn) onUse(drawn);
      else setProblem("Draw a molecule first.");
    } catch (reason) {
      setProblem(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setReading(false);
    }
  }

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-2 sm:p-6" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Molecule sketcher"
        className="bg-raised border-subtle shadow-card flex h-[min(46rem,94vh)] w-[min(68rem,98vw)] flex-col overflow-hidden rounded-xl border"
      >
        <header className="border-subtle flex h-11 shrink-0 items-center gap-2 border-b px-4">
          <span aria-hidden="true" className="size-2 rounded-full" style={{ background: "var(--color-molecule)" }} />
          <h2 className="text-sm font-semibold">Sketch a molecule</h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close the sketcher"
            className="text-muted hover:text-body ml-auto grid size-7 cursor-pointer place-items-center rounded-lg text-base"
          >
            ×
          </button>
        </header>

        <div className={`ketcher-host relative min-h-0 flex-1 ${theme === "dark" ? "ketcher-dark" : ""}`}>
          <Suspense fallback={<p className="text-muted grid h-full place-items-center text-xs">Loading the sketcher…</p>}>
            <KetcherEditor initial={smiles} onReady={setKetcher} onError={setProblem} />
          </Suspense>
        </div>

        <footer className="border-subtle flex shrink-0 items-center gap-2 border-t px-4 py-2.5">
          <span className="text-muted min-w-0 flex-1 truncate text-xs" role="status" style={problem ? { color: "var(--color-accent)" } : undefined}>
            {problem ?? (ketcher ? "Use puts the SMILES in your message." : "")}
          </span>
          <button
            type="button"
            onClick={onClose}
            className="text-muted hover:text-body hover:bg-sunken h-8 cursor-pointer rounded-lg px-3 text-xs font-medium"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={() => void use()}
            disabled={!ketcher || reading}
            className="h-8 cursor-pointer rounded-lg px-4 text-xs font-semibold text-white hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
            style={{ background: "var(--color-accent)" }}
          >
            Use
          </button>
        </footer>
      </div>
    </div>,
    document.body,
  );
}
