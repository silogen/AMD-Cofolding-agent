// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useRef, useState } from "react";
import type { StructurePanel } from "../../types/api";
import type { Theme } from "../../hooks/useTheme";

/** How the complex is drawn. The ligand is always the thing being designed. */
export type ViewStyle = "cartoon" | "surface" | "ligand";

interface Props {
  panel: StructurePanel;
  theme: Theme;
  style: ViewStyle;
  /** Fills its container; the container decides the height. */
  className?: string;
}

type Viewer = {
  addModel: (data: string, format: string) => void;
  setStyle: (selection: object, style: object) => void;
  addSurface: (type: unknown, style: object, selection: object) => void;
  setBackgroundColor: (color: string) => void;
  zoomTo: (selection?: object) => void;
  render: () => void;
  resize: () => void;
  clear: () => void;
  removeAllModels: () => void;
  removeAllSurfaces: () => void;
};

/** Canvas background per theme, matched to the card surface tokens. */
const BACKGROUND: Record<Theme, string> = {
  light: "#f3f4f7",
  dark: "#242830",
};

/**
 * Anything that is not protein, water or a common buffer ion is the designed
 * ligand. Boltz writes the ligand as a HETATM residue, so selecting by "not
 * protein" and excluding solvent is reliable without knowing its name.
 */
const LIGAND_SELECTION = {
  not: { resn: ["HOH", "WAT", "NA", "CL", "K", "MG", "ZN", "CA", "SO4", "PO4"] },
  hetflag: true,
};

/**
 * 3D view of a protein-ligand complex produced by Boltz.
 *
 * The viewer is created once per mount and reused: 3Dmol allocates a WebGL
 * context per viewer, and browsers cap how many contexts may exist at once, so
 * creating one per render would exhaust them in a long session.
 */
export function StructureViewer({ panel, theme, style, className = "" }: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<Viewer | null>(null);
  const failedRef = useRef(false);
  const [state, setState] = useState<"loading" | "ready" | "error">("loading");
  const [message, setMessage] = useState<string>("");

  // -- create the viewer once ---------------------------------------------
  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    let disposed = false;
    failedRef.current = false;

    const fail = (text: string) => {
      failedRef.current = true;
      if (!disposed) {
        setMessage(text);
        setState("error");
      }
    };

    // 3Dmol is large and only needed once a structure exists, so it is loaded
    // on demand rather than in the app bundle.
    import("3dmol")
      .then(($3Dmol) => {
        if (disposed || !hostRef.current) return;
        try {
          viewerRef.current = $3Dmol.createViewer(hostRef.current, {
            backgroundColor: BACKGROUND[theme],
          }) as unknown as Viewer;
        } catch {
          // 3Dmol needs WebGL. It is missing on some remote desktops, on
          // machines with WebGL disabled by policy, and in software-rendered
          // browsers. Say so plainly instead of leaving an empty box: the
          // structure file itself is still downloadable from the card.
          fail("This browser cannot show 3D structures because WebGL is not available.");
          return;
        }
        // Loading the structure is handled by the effect below, which reruns
        // once the viewer exists.
        if (!disposed) setState("loading");
      })
      .catch(() => fail("The 3D viewer could not be loaded."));

    return () => {
      disposed = true;
      try {
        viewerRef.current?.clear();
      } catch {
        // The WebGL context may already be gone during teardown.
      }
      viewerRef.current = null;
    };
    // The viewer is built once; the theme is applied by its own effect.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // -- load the file and apply the style ----------------------------------
  useEffect(() => {
    let cancelled = false;

    async function load() {
      // Wait for the dynamic import above to finish creating the viewer.
      const viewer = await waitForViewer(viewerRef, failedRef);
      if (cancelled || !viewer) return;

      setState("loading");
      try {
        const response = await fetch(panel.url);
        if (!response.ok) {
          throw new Error(`The structure file could not be read (HTTP ${response.status}).`);
        }
        const data = await response.text();
        if (cancelled) return;

        viewer.removeAllSurfaces();
        viewer.removeAllModels();
        viewer.addModel(data, panel.format);
        applyStyle(viewer, style, theme);
        viewer.setBackgroundColor(BACKGROUND[theme]);
        viewer.zoomTo(style === "ligand" ? LIGAND_SELECTION : {});
        viewer.render();
        setState("ready");
      } catch (error) {
        if (cancelled) return;
        setMessage(error instanceof Error ? error.message : "The structure could not be shown.");
        setState("error");
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, [panel.url, panel.format, style, theme]);

  // -- follow container resizes -------------------------------------------
  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const observer = new ResizeObserver(() => {
      try {
        viewerRef.current?.resize();
        viewerRef.current?.render();
      } catch {
        // Resizes can fire while the viewer is being torn down.
      }
    });
    observer.observe(host);
    return () => observer.disconnect();
  }, []);

  return (
    <div className={`bg-sunken relative overflow-hidden ${className}`}>
      <div ref={hostRef} className="viewer-host absolute inset-0" />

      {state === "loading" && (
        <p className="text-muted absolute inset-0 grid place-items-center text-xs" role="status">
          Loading structure…
        </p>
      )}
      {state === "error" && (
        <p className="text-muted absolute inset-0 grid place-items-center px-6 text-center text-xs">
          {message}
        </p>
      )}
    </div>
  );
}

/** Apply one of the three view styles to a loaded model. */
function applyStyle(viewer: Viewer, style: ViewStyle, theme: Theme): void {
  const ligandStyle = {
    stick: { radius: 0.18, colorscheme: theme === "dark" ? "cyanCarbon" : "greenCarbon" },
    sphere: { radius: 0.35 },
  };

  // Clear whatever the previous style left behind.
  viewer.setStyle({}, {});

  switch (style) {
    case "cartoon":
      viewer.setStyle({}, { cartoon: { color: "spectrum", opacity: 0.95 } });
      viewer.setStyle(LIGAND_SELECTION, ligandStyle);
      break;

    case "surface":
      viewer.setStyle({}, { cartoon: { color: "spectrum", opacity: 0.85 } });
      viewer.setStyle(LIGAND_SELECTION, ligandStyle);
      try {
        // A surface is expensive and can fail on a malformed model; the
        // cartoon underneath still gives a usable view if it does.
        viewer.addSurface("VDW", { opacity: 0.68, color: "white" }, { hetflag: false });
      } catch {
        /* keep the cartoon */
      }
      break;

    case "ligand":
      viewer.setStyle({}, { line: { opacity: 0.25 } });
      viewer.setStyle(LIGAND_SELECTION, ligandStyle);
      break;
  }
}

/**
 * Resolve once the viewer has been created by the dynamic import.
 *
 * Stops early when creating the viewer already failed, so a browser without
 * WebGL shows its message at once instead of spinning, and gives up after a
 * few seconds so a stalled import cannot wait forever.
 */
async function waitForViewer(
  ref: React.RefObject<Viewer | null>,
  failedRef: React.RefObject<boolean>,
): Promise<Viewer | null> {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (failedRef.current) return null;
    if (ref.current) return ref.current;
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  return null;
}
