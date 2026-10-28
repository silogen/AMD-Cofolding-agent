// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import type { Theme } from "../../hooks/useTheme";
import { colourDomain, MIN_MAP_POINTS, pointDensity, pointsInside, viridis, viridisCss, viridisFrom, type Molecules } from "../../lib/runAnalysis";
import { MolDepiction } from "../MolDepiction/MolDepiction";
import { short, useWidth } from "./plotUtils";
import { Placeholder } from "./shared";
import type { MapLayout } from "./useUmap";

export type ColourBy = string;
export const DENSITY = "__density";
export const STEP = "__step";

interface Props {
  molecules: Molecules;
  layout: MapLayout | null;
  working: boolean;
  colourBy: ColourBy;
  mask: Uint8Array | null;
  mode: "box" | "lasso";
  /** A region drawn on the map, as SMILES; *add* when shift was held. */
  onRegion: (smiles: string[] | null, add: boolean) => void;
  hovered: number | null;
  onHover: (index: number | null) => void;
  onOpen: (index: number) => void;
  theme: Theme;
}

const PAD = 12;
const HOVER_PX = 8;
const NAN_COLOUR = "rgba(140,140,150,0.6)";

type Drag = { kind: "box"; x0: number; y0: number; x1: number; y1: number; shift: boolean } | { kind: "lasso"; points: number[]; shift: boolean };

/**
 * The molecules on a 2D map of their fingerprints, coloured by an endpoint,
 * the step or the density of the map. Drag to select a box or a lasso
 * (shift adds), double-click or Esc to clear; hover a point for its structure,
 * click it to open the molecule.
 */
export function UmapPlot({ molecules, layout, working, colourBy, mask, mode, onRegion, hovered, onHover, onOpen, theme }: Props) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const canvas = useRef<HTMLCanvasElement>(null);
  const [drag, setDrag] = useState<Drag | null>(null);
  /** The point under the pointer, as opposed to one hovered in the table. */
  const [own, setOwn] = useState<number | null>(null);
  const height = Math.round(Math.min(380, Math.max(240, width * 0.78)));

  // Points of the layout that are still molecules of the run, with their positions in the layout.
  const points = useMemo(() => {
    if (!layout || !layout.smiles.length) return null;
    const index: number[] = [];
    const xs: number[] = [];
    const ys: number[] = [];
    layout.smiles.forEach((s, i) => {
      const m = molecules.index.get(s);
      if (m === undefined) return;
      index.push(m);
      xs.push(layout.coords[2 * i]);
      ys.push(layout.coords[2 * i + 1]);
    });
    return { index: Int32Array.from(index), xs: Float32Array.from(xs), ys: Float32Array.from(ys) };
  }, [layout, molecules]);

  const screen = useMemo(() => {
    if (!points || !width) return null;
    let x0 = Infinity;
    let x1 = -Infinity;
    let y0 = Infinity;
    let y1 = -Infinity;
    for (let i = 0; i < points.xs.length; i++) {
      x0 = Math.min(x0, points.xs[i]);
      x1 = Math.max(x1, points.xs[i]);
      y0 = Math.min(y0, points.ys[i]);
      y1 = Math.max(y1, points.ys[i]);
    }
    const scale = Math.min((width - 2 * PAD) / (x1 - x0 || 1), (height - 2 * PAD) / (y1 - y0 || 1));
    const ox = (width - (x1 - x0) * scale) / 2;
    const oy = (height - (y1 - y0) * scale) / 2;
    const px = points.xs.map((x) => ox + (x - x0) * scale);
    const py = points.ys.map((y) => oy + (y1 - y) * scale);
    return { px, py };
  }, [points, width, height]);

  const colours = useMemo(() => {
    if (!points) return null;
    const raw =
      colourBy === DENSITY
        ? pointDensity(points.xs, points.ys)
        : Float64Array.from(points.index, (m) => (colourBy === STEP ? molecules.step[m] : (molecules.values.get(colourBy)?.[m] ?? NaN)));
    const domain = colourDomain(raw);
    const from = viridisFrom(theme === "dark");
    const css = Array.from(raw, (v) => {
      if (!domain || !Number.isFinite(v)) return NAN_COLOUR;
      const [r, g, b] = viridis(from + (1 - from) * ((v - domain[0]) / (domain[1] - domain[0])));
      return `rgb(${r},${g},${b})`;
    });
    // Drawn low to high, so the best of an endpoint stay on top.
    const order = Array.from(raw, (_, i) => i).sort((a, b) => (Number.isFinite(raw[a]) ? raw[a] : -Infinity) - (Number.isFinite(raw[b]) ? raw[b] : -Infinity));
    return { css, order, domain, gradient: viridisCss(from) };
  }, [points, colourBy, molecules, theme]);

  const hoveredPoint = useMemo(() => {
    if (hovered === null || !points) return -1;
    return points.index.indexOf(hovered);
  }, [hovered, points]);

  useEffect(() => {
    const element = canvas.current;
    if (!element || !screen || !colours || !points) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    element.width = Math.round(width * dpr);
    element.height = Math.round(height * dpr);
    const ctx = element.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const { px, py } = screen;
    const dot = (i: number, r: number) => {
      ctx.fillStyle = colours.css[i];
      ctx.beginPath();
      ctx.arc(px[i], py[i], r, 0, Math.PI * 2);
      ctx.fill();
    };
    const radius = points.xs.length > 1500 ? 2 : 2.6;
    if (mask) {
      ctx.globalAlpha = 0.14;
      for (const i of colours.order) if (!mask[points.index[i]]) dot(i, radius - 0.4);
      ctx.globalAlpha = 1;
      for (const i of colours.order) if (mask[points.index[i]]) dot(i, radius + 0.6);
    } else {
      ctx.globalAlpha = 0.85;
      for (const i of colours.order) dot(i, radius);
      ctx.globalAlpha = 1;
    }
    if (hoveredPoint >= 0) {
      ctx.strokeStyle = theme === "dark" ? "#f4f4f5" : "#0a0a0b";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(px[hoveredPoint], py[hoveredPoint], radius + 3.5, 0, Math.PI * 2);
      ctx.stroke();
    }
  }, [screen, colours, points, mask, hoveredPoint, width, height, theme]);

  if (!layout || !layout.smiles.length || !points) {
    return (
      <div ref={ref}>
        <Placeholder height={260}>
          {molecules.n < MIN_MAP_POINTS ? "Too few molecules for a map yet." : layout?.failed && !working ? layout.failed : "Mapping the molecules…"}
        </Placeholder>
      </div>
    );
  }

  const local = (event: PointerEvent) => {
    const box = event.currentTarget.getBoundingClientRect();
    return [event.clientX - box.left, event.clientY - box.top] as const;
  };
  const nearest = (x: number, y: number) => {
    if (!screen) return -1;
    let best = -1;
    let bestD = HOVER_PX * HOVER_PX;
    for (let i = 0; i < screen.px.length; i++) {
      const d = (screen.px[i] - x) ** 2 + (screen.py[i] - y) ** 2;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return best;
  };

  function down(event: PointerEvent<HTMLDivElement>) {
    if (event.button !== 0) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    event.currentTarget.focus({ preventScroll: true });
    const [x, y] = local(event);
    setDrag(mode === "box" ? { kind: "box", x0: x, y0: y, x1: x, y1: y, shift: event.shiftKey } : { kind: "lasso", points: [x, y], shift: event.shiftKey });
  }

  function move(event: PointerEvent<HTMLDivElement>) {
    const [x, y] = local(event);
    if (drag?.kind === "box") setDrag({ ...drag, x1: x, y1: y });
    else if (drag?.kind === "lasso") {
      const n = drag.points.length;
      if ((drag.points[n - 2] - x) ** 2 + (drag.points[n - 1] - y) ** 2 > 9) setDrag({ ...drag, points: [...drag.points, x, y] });
    }
    if (!drag) {
      const i = nearest(x, y);
      const m = i >= 0 ? points!.index[i] : null;
      if (m !== own) setOwn(m);
      if (m !== hovered) onHover(m);
    }
  }

  function up(event: PointerEvent<HTMLDivElement>) {
    if (!drag || !screen) return;
    const [x, y] = local(event);
    const current = drag;
    setDrag(null);
    const moved = current.kind === "box" ? Math.abs(current.x1 - current.x0) + Math.abs(current.y1 - current.y0) > 6 : current.points.length > 6;
    if (!moved) {
      const i = nearest(x, y);
      if (i >= 0) onOpen(points!.index[i]);
      return;
    }
    const inside = pointsInside(screen.px, screen.py, current.kind === "box" ? current : { kind: "lasso", polygon: current.points });
    onRegion(
      inside.map((i) => molecules.smiles[points!.index[i]]),
      current.shift,
    );
  }

  function key(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === "Escape") {
      setDrag(null);
      onRegion(null, false);
    }
  }

  const hoverAt = hoveredPoint >= 0 && screen && !drag ? { x: screen.px[hoveredPoint], y: screen.py[hoveredPoint] } : null;
  const domain = colours?.domain ?? null;

  return (
    <div
      ref={ref}
      tabIndex={0}
      onPointerDown={down}
      onPointerMove={move}
      onPointerUp={up}
      onPointerCancel={() => setDrag(null)}
      onPointerLeave={() => {
        if (drag) return;
        setOwn(null);
        if (hovered !== null) onHover(null);
      }}
      onDoubleClick={() => onRegion(null, false)}
      onKeyDown={key}
      className={`relative touch-none rounded-lg outline-none select-none ${hoverAt ? "cursor-pointer" : "cursor-crosshair"}`}
      style={{ height }}
      aria-label="Map of the molecules by structure"
      role="application"
    >
      <canvas ref={canvas} style={{ width, height }} className="block" />
      {drag && (
        <svg width={width} height={height} className="pointer-events-none absolute inset-0">
          {drag.kind === "box" ? (
            <rect
              x={Math.min(drag.x0, drag.x1)}
              y={Math.min(drag.y0, drag.y1)}
              width={Math.abs(drag.x1 - drag.x0)}
              height={Math.abs(drag.y1 - drag.y0)}
              fill="var(--color-accent)"
              fillOpacity={0.06}
              stroke="var(--color-accent)"
              strokeDasharray="4 3"
            />
          ) : (
            <polygon points={drag.points.join(" ")} fill="var(--color-accent)" fillOpacity={0.06} stroke="var(--color-accent)" strokeDasharray="4 3" />
          )}
        </svg>
      )}
      {domain && (
        <div className="text-faint pointer-events-none absolute bottom-1 left-1 flex items-center gap-1 text-[9px] tabular-nums">
          <span>{colourBy === DENSITY ? "sparse" : short(domain[0])}</span>
          <span className="inline-block h-1.5 w-16 rounded-full" style={{ background: colours?.gradient }} />
          <span>{colourBy === DENSITY ? "dense" : short(domain[1])}</span>
        </div>
      )}
      {working && <span className="absolute top-1 right-1 size-1.5 animate-pulse rounded-full" style={{ background: "var(--color-structure)" }} title="Updating the map" />}
      {hoverAt && own !== null && own === hovered && (
        <div
          className="bg-raised border-subtle shadow-card pointer-events-none absolute z-10 rounded-lg border p-1"
          style={{
            left: Math.min(Math.max(0, hoverAt.x + 12), width - 148),
            top: hoverAt.y > height - 112 ? hoverAt.y - 112 : hoverAt.y + 12,
          }}
        >
          <MolDepiction smiles={molecules.smiles[own]} theme={theme} width={140} height={96} className="h-24 w-[140px]" />
        </div>
      )}
    </div>
  );
}
