// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import type { Theme } from "../../hooks/useTheme";
import { colourScale, formatEndpoint, formatValue, heading, MIN_MAP_POINTS, pointDensity, pointsInside, scaleCss, type Endpoint, type Goal, type Molecules } from "../../lib/runAnalysis";
import { MolDepiction } from "../MolDepiction/MolDepiction";
import { useWidth } from "./plotUtils";
import { Placeholder } from "./shared";
import type { MapLayout } from "./useUmap";

export const DENSITY = "__density";
export const STEP = "__step";

export type MapMode = "lasso" | "box" | "pan";

interface Props {
  molecules: Molecules;
  layout: MapLayout | null;
  working: boolean;
  /** An endpoint key, STEP or DENSITY. */
  colourBy: string;
  /** The endpoint coloured by and its better side, when colouring by one. */
  endpoint: Endpoint | null;
  goal: Goal;
  mask: Uint8Array | null;
  mode: MapMode;
  /** A region drawn on the map, as SMILES; *add* when shift was held. */
  onRegion: (smiles: string[] | null, add: boolean) => void;
  hovered: number | null;
  onHover: (index: number | null) => void;
  onOpen: (index: number) => void;
  theme: Theme;
}

const PAD = 12;
const HOVER_PX = 8;
const NAN_COLOUR = "rgba(140,140,150,0.55)";
const MAX_ZOOM = 40;

type Drag =
  | { kind: "box"; x0: number; y0: number; x1: number; y1: number; shift: boolean }
  | { kind: "lasso"; points: number[]; shift: boolean }
  | { kind: "pan"; x0: number; y0: number; view: View; moved: boolean };

interface View {
  k: number;
  x: number;
  y: number;
}
const HOME: View = { k: 1, x: 0, y: 0 };

/**
 * The molecules on a 2D map of their fingerprints, coloured by an endpoint,
 * the step or the density of the map. Wheel to zoom about the pointer, drag
 * to select a box or a lasso (shift adds) or, in pan mode or with the middle
 * button, to move; double-click resets the view, Esc clears the selection.
 * Hover a point for its structure, click it to open the molecule.
 */
export function UmapPlot({ molecules, layout, working, colourBy, endpoint, goal, mask, mode, onRegion, hovered, onHover, onOpen, theme }: Props) {
  const [measure, width] = useWidth<HTMLDivElement>();
  const [box, setBox] = useState<HTMLDivElement | null>(null);
  const ref = useCallback(
    (node: HTMLDivElement | null) => {
      measure(node);
      setBox(node);
    },
    [measure],
  );
  const canvas = useRef<HTMLCanvasElement>(null);
  const [drag, setDrag] = useState<Drag | null>(null);
  const [view, setView] = useState<View>(HOME);
  const viewRef = useRef(view);
  useEffect(() => {
    viewRef.current = view;
  }, [view]);
  /** The point under the pointer, as opposed to one hovered in the table. */
  const [own, setOwn] = useState<number | null>(null);
  const height = Math.round(Math.min(380, Math.max(240, width * 0.78)));
  const dark = theme === "dark";

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

  const base = useMemo(() => {
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
    return { px: points.xs.map((x) => ox + (x - x0) * scale), py: points.ys.map((y) => oy + (y1 - y) * scale) };
  }, [points, width, height]);

  /** Where each point is on screen, zoomed and panned. */
  const screen = useMemo(() => (base ? { px: base.px.map((x) => x * view.k + view.x), py: base.py.map((y) => y * view.k + view.y) } : null), [base, view]);

  const colouring = useMemo(() => {
    if (!points) return null;
    const raw =
      colourBy === DENSITY
        ? pointDensity(points.xs, points.ys)
        : Float64Array.from(points.index, (m) => (colourBy === STEP ? molecules.step[m] : (molecules.values.get(colourBy)?.[m] ?? NaN)));
    const scale = colourScale(raw, colourBy === DENSITY || colourBy === STEP || !endpoint ? "linear" : endpoint.scale, colourBy === DENSITY || colourBy === STEP ? { direction: "up" } : goal);
    const css = Array.from(raw, (v) => {
      const p = scale ? scale.position(v) : NaN;
      if (!scale || !Number.isFinite(p)) return NAN_COLOUR;
      const [r, g, b] = scale.paint(p, dark);
      return `rgb(${r},${g},${b})`;
    });
    // Drawn worst to best, so the best stay on top.
    const good = Array.from(raw, (v) => (scale ? scale.goodness(v) : -1));
    const order = Array.from(raw, (_, i) => i).sort((a, b) => good[a] - good[b]);
    return { raw, css, order, scale, gradient: scale ? scaleCss(scale, dark) : "" };
  }, [points, colourBy, molecules, endpoint, goal, dark]);

  const hoveredPoint = useMemo(() => {
    if (hovered === null || !points) return -1;
    return points.index.indexOf(hovered);
  }, [hovered, points]);

  useEffect(() => {
    const element = canvas.current;
    if (!element || !screen || !colouring || !points) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    element.width = Math.round(width * dpr);
    element.height = Math.round(height * dpr);
    const ctx = element.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const { px, py } = screen;
    const dot = (i: number, r: number) => {
      if (px[i] < -r || py[i] < -r || px[i] > width + r || py[i] > height + r) return;
      ctx.fillStyle = colouring.css[i];
      ctx.beginPath();
      ctx.arc(px[i], py[i], r, 0, Math.PI * 2);
      ctx.fill();
    };
    const radius = (points.xs.length > 1500 ? 2 : 2.6) * Math.min(1.8, Math.sqrt(view.k));
    if (mask) {
      ctx.globalAlpha = 0.12;
      for (const i of colouring.order) if (!mask[points.index[i]]) dot(i, radius - 0.4);
      ctx.globalAlpha = 1;
      for (const i of colouring.order) if (mask[points.index[i]]) dot(i, radius + 0.6);
    } else {
      ctx.globalAlpha = 0.85;
      for (const i of colouring.order) dot(i, radius);
      ctx.globalAlpha = 1;
    }
    if (hoveredPoint >= 0) {
      ctx.strokeStyle = dark ? "#f4f4f5" : "#0a0a0b";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(px[hoveredPoint], py[hoveredPoint], radius + 3.5, 0, Math.PI * 2);
      ctx.stroke();
    }
  }, [screen, colouring, points, mask, hoveredPoint, width, height, dark, view.k]);

  // Wheel zooms about the pointer; zooming out of the full view scrolls the page instead.
  useEffect(() => {
    if (!box) return;
    const onWheel = (event: WheelEvent) => {
      const zoomOut = event.deltaY > 0;
      setView((v) => {
        if (zoomOut && v.k <= 1) return v.k === 1 && !v.x && !v.y ? v : HOME;
        const rect = box.getBoundingClientRect();
        const mx = event.clientX - rect.left;
        const my = event.clientY - rect.top;
        const k = Math.min(MAX_ZOOM, Math.max(1, v.k * Math.exp(-event.deltaY * (event.deltaMode === 1 ? 0.05 : 0.0015))));
        if (k === 1) return HOME;
        const f = k / v.k;
        return { k, x: mx - (mx - v.x) * f, y: my - (my - v.y) * f };
      });
      if (!zoomOut || viewRef.current.k > 1) event.preventDefault();
    };
    box.addEventListener("wheel", onWheel, { passive: false });
    return () => box.removeEventListener("wheel", onWheel);
  }, [box]);

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
    const rect = event.currentTarget.getBoundingClientRect();
    return [event.clientX - rect.left, event.clientY - rect.top] as const;
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
    if (event.button !== 0 && event.button !== 1) return;
    if (event.button === 1) event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    event.currentTarget.focus({ preventScroll: true });
    const [x, y] = local(event);
    if (mode === "pan" || event.button === 1) setDrag({ kind: "pan", x0: x, y0: y, view, moved: false });
    else setDrag(mode === "box" ? { kind: "box", x0: x, y0: y, x1: x, y1: y, shift: event.shiftKey } : { kind: "lasso", points: [x, y], shift: event.shiftKey });
  }

  function move(event: PointerEvent<HTMLDivElement>) {
    const [x, y] = local(event);
    if (drag?.kind === "pan") {
      const dx = x - drag.x0;
      const dy = y - drag.y0;
      if (drag.moved || dx * dx + dy * dy > 9) {
        if (!drag.moved) setDrag({ ...drag, moved: true });
        setView({ k: drag.view.k, x: drag.view.x + dx, y: drag.view.y + dy });
      }
    } else if (drag?.kind === "box") setDrag({ ...drag, x1: x, y1: y });
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
    const moved = current.kind === "pan" ? current.moved : current.kind === "box" ? Math.abs(current.x1 - current.x0) + Math.abs(current.y1 - current.y0) > 6 : current.points.length > 6;
    if (!moved) {
      // The second click of a double-click resets the view; it does not open the molecule again.
      const i = event.detail > 1 || event.button !== 0 ? -1 : nearest(x, y);
      if (i >= 0) onOpen(points!.index[i]);
      return;
    }
    if (current.kind === "pan") return;
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
  const zoomed = view.k !== 1 || view.x !== 0 || view.y !== 0;
  const scale = colouring?.scale ?? null;
  const name = colourBy === DENSITY ? "Density" : colourBy === STEP ? "Step" : endpoint ? heading(endpoint) : colourBy;
  const tick = (v: number) => (colourBy === STEP ? formatValue(v, 0) : endpoint ? formatEndpoint(endpoint, v) : formatValue(v, 2));
  const hoverValue = own !== null && colourBy !== DENSITY ? (colourBy === STEP ? molecules.step[own] : (molecules.values.get(colourBy)?.[own] ?? NaN)) : NaN;
  const better = scale?.direction === "down" ? "lower is better" : scale?.direction === "up" && endpoint && endpoint.kind !== "total" ? "higher is better" : scale?.target ? `TTP window ${tick(scale.target[0])}–${tick(scale.target[1])}` : "";
  const cursor = drag?.kind === "pan" ? "cursor-grabbing" : hoverAt ? "cursor-pointer" : mode === "pan" ? "cursor-grab" : "cursor-crosshair";

  return (
    <div
      ref={ref}
      tabIndex={0}
      onPointerDown={down}
      onMouseDown={(event) => event.button === 1 && event.preventDefault()}
      onPointerMove={move}
      onPointerUp={up}
      onPointerCancel={() => setDrag(null)}
      onPointerLeave={() => {
        if (drag) return;
        setOwn(null);
        if (hovered !== null) onHover(null);
      }}
      onDoubleClick={() => setView(HOME)}
      onKeyDown={key}
      className={`relative touch-none overflow-hidden rounded-lg outline-none select-none ${cursor}`}
      style={{ height }}
      aria-label="Map of the molecules by structure"
      role="application"
    >
      <canvas ref={canvas} style={{ width, height }} className="block" />
      {drag && drag.kind !== "pan" && (
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
      {scale && (
        <div
          className="pointer-events-none absolute bottom-1 left-1 rounded-md px-1.5 pt-0.5 pb-0 text-[9.5px] tabular-nums"
          style={{ background: "color-mix(in srgb, var(--surface-raised) 82%, transparent)" }}
          title={better}
        >
          <div className="text-muted mb-0.5 flex items-baseline gap-1.5">
            <span>{name}</span>
            {better && <span className="text-faint text-[9px]">{better}</span>}
          </div>
          <div className="relative h-1.5 w-36 rounded-full" style={{ background: colouring?.gradient }}>
            {scale.target && (
              <span
                className="absolute -inset-y-px rounded-sm border"
                style={{
                  left: `${scale.position(scale.target[0]) * 100}%`,
                  width: `${Math.max(1, (scale.position(scale.target[1]) - scale.position(scale.target[0])) * 100)}%`,
                  borderColor: "var(--text-body)",
                  opacity: 0.5,
                }}
              />
            )}
          </div>
          <div className="text-faint relative h-3.5 w-36">
            {colourBy === DENSITY ? (
              <>
                <span className="absolute left-0">sparse</span>
                <span className="absolute right-0">dense</span>
              </>
            ) : (
              scale.ticks.map((t) => {
                const p = scale.position(t);
                return (
                  <span key={t} className="absolute top-0 whitespace-nowrap" style={{ left: `${p * 100}%`, transform: `translateX(${p < 0.1 ? 0 : p > 0.9 ? -100 : -50}%)` }}>
                    {tick(t)}
                  </span>
                );
              })
            )}
          </div>
        </div>
      )}
      {zoomed && (
        <button
          type="button"
          onPointerDown={(event) => event.stopPropagation()}
          onClick={() => setView(HOME)}
          className="text-muted hover:text-body absolute top-1 left-1 cursor-pointer rounded-md px-1.5 py-0.5 text-[10px]"
          style={{ background: "color-mix(in srgb, var(--surface-raised) 82%, transparent)" }}
          title="Reset the view (double-click)"
        >
          Reset view · {view.k.toFixed(1)}×
        </button>
      )}
      {working && <span className="absolute top-1 right-1 size-1.5 animate-pulse rounded-full" style={{ background: "var(--color-structure)" }} title="Updating the map" />}
      {hoverAt && own !== null && own === hovered && (
        <div
          className="bg-raised border-subtle shadow-card pointer-events-none absolute z-10 rounded-lg border p-1"
          style={{
            left: Math.min(Math.max(0, hoverAt.x + 12), width - 148),
            top: hoverAt.y > height - 126 ? hoverAt.y - 126 : hoverAt.y + 12,
          }}
        >
          <MolDepiction smiles={molecules.smiles[own]} theme={theme} width={140} height={96} className="h-24 w-[140px]" />
          {Number.isFinite(hoverValue) && (
            <div className="text-muted flex justify-between gap-2 px-1 text-[10px] tabular-nums">
              <span className="truncate">{colourBy === STEP ? "Step" : (endpoint?.label ?? "")}</span>
              <span className="text-body font-mono">
                {tick(hoverValue)}
                {endpoint?.unit && colourBy !== STEP ? <span className="text-faint"> {endpoint.unit}</span> : null}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
