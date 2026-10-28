// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { memo, useEffect, useMemo, useRef, useState, type PointerEvent } from "react";
import type { Theme } from "../../hooks/useTheme";
import { axisOf, bandwidth, extent, formatEndpoint, heading, jitter, kde, quartiles, type Endpoint, type Molecules, type Ranges } from "../../lib/runAnalysis";
import { useWidth } from "./plotUtils";
import { Placeholder } from "./shared";

interface Props {
  molecules: Molecules;
  /** The endpoints shown, in order. */
  endpoints: readonly Endpoint[];
  mask: Uint8Array | null;
  ranges: Ranges;
  onRange: (key: string, range: [number, number] | null) => void;
  colours: ReadonlyMap<string, string>;
  hovered: number | null;
  theme: Theme;
}

const HEIGHT = 230;
const PLOT = HEIGHT - 26;
const TOP = 14;
const BOTTOM = 14;
const POINTS = 48;

/**
 * One violin per endpoint, each on its own scale (log for heavy-tailed
 * values), with one faint jittered dot per molecule. Drag up or down on one
 * to keep a range of its values, click it to let go; ranges on several
 * endpoints combine. The molecules selected anywhere show as a solid overlay.
 */
export function ViolinPlot({ molecules, endpoints, mask, ranges, onRange, colours, hovered, theme }: Props) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const offsets = useMemo(() => Float64Array.from(molecules.smiles, jitter), [molecules]);
  const columnWidth = endpoints.length ? Math.max(72, Math.min(140, Math.floor(width / endpoints.length))) : 0;
  return (
    <div ref={ref} className="overflow-x-auto">
      {!endpoints.length ? (
        <Placeholder height={HEIGHT}>Pick endpoints to compare.</Placeholder>
      ) : (
        <div className="flex" style={{ width: Math.max(width, columnWidth * endpoints.length) }}>
          {endpoints.map((e) => (
            <Violin
              key={e.key}
              endpoint={e}
              values={molecules.values.get(e.key) ?? null}
              offsets={offsets}
              width={columnWidth}
              colour={colours.get(e.key) ?? "var(--text-muted)"}
              mask={mask}
              range={ranges[e.key] ?? null}
              onRange={onRange}
              hovered={hovered}
              theme={theme}
            />
          ))}
        </div>
      )}
    </div>
  );
}

interface ViolinProps {
  endpoint: Endpoint;
  values: Float64Array | null;
  offsets: Float64Array;
  width: number;
  colour: string;
  mask: Uint8Array | null;
  range: readonly [number, number] | null;
  onRange: (key: string, range: [number, number] | null) => void;
  hovered: number | null;
  theme: Theme;
}

const Violin = memo(function Violin({ endpoint, values, offsets, width, colour, mask, range, onRange, hovered, theme }: ViolinProps) {
  const [draft, setDraft] = useState<{ from: number; to: number } | null>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const axis = useMemo(() => axisOf(endpoint.scale), [endpoint.scale]);
  const full = useMemo(() => {
    if (!values) return null;
    const all: number[] = [];
    for (const v of values) {
      const a = axis.to(v);
      if (Number.isFinite(a)) all.push(a);
    }
    const ext = extent(all);
    if (!ext) return null;
    const [lo, hi] = ext;
    const h = bandwidth(all, lo, hi);
    const raw = kde(all, lo, hi, h, POINTS, 1);
    const peak = Math.max(...raw) || 1;
    return { lo, hi, h, peak, grid: raw.map((v) => v / peak), quartiles: quartiles(all), count: all.length };
  }, [values, axis]);
  const subset = useMemo(() => {
    if (!values || !full || !mask) return null;
    const picked: number[] = [];
    for (let i = 0; i < values.length; i++) {
      const a = mask[i] ? axis.to(values[i]) : NaN;
      if (Number.isFinite(a)) picked.push(a);
    }
    return { grid: kde(picked, full.lo, full.hi, full.h, POINTS, full.peak), count: picked.length };
  }, [values, full, mask, axis]);

  const plotH = PLOT - TOP - BOTTOM;
  const cx = width / 2;
  const half = width * 0.4;

  useEffect(() => {
    const element = canvas.current;
    if (!element || !full || !values) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    element.width = Math.round(width * dpr);
    element.height = Math.round(PLOT * dpr);
    const ctx = element.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, PLOT);
    ctx.fillStyle = getComputedStyle(element).color;
    const { lo, hi, grid } = full;
    const span = hi - lo;
    const draw = (keep: (i: number) => boolean, alpha: number, size: number) => {
      ctx.globalAlpha = alpha;
      for (let i = 0; i < values.length; i++) {
        if (!keep(i)) continue;
        const a = axis.to(values[i]);
        if (!Number.isFinite(a)) continue;
        const g = span > 0 ? ((a - lo) / span) * (POINTS - 1) : 0;
        const at = Math.min(POINTS - 2, Math.floor(g));
        const d = span > 0 ? grid[at] + (grid[at + 1] - grid[at]) * (g - at) : 1;
        const x = cx + offsets[i] * Math.max(0.08, d) * half * 0.85;
        const y = span > 0 ? TOP + (1 - (a - lo) / span) * plotH : TOP + plotH / 2;
        ctx.fillRect(x - size / 2, y - size / 2, size, size);
      }
    };
    if (mask) {
      draw((i) => !mask[i], 0.07, 1.7);
      draw((i) => mask[i] === 1, 0.9, 3);
    } else draw(() => true, full.count > 1500 ? 0.14 : full.count > 300 ? 0.22 : 0.4, 1.7);
    ctx.globalAlpha = 1;
  }, [full, values, offsets, mask, width, cx, half, plotH, axis, colour, theme]);

  if (!full) {
    return (
      <div style={{ width }} className="shrink-0">
        <Placeholder height={PLOT}>no values</Placeholder>
        <Label endpoint={endpoint} />
      </div>
    );
  }
  const { lo, hi } = full;
  const flat = hi === lo;
  const y = (a: number) => (flat ? TOP + plotH / 2 : TOP + (1 - (a - lo) / (hi - lo)) * plotH);
  const yValue = (v: number) => y(Math.min(hi, Math.max(lo, axis.to(v))));
  const valueAt = (py: number) => axis.from(flat ? lo : Math.min(hi, Math.max(lo, lo + (1 - (py - TOP) / plotH) * (hi - lo))));
  const shape = (grid: Float64Array) => {
    const ys = Array.from(grid, (_, g) => TOP + (1 - g / (POINTS - 1)) * plotH);
    const right = Array.from(grid, (d, g) => `${g ? "L" : "M"}${(cx + d * half).toFixed(1)},${ys[g].toFixed(1)}`).join("");
    const left = Array.from(grid, (_, g) => g)
      .reverse()
      .map((g) => `L${(cx - grid[g] * half).toFixed(1)},${ys[g].toFixed(1)}`)
      .join("");
    return `${right}${left}Z`;
  };
  const shown = draft ? ([Math.min(draft.from, draft.to), Math.max(draft.from, draft.to)] as const) : range;
  const hoverValue = hovered !== null && values ? values[hovered] : NaN;
  const format = (v: number) => formatEndpoint(endpoint, v);

  function local(event: PointerEvent<SVGSVGElement>) {
    return event.clientY - event.currentTarget.getBoundingClientRect().top;
  }

  return (
    <div style={{ width }} className="shrink-0">
      <div className="relative" style={{ height: PLOT }}>
        <svg
          width={width}
          height={PLOT}
          className="block cursor-ns-resize touch-none select-none"
          role="img"
          aria-label={`Distribution of ${heading(endpoint)}${range ? `, ${format(range[0])} to ${format(range[1])} kept` : ""}`}
          onPointerDown={(event) => {
            event.currentTarget.setPointerCapture(event.pointerId);
            const v = valueAt(local(event));
            setDraft({ from: v, to: v });
          }}
          onPointerMove={(event) => {
            if (draft) setDraft({ ...draft, to: valueAt(local(event)) });
          }}
          onPointerUp={(event) => {
            if (!draft) return;
            const to = valueAt(local(event));
            setDraft(null);
            if (Math.abs(yValue(to) - yValue(draft.from)) < 4) onRange(endpoint.key, null);
            else onRange(endpoint.key, [Math.min(draft.from, to), Math.max(draft.from, to)]);
          }}
          onPointerCancel={() => setDraft(null)}
        >
          <title>Drag to keep a range · click to clear</title>
          <text x={cx} y={TOP - 4} textAnchor="middle" fontSize={9} fill="var(--text-faint)">
            {format(axis.from(hi))}
          </text>
          <text x={cx} y={TOP + plotH + 11} textAnchor="middle" fontSize={9} fill="var(--text-faint)">
            {flat ? "" : format(axis.from(lo))}
          </text>
          {flat ? (
            <line x1={cx - half} x2={cx + half} y1={y(lo)} y2={y(lo)} stroke={colour} strokeOpacity={0.6} strokeWidth={1.5} />
          ) : (
            <>
              <path d={shape(full.grid)} fill={colour} fillOpacity={subset ? 0.05 : 0.1} stroke={colour} strokeOpacity={0.5} strokeWidth={1} />
              {subset && subset.count > 0 && <path d={shape(subset.grid)} fill={colour} fillOpacity={0.35} />}
              {full.quartiles && (
                <>
                  <line x1={cx} x2={cx} y1={y(full.quartiles.q3)} y2={y(full.quartiles.q1)} stroke="var(--text-body)" strokeOpacity={0.4} strokeWidth={3} strokeLinecap="round" />
                  <circle cx={cx} cy={y(full.quartiles.median)} r={2.2} fill="var(--surface-raised)" stroke="var(--text-body)" strokeOpacity={0.7} />
                </>
              )}
            </>
          )}
          {Number.isFinite(hoverValue) && <line x1={cx - half} x2={cx + half} y1={yValue(hoverValue)} y2={yValue(hoverValue)} stroke="var(--text-body)" strokeWidth={1.25} />}
          {shown && (
            <g pointerEvents="none">
              <rect
                x={3}
                y={yValue(shown[1])}
                width={width - 6}
                height={Math.max(1, yValue(shown[0]) - yValue(shown[1]))}
                rx={3}
                fill="var(--color-accent)"
                fillOpacity={0.06}
                stroke="var(--color-accent)"
                strokeOpacity={0.6}
                strokeDasharray="3 2"
              />
              <text x={width - 5} y={yValue(shown[1]) - 2} textAnchor="end" fontSize={9} fill="var(--color-accent)">
                {format(shown[1])}
              </text>
              <text x={width - 5} y={yValue(shown[0]) + 9} textAnchor="end" fontSize={9} fill="var(--color-accent)">
                {format(shown[0])}
              </text>
            </g>
          )}
        </svg>
        <canvas ref={canvas} aria-hidden="true" className="pointer-events-none absolute inset-0" style={{ width, height: PLOT, color: colour }} />
      </div>
      <Label endpoint={endpoint} />
    </div>
  );
});

function Label({ endpoint }: { endpoint: Endpoint }) {
  const scaleNote = endpoint.scale === "linear" ? "" : " · log scale";
  return (
    <div className="h-[26px] px-1 text-center leading-tight" title={`${endpoint.key}${endpoint.source === "score" ? " · score (0–1): the CSV has no raw value" : ""}${scaleNote}`}>
      <div className="text-muted truncate text-[10.5px]">{endpoint.label}</div>
      <div className="text-faint truncate text-[9.5px]">
        {endpoint.unit || "\u00a0"}
        {scaleNote && <span className="opacity-70"> · log</span>}
      </div>
    </div>
  );
}
