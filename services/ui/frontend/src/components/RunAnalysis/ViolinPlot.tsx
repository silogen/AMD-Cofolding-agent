// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { memo, useMemo, useState, type PointerEvent } from "react";
import { bandwidth, extent, finiteValues, kde, quartiles, type Endpoint, type Molecules, type Ranges } from "../../lib/runAnalysis";
import { short, useWidth } from "./plotUtils";
import { Placeholder } from "./shared";

interface Props {
  molecules: Molecules;
  endpoints: readonly Endpoint[];
  shown: readonly string[];
  mask: Uint8Array | null;
  ranges: Ranges;
  onRange: (key: string, range: [number, number] | null) => void;
  colours: ReadonlyMap<string, string>;
  hovered: number | null;
}

const HEIGHT = 230;
const TOP = 14;
const BOTTOM = 34;
const POINTS = 48;

/**
 * One violin per endpoint, each on its own scale. Drag up or down on one to
 * keep a range of its values, click it to let go; ranges on several endpoints
 * combine. The molecules selected anywhere show as a solid overlay.
 */
export function ViolinPlot({ molecules, endpoints, shown, mask, ranges, onRange, colours, hovered }: Props) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const list = shown.map((key) => endpoints.find((e) => e.key === key)).filter((e): e is Endpoint => !!e);
  const columnWidth = list.length ? Math.max(64, Math.min(130, Math.floor(width / list.length))) : 0;
  return (
    <div>
      <div ref={ref} className="overflow-x-auto">
        {!list.length ? (
          <Placeholder height={HEIGHT}>Pick endpoints to compare.</Placeholder>
        ) : (
          <div className="flex" style={{ width: Math.max(width, columnWidth * list.length) }}>
            {list.map((e) => (
              <Violin
                key={e.key}
                endpoint={e}
                values={molecules.values.get(e.key) ?? null}
                width={columnWidth}
                colour={colours.get(e.key) ?? "var(--text-muted)"}
                mask={mask}
                range={ranges[e.key] ?? null}
                onRange={onRange}
                hovered={hovered}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

interface ViolinProps {
  endpoint: Endpoint;
  values: Float64Array | null;
  width: number;
  colour: string;
  mask: Uint8Array | null;
  range: readonly [number, number] | null;
  onRange: (key: string, range: [number, number] | null) => void;
  hovered: number | null;
}

const Violin = memo(function Violin({ endpoint, values, width, colour, mask, range, onRange, hovered }: ViolinProps) {
  const [draft, setDraft] = useState<{ from: number; to: number } | null>(null);
  const full = useMemo(() => {
    if (!values) return null;
    const all = finiteValues(values);
    const ext = extent(all);
    if (!ext) return null;
    const [lo, hi] = ext;
    const h = bandwidth(all, lo, hi);
    const raw = kde(all, lo, hi, h, POINTS, 1);
    const peak = Math.max(...raw) || 1;
    return { lo, hi, h, peak, grid: raw.map((v) => v / peak), quartiles: quartiles(all), count: all.length };
  }, [values]);
  const subset = useMemo(() => {
    if (!values || !full || !mask) return null;
    const picked = finiteValues(values, mask);
    return { grid: kde(picked, full.lo, full.hi, full.h, POINTS, full.peak), count: picked.length };
  }, [values, full, mask]);

  const plotH = HEIGHT - TOP - BOTTOM;
  if (!full) {
    return (
      <div style={{ width }} className="shrink-0">
        <Placeholder height={HEIGHT - 20}>no values</Placeholder>
        <Label endpoint={endpoint} />
      </div>
    );
  }
  const { lo, hi } = full;
  const flat = hi === lo;
  const y = (v: number) => (flat ? TOP + plotH / 2 : TOP + (1 - (v - lo) / (hi - lo)) * plotH);
  const valueAt = (py: number) => (flat ? lo : Math.min(hi, Math.max(lo, lo + (1 - (py - TOP) / plotH) * (hi - lo))));
  const cx = width / 2;
  const half = width * 0.4;
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

  function local(event: PointerEvent<SVGSVGElement>) {
    return event.clientY - event.currentTarget.getBoundingClientRect().top;
  }

  return (
    <div style={{ width }} className="shrink-0">
      <svg
        width={width}
        height={HEIGHT - 20}
        className="block cursor-ns-resize touch-none select-none"
        role="img"
        aria-label={`Distribution of ${endpoint.label}${range ? `, ${short(range[0])} to ${short(range[1])} kept` : ""}`}
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
          if (Math.abs(y(to) - y(draft.from)) < 4) onRange(endpoint.key, null);
          else onRange(endpoint.key, [Math.min(draft.from, to), Math.max(draft.from, to)]);
        }}
        onPointerCancel={() => setDraft(null)}
      >
        <title>Drag to keep a range · click to clear</title>
        <text x={cx} y={TOP - 4} textAnchor="middle" fontSize={9} fill="var(--text-faint)">
          {short(hi)}
        </text>
        <text x={cx} y={TOP + plotH + 11} textAnchor="middle" fontSize={9} fill="var(--text-faint)">
          {flat ? "" : short(lo)}
        </text>
        {flat ? (
          <line x1={cx - half} x2={cx + half} y1={y(lo)} y2={y(lo)} stroke={colour} strokeWidth={2} />
        ) : (
          <>
            <path d={shape(full.grid)} fill={colour} fillOpacity={subset ? 0.1 : 0.2} stroke={colour} strokeOpacity={0.6} strokeWidth={1} />
            {subset && subset.count > 0 && <path d={shape(subset.grid)} fill={colour} fillOpacity={0.6} />}
            {full.quartiles && (
              <>
                <line x1={cx} x2={cx} y1={y(full.quartiles.q3)} y2={y(full.quartiles.q1)} stroke="var(--text-body)" strokeOpacity={0.45} strokeWidth={3} strokeLinecap="round" />
                <circle cx={cx} cy={y(full.quartiles.median)} r={2.2} fill="var(--surface-raised)" stroke="var(--text-body)" strokeOpacity={0.7} />
              </>
            )}
          </>
        )}
        {Number.isFinite(hoverValue) && <line x1={cx - half} x2={cx + half} y1={y(hoverValue)} y2={y(hoverValue)} stroke="var(--text-body)" strokeWidth={1.25} />}
        {shown && (
          <g pointerEvents="none">
            <rect x={3} y={y(shown[1])} width={width - 6} height={Math.max(1, y(shown[0]) - y(shown[1]))} rx={3} fill="var(--color-accent)" fillOpacity={0.07} stroke="var(--color-accent)" strokeOpacity={0.6} strokeDasharray="3 2" />
            <text x={width - 5} y={y(shown[1]) - 2} textAnchor="end" fontSize={9} fill="var(--color-accent)">
              {short(shown[1])}
            </text>
            <text x={width - 5} y={y(shown[0]) + 9} textAnchor="end" fontSize={9} fill="var(--color-accent)">
              {short(shown[0])}
            </text>
          </g>
        )}
      </svg>
      <Label endpoint={endpoint} />
    </div>
  );
});

function Label({ endpoint }: { endpoint: Endpoint }) {
  return (
    <div className="text-muted h-5 truncate px-1 text-center text-[10.5px]" title={endpoint.key}>
      {endpoint.label}
    </div>
  );
}
