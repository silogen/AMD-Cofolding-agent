// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useMemo, useState, type PointerEvent } from "react";
import { linear, scoreDomain, stepTicks, ticks } from "../../lib/jobs";
import { smooth, smoothWindow, type Endpoint, type StepStats } from "../../lib/runAnalysis";
import { short, useWidth } from "./plotUtils";
import { Chip, Placeholder } from "./shared";

interface Props {
  /** Total and components: the lines that can be drawn. */
  endpoints: readonly Endpoint[];
  stats: ReadonlyMap<string, StepStats[]>;
  shown: ReadonlySet<string>;
  onToggle: (key: string) => void;
  colours: ReadonlyMap<string, string>;
  live: boolean;
}

const HEIGHT = 190;
const PAD = { left: 30, right: 8, top: 8, bottom: 20 };

/**
 * Mean of each endpoint per step, smoothed, with the raw means faint behind;
 * hovering a legend entry brings its line forward with its min–max band.
 */
export function TrajectoryPlot({ endpoints, stats, shown, onToggle, colours, live }: Props) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [focus, setFocus] = useState<string | null>(null);
  const [smoothOn, setSmooth] = useState(true);
  const [band, setBand] = useState(false);
  const [hover, setHover] = useState<number | null>(null);

  const lines = useMemo(() => endpoints.filter((e) => stats.has(e.key)), [endpoints, stats]);
  const drawn = useMemo(() => lines.filter((e) => shown.has(e.key)), [lines, shown]);
  const steps = useMemo(() => {
    const all = new Set<number>();
    for (const e of drawn) for (const p of stats.get(e.key) ?? []) all.add(p.step);
    return [...all].sort((a, b) => a - b);
  }, [drawn, stats]);
  const count = Math.max(0, ...lines.map((e) => stats.get(e.key)?.length ?? 0));
  const span = smoothOn ? smoothWindow(count) : 1;
  const banded = focus ?? (band ? (drawn[0]?.key ?? null) : null);

  const w = Math.max(200, width);
  const geometry = useMemo(() => {
    const values: number[] = [];
    for (const e of drawn) for (const p of stats.get(e.key) ?? []) values.push(p.mean);
    if (banded) for (const p of stats.get(banded) ?? []) values.push(p.min, p.max);
    const [lo, hi] = scoreDomain(values);
    const first = steps[0] ?? 0;
    const last = Math.max(first + 1, steps.at(-1) ?? 1);
    return { lo, hi, first, last, x: linear([first, last], [PAD.left, w - PAD.right]), y: linear([lo, hi], [HEIGHT - PAD.bottom, PAD.top]) };
  }, [drawn, stats, banded, steps, w]);

  const paths = useMemo(() => {
    const { x, y } = geometry;
    const line = (points: { step: number; v: number }[]) =>
      points
        .filter((p) => Number.isFinite(p.v))
        .map((p, i) => `${i ? "L" : "M"}${x(p.step).toFixed(1)},${y(p.v).toFixed(1)}`)
        .join("");
    return drawn.map((e) => {
      const series = stats.get(e.key) ?? [];
      const smoothed = smooth(
        series.map((p) => p.mean),
        span,
      );
      return {
        key: e.key,
        raw: span > 1 ? line(series.map((p) => ({ step: p.step, v: p.mean }))) : "",
        main: line(series.map((p, i) => ({ step: p.step, v: smoothed[i] }))),
        band:
          banded === e.key && series.length > 1
            ? `${series.map((p, i) => `${i ? "L" : "M"}${x(p.step).toFixed(1)},${y(p.max).toFixed(1)}`).join("")}${[...series]
                .reverse()
                .map((p) => `L${x(p.step).toFixed(1)},${y(p.min).toFixed(1)}`)
                .join("")}Z`
            : "",
      };
    });
  }, [drawn, stats, geometry, span, banded]);

  if (count < 2) {
    return <Placeholder>{count === 1 ? "One step so far: the trajectory starts at the second." : live ? "Waiting for the first steps…" : "No steps recorded."}</Placeholder>;
  }

  const { lo, hi, first, last, x, y } = geometry;
  const hovered = hover === null ? null : steps[hover];

  function onMove(event: PointerEvent<SVGSVGElement>) {
    if (!steps.length) return;
    const box = event.currentTarget.getBoundingClientRect();
    const px = event.clientX - box.left;
    let nearest = 0;
    for (let i = 1; i < steps.length; i++) if (Math.abs(x(steps[i]) - px) < Math.abs(x(steps[nearest]) - px)) nearest = i;
    setHover(nearest);
  }

  const valueAt = (key: string, step: number) => stats.get(key)?.find((p) => p.step === step) ?? null;
  const yTicks = ticks(lo, hi, hi - lo <= 1.0001 && lo === 0 ? 3 : 4);
  const dim = (key: string) => (focus && focus !== key ? 0.18 : 1);

  return (
    <div>
      <div className="mb-1 flex flex-wrap items-center gap-x-0.5 gap-y-0.5">
        {lines.map((e) => (
          <Chip
            key={e.key}
            label={e.label}
            colour={colours.get(e.key) ?? "var(--text-muted)"}
            on={shown.has(e.key)}
            onClick={() => onToggle(e.key)}
            onHover={(on) => setFocus(on && shown.has(e.key) ? e.key : null)}
          />
        ))}
        <span className="ml-auto flex items-center gap-0.5">
          <Toggle on={smoothOn} onClick={() => setSmooth((v) => !v)} title="Moving average over a few steps; raw means stay faint behind">
            Smooth
          </Toggle>
          <Toggle on={band} onClick={() => setBand((v) => !v)} title="Min–max range per step (hover a legend entry for its own)">
            Range
          </Toggle>
        </span>
      </div>
      <div ref={ref} className="relative">
        {width > 0 && (
          <svg
            width={w}
            height={HEIGHT}
            className="block touch-none select-none"
            role="img"
            aria-label={`Score trajectory over ${count} steps`}
            onPointerMove={onMove}
            onPointerLeave={() => setHover(null)}
          >
            {yTicks.map((t) => (
              <g key={t}>
                <line x1={PAD.left} x2={w - PAD.right} y1={y(t)} y2={y(t)} stroke="var(--border-subtle)" />
                <text x={PAD.left - 5} y={y(t)} dy="0.32em" textAnchor="end" fontSize={9.5} fill="var(--text-faint)">
                  {short(t)}
                </text>
              </g>
            ))}
            {stepTicks(last, Math.max(3, Math.min(8, Math.floor(w / 80))))
              .filter((t) => t >= first)
              .map((t) => (
                <text key={t} x={x(t)} y={HEIGHT - 5} textAnchor="middle" fontSize={9.5} fill="var(--text-faint)">
                  {t}
                </text>
              ))}
            {paths.map((p) => {
              const colour = colours.get(p.key) ?? "var(--text-muted)";
              const total = p.key === drawn.find((e) => e.kind === "total")?.key;
              return (
                <g key={p.key} opacity={dim(p.key)} className="transition-opacity">
                  {p.band && <path d={p.band} fill={colour} opacity={0.12} />}
                  {p.raw && <path d={p.raw} fill="none" stroke={colour} strokeWidth={1} opacity={0.22} />}
                  <path d={p.main} fill="none" stroke={colour} strokeWidth={focus === p.key || total ? 2.25 : 1.5} strokeLinejoin="round" strokeLinecap="round" />
                </g>
              );
            })}
            {hovered !== null && (
              <g pointerEvents="none">
                <line x1={x(hovered)} x2={x(hovered)} y1={PAD.top} y2={HEIGHT - PAD.bottom} stroke="var(--border-strong)" />
                {drawn.map((e) => {
                  const v = valueAt(e.key, hovered);
                  return v ? <circle key={e.key} cx={x(hovered)} cy={y(v.mean)} r={2.75} fill={colours.get(e.key)} stroke="var(--surface-raised)" strokeWidth={1} /> : null;
                })}
              </g>
            )}
          </svg>
        )}
        {hovered !== null && (
          <div
            className="bg-raised border-subtle shadow-card pointer-events-none absolute top-1 z-10 rounded-md border px-2 py-1 text-[10.5px] tabular-nums"
            style={x(hovered) > w * 0.6 ? { right: w - x(hovered) + 10 } : { left: x(hovered) + 10 }}
          >
            <div className="text-faint mb-0.5">Step {hovered}</div>
            {drawn.map((e) => {
              const v = valueAt(e.key, hovered);
              return (
                <div key={e.key} className="flex items-center gap-1.5 whitespace-nowrap">
                  <span className="inline-block size-1.5 rounded-full" style={{ background: colours.get(e.key) }} />
                  <span className="text-muted max-w-32 truncate">{e.label}</span>
                  <span className="ml-auto pl-2 font-mono">{v ? short(v.mean) : "—"}</span>
                  {v && banded === e.key && (
                    <span className="text-faint font-mono">
                      {short(v.min)}–{short(v.max)}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

function Toggle({ on, onClick, title, children }: { on: boolean; onClick: () => void; title: string; children: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={on}
      title={title}
      className={`cursor-pointer rounded px-1.5 py-0.5 text-[10.5px] ${on ? "bg-inset text-body" : "text-faint hover:text-muted"}`}
    >
      {children}
    </button>
  );
}
