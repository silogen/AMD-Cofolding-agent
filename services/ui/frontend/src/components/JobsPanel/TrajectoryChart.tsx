// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useRef, useState, type PointerEvent } from "react";
import { chartSeries, fmt, linePath, linear, runningBest, scoreDomain, stepTicks, stepsDone, ticks, trajectoryPoints, withBest, type ChartSeries, type JobStatus } from "../../lib/jobs";

interface Props {
  status: JobStatus;
  /** Best total per step, from the results, for runs whose status reports none. */
  best?: ReadonlyMap<number, number> | null;
}

const WIDTH = 640;
const HEIGHT = 220;
const PAD = { left: 36, right: 12, top: 12, bottom: 24 };

const MEAN = "var(--color-structure)";
const BEST = "var(--color-molecule)";
const COLOURS = [
  "var(--color-table)",
  "var(--color-ok)",
  "oklch(0.7 0.12 200)",
  "oklch(0.68 0.16 340)",
  "oklch(0.76 0.13 95)",
  "oklch(0.62 0.1 30)",
  "oklch(0.6 0.12 280)",
  "oklch(0.7 0.1 150)",
];
const TOTAL = "total";

/**
 * Total score per step, its mean and the best so far, with one line per
 * objective of the TTP; chips pick what is drawn, and a Boltz confidence or
 * two hide under More. A plateau the diagnostics detected is shaded from the
 * step it started.
 */
export function TrajectoryChart({ status, best = null }: Props) {
  const [off, setOff] = useState<Set<string>>(() => new Set());
  const [extra, setExtra] = useState<Set<string>>(() => new Set());
  const [focus, setFocus] = useState<string | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  const { points: raw } = useMemo(() => trajectoryPoints(status), [status]);
  const points = useMemo(() => withBest(raw, best), [raw, best]);
  const { objectives, more } = useMemo(() => chartSeries(status), [status]);
  const colour = useMemo(() => {
    const map = new Map<string, string>();
    [...objectives, ...more].forEach((series, i) => map.set(series.key, COLOURS[i % COLOURS.length]));
    return map;
  }, [objectives, more]);
  const showTotal = !off.has(TOTAL);
  const drawn = useMemo(() => [...objectives.filter((s) => !off.has(s.key)), ...more.filter((s) => extra.has(s.key))], [objectives, more, off, extra]);
  const hasBest = points.some((p) => p.max !== null);

  const geometry = useMemo(() => {
    const values = showTotal ? points.flatMap((p) => (p.max === null ? [p.mean] : [p.mean, p.max])) : [];
    for (const series of drawn) values.push(...(status.componentTrajectories[series.key] ?? []).map((p) => p.mean));
    const [lo, hi] = scoreDomain(values);
    // stepsDone, not currentStep: the status reports the next step to run, one past the axis.
    const lastStep = Math.max(1, points.at(-1)?.step ?? 0, stepsDone(status));
    const x = linear([0, lastStep], [PAD.left, WIDTH - PAD.right]);
    const y = linear([lo, hi], [HEIGHT - PAD.bottom, PAD.top]);
    return { lo, hi, lastStep, x, y };
  }, [points, showTotal, drawn, status]);

  if (!points.length) {
    return (
      <div className="text-faint grid h-40 place-items-center text-xs">
        {status.state === "queued" || status.state === "running" ? "Waiting for the first scored step…" : "No score trajectory recorded."}
      </div>
    );
  }

  const { lo, hi, lastStep, x, y } = geometry;
  const mean = linePath(points.map((p) => ({ x: p.step, y: p.mean })), x, y);
  const bestSoFar = linePath(runningBest(points), x, y);
  const plateauFrom = status.diagnostics?.plateau ? status.diagnostics.plateauSince : null;
  const hovered = hover === null ? null : points[hover];
  const xTicks = stepTicks(lastStep, 6);
  const dim = (key: string) => (focus !== null && focus !== key ? 0.2 : 1);
  const valueAt = (key: string) => (hovered ? status.componentTrajectories[key]?.find((p) => p.step === hovered.step)?.mean : undefined);

  function onMove(event: PointerEvent<SVGSVGElement>) {
    const box = event.currentTarget.getBoundingClientRect();
    const step = ((event.clientX - box.left) / box.width) * WIDTH;
    let nearest = 0;
    for (let i = 1; i < points.length; i++) {
      if (Math.abs(x(points[i].step) - step) < Math.abs(x(points[nearest].step) - step)) nearest = i;
    }
    setHover(nearest);
  }

  const flip = (set: Set<string>, key: string) => {
    const next = new Set(set);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    return next;
  };

  return (
    <div>
      <div className="mb-1.5 flex flex-wrap items-center gap-1 text-[11px]">
        <Chip
          label="Total"
          colour={MEAN}
          on={showTotal}
          value={hovered && showTotal ? fmt(hovered.mean, 2) : undefined}
          onToggle={() => setOff((s) => flip(s, TOTAL))}
          onFocus={(on) => setFocus(on ? TOTAL : null)}
        />
        {objectives.map((series) => (
          <Chip
            key={series.key}
            label={series.label}
            colour={colour.get(series.key) ?? MEAN}
            on={!off.has(series.key)}
            value={hovered && !off.has(series.key) ? fmt(valueAt(series.key), 2) : undefined}
            onToggle={() => setOff((s) => flip(s, series.key))}
            onFocus={(on) => setFocus(on ? series.key : null)}
          />
        ))}
        {more.length > 0 && <More series={more} shown={extra} colour={colour} onToggle={(key) => setExtra((s) => flip(s, key))} />}
        {showTotal && (
          <span className="text-faint ml-auto hidden items-center gap-2 @md:inline-flex">
            <Swatch colour={MEAN} width={2} /> mean
            {hasBest && (
              <>
                <Swatch colour={BEST} width={1.5} /> best so far
              </>
            )}
          </span>
        )}
      </div>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="block h-auto w-full touch-none select-none"
        role="img"
        aria-label={`Score trajectory over ${points.length} steps`}
        onPointerMove={onMove}
        onPointerLeave={() => setHover(null)}
      >
        {plateauFrom !== null && (
          <rect
            x={x(plateauFrom)}
            y={PAD.top}
            width={Math.max(0, WIDTH - PAD.right - x(plateauFrom))}
            height={HEIGHT - PAD.top - PAD.bottom}
            fill={BEST}
            opacity={0.08}
          >
            <title>Plateau since step {plateauFrom}</title>
          </rect>
        )}
        {ticks(lo, hi, 5).map((t) => (
          <g key={t}>
            <line x1={PAD.left} x2={WIDTH - PAD.right} y1={y(t)} y2={y(t)} stroke="var(--border-subtle)" strokeWidth={1} />
            <text x={PAD.left - 6} y={y(t)} dy="0.32em" textAnchor="end" fontSize={10} fill="var(--text-faint)">
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={t} x={x(t)} y={HEIGHT - 6} textAnchor="middle" fontSize={10} fill="var(--text-faint)">
            {t}
          </text>
        ))}
        {drawn.map((series) => (
          <path
            key={series.key}
            d={linePath((status.componentTrajectories[series.key] ?? []).map((p) => ({ x: p.step, y: p.mean })), x, y)}
            fill="none"
            stroke={colour.get(series.key)}
            strokeWidth={focus === series.key ? 2 : 1.25}
            strokeDasharray={more.includes(series) ? "4 3" : undefined}
            opacity={0.85 * dim(series.key)}
            className="transition-opacity"
          />
        ))}
        {showTotal && (
          <g opacity={dim(TOTAL)} className="transition-opacity">
            {hasBest && <path d={bestSoFar} fill="none" stroke={BEST} strokeWidth={1.5} />}
            <path d={mean} fill="none" stroke={MEAN} strokeWidth={2.25} />
          </g>
        )}
        {hovered && (
          <g>
            <line x1={x(hovered.step)} x2={x(hovered.step)} y1={PAD.top} y2={HEIGHT - PAD.bottom} stroke="var(--border-strong)" strokeWidth={1} />
            {showTotal && <circle cx={x(hovered.step)} cy={y(hovered.mean)} r={3} fill={MEAN} />}
            {showTotal && hovered.max !== null && <circle cx={x(hovered.step)} cy={y(hovered.max)} r={2.5} fill={BEST} />}
          </g>
        )}
      </svg>
      <div className="text-muted h-4 text-[11px] tabular-nums">
        {hovered
          ? `Step ${hovered.step} · mean ${fmt(hovered.mean)}${hovered.max !== null ? ` · best ${fmt(hovered.max)}` : ""}${hovered.molecules !== null ? ` · ${hovered.molecules} molecules` : ""}`
          : `${points.length} step${points.length === 1 ? "" : "s"}`}
      </div>
    </div>
  );
}

function Chip({
  label,
  colour,
  on,
  value,
  onToggle,
  onFocus,
}: {
  label: string;
  colour: string;
  on: boolean;
  value?: string;
  onToggle: () => void;
  onFocus: (on: boolean) => void;
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      onPointerEnter={() => on && onFocus(true)}
      onPointerLeave={() => onFocus(false)}
      aria-pressed={on}
      className={`inline-flex cursor-pointer items-center gap-1 rounded-full border px-1.5 py-px ${on ? "border-subtle text-body bg-raised" : "text-faint hover:text-muted border-transparent"}`}
    >
      <span className="inline-block size-1.5 rounded-full" style={{ background: on ? colour : "var(--border-strong)" }} />
      {label}
      {value !== undefined && <span className="text-muted font-mono tabular-nums">{value}</span>}
    </button>
  );
}

function More({ series, shown, colour, onToggle }: { series: ChartSeries[]; shown: Set<string>; colour: Map<string, string>; onToggle: (key: string) => void }) {
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (event: globalThis.PointerEvent) => {
      if (!box.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", close);
    return () => document.removeEventListener("pointerdown", close);
  }, [open]);
  const count = series.filter((s) => shown.has(s.key)).length;
  return (
    <span ref={box} className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="text-faint hover:text-muted inline-flex cursor-pointer items-center gap-0.5 rounded-full px-1.5 py-px"
      >
        More{count > 0 && ` · ${count}`}
        <svg width={8} height={8} viewBox="0 0 10 10" aria-hidden="true">
          <path d="M2 3.5 5 6.5 8 3.5" fill="none" stroke="currentColor" strokeWidth={1.4} />
        </svg>
      </button>
      {open && (
        <span className="bg-raised border-subtle shadow-card absolute top-full left-0 z-20 mt-1 flex w-max flex-col gap-0.5 rounded-lg border p-1">
          {series.map((s) => (
            <label key={s.key} className="hover:bg-inset flex cursor-pointer items-center gap-1.5 rounded px-1.5 py-0.5" title={s.key}>
              <input type="checkbox" checked={shown.has(s.key)} onChange={() => onToggle(s.key)} className="size-3" />
              <Swatch colour={colour.get(s.key) ?? MEAN} width={1.5} dashed />
              {s.label}
            </label>
          ))}
        </span>
      )}
    </span>
  );
}

function Swatch({ colour, width, dashed }: { colour: string; width: number; dashed?: boolean }) {
  return (
    <svg width={14} height={6} aria-hidden="true">
      <line x1={0} x2={14} y1={3} y2={3} stroke={colour} strokeWidth={width} strokeDasharray={dashed ? "3 2" : undefined} />
    </svg>
  );
}
