// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useMemo, useState, type PointerEvent } from "react";
import { fmt, linePath, linear, runningBest, scoreDomain, ticks, trajectoryPoints, type JobStatus } from "../../lib/jobs";

interface Props {
  status: JobStatus;
}

const WIDTH = 640;
const HEIGHT = 220;
const PAD = { left: 36, right: 12, top: 12, bottom: 24 };

const MEAN = "var(--color-structure)";
const BEST = "var(--color-molecule)";
const COMPONENT_COLOURS = [
  "var(--color-table)",
  "var(--color-ok)",
  "oklch(0.7 0.12 200)",
  "oklch(0.68 0.16 340)",
  "oklch(0.76 0.13 95)",
  "oklch(0.62 0.1 30)",
];

/**
 * Total score per step: the batch mean, the best molecule of each step and
 * the best so far, with the scoring components on demand. A plateau the
 * diagnostics detected is shaded from the step it started.
 */
export function TrajectoryChart({ status }: Props) {
  const [shown, setShown] = useState<Set<string>>(() => new Set());
  const [hover, setHover] = useState<number | null>(null);
  const { points, from } = useMemo(() => trajectoryPoints(status), [status]);
  const componentNames = Object.keys(status.componentTrajectories).filter((name) => name !== from);
  const hasBest = points.some((p) => p.max !== null);

  const geometry = useMemo(() => {
    const values = points.flatMap((p) => (p.max === null ? [p.mean] : [p.mean, p.max]));
    for (const name of shown) values.push(...(status.componentTrajectories[name] ?? []).map((p) => p.mean));
    const [lo, hi] = scoreDomain(values);
    const lastStep = Math.max(1, points.at(-1)?.step ?? 0, status.currentStep ?? 0);
    const x = linear([0, lastStep], [PAD.left, WIDTH - PAD.right]);
    const y = linear([lo, hi], [HEIGHT - PAD.bottom, PAD.top]);
    return { lo, hi, lastStep, x, y };
  }, [points, shown, status.componentTrajectories, status.currentStep]);

  if (!points.length) {
    return (
      <div className="text-faint grid h-40 place-items-center text-xs">
        {status.state === "queued" || status.state === "running" ? "Waiting for the first scored step…" : "No score trajectory recorded."}
      </div>
    );
  }

  const { lo, hi, lastStep, x, y } = geometry;
  const mean = linePath(points.map((p) => ({ x: p.step, y: p.mean })), x, y);
  const stepMax = linePath(points.filter((p) => p.max !== null).map((p) => ({ x: p.step, y: p.max as number })), x, y);
  const best = linePath(runningBest(points), x, y);
  const plateauFrom = status.diagnostics?.plateau ? status.diagnostics.plateauSince : null;
  const hovered = hover === null ? null : points[hover];
  const xTicks = ticks(0, lastStep, 6).map((t) => Math.round(t));

  function onMove(event: PointerEvent<SVGSVGElement>) {
    const box = event.currentTarget.getBoundingClientRect();
    const step = ((event.clientX - box.left) / box.width) * WIDTH;
    let nearest = 0;
    for (let i = 1; i < points.length; i++) {
      if (Math.abs(x(points[i].step) - step) < Math.abs(x(points[nearest].step) - step)) nearest = i;
    }
    setHover(nearest);
  }

  function toggle(name: string) {
    setShown((current) => {
      const next = new Set(current);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }

  return (
    <div>
      <div className="mb-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px]">
        <Legend colour={MEAN} label="mean total" />
        {hasBest && <Legend colour={BEST} label="best of step" dashed />}
        {hasBest && <Legend colour={BEST} label="best so far" />}
        {componentNames.map((name, i) => (
          <button
            key={name}
            type="button"
            onClick={() => toggle(name)}
            aria-pressed={shown.has(name)}
            className={`cursor-pointer rounded-full border px-1.5 py-px ${shown.has(name) ? "border-strong text-body" : "border-subtle text-faint hover:text-muted"}`}
          >
            <span className="mr-1 inline-block size-1.5 rounded-full align-middle" style={{ background: COMPONENT_COLOURS[i % COMPONENT_COLOURS.length] }} />
            {name}
          </button>
        ))}
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
        {xTicks.map((t, i) => (
          <text key={`${t}-${i}`} x={x(t)} y={HEIGHT - 6} textAnchor="middle" fontSize={10} fill="var(--text-faint)">
            {t}
          </text>
        ))}
        {[...shown].map((name) => {
          const series = status.componentTrajectories[name] ?? [];
          const colour = COMPONENT_COLOURS[componentNames.indexOf(name) % COMPONENT_COLOURS.length];
          return (
            <path
              key={name}
              d={linePath(series.map((p) => ({ x: p.step, y: p.mean })), x, y)}
              fill="none"
              stroke={colour}
              strokeWidth={1.25}
              opacity={0.85}
            />
          );
        })}
        {hasBest && <path d={stepMax} fill="none" stroke={BEST} strokeWidth={1} strokeDasharray="3 3" opacity={0.7} />}
        {hasBest && <path d={best} fill="none" stroke={BEST} strokeWidth={1.75} />}
        <path d={mean} fill="none" stroke={MEAN} strokeWidth={2} />
        {hovered && (
          <g>
            <line
              x1={x(hovered.step)}
              x2={x(hovered.step)}
              y1={PAD.top}
              y2={HEIGHT - PAD.bottom}
              stroke="var(--border-strong)"
              strokeWidth={1}
            />
            <circle cx={x(hovered.step)} cy={y(hovered.mean)} r={3} fill={MEAN} />
            {hovered.max !== null && <circle cx={x(hovered.step)} cy={y(hovered.max)} r={3} fill={BEST} />}
          </g>
        )}
      </svg>
      <div className="text-muted h-4 text-[11px] tabular-nums">
        {hovered
          ? `Step ${hovered.step} · mean ${fmt(hovered.mean)}${hovered.max !== null ? ` · best ${fmt(hovered.max)}` : ""}${hovered.molecules !== null ? ` · ${hovered.molecules} molecules` : ""}`
          : `${points.length} sampled step${points.length === 1 ? "" : "s"}`}
      </div>
    </div>
  );
}

function Legend({ colour, label, dashed }: { colour: string; label: string; dashed?: boolean }) {
  return (
    <span className="text-muted inline-flex items-center gap-1">
      <svg width={14} height={6} aria-hidden="true">
        <line x1={0} x2={14} y1={3} y2={3} stroke={colour} strokeWidth={2} strokeDasharray={dashed ? "3 2" : undefined} />
      </svg>
      {label}
    </span>
  );
}
