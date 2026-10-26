// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useState } from "react";
import { describeRule, radarPoint, type RadarAxis, type RadarSeries } from "../../lib/radar";

interface Props {
  axes: RadarAxis[];
  series: RadarSeries[];
  /** Molecules with predictions, when more than the series shown. */
  total: number;
}

const COLOURS = ["var(--color-molecule)", "var(--color-structure)", "var(--color-table)", "var(--color-ok)", "var(--color-accent)"];
const WIDTH = 380;
const HEIGHT = 250;
const CX = WIDTH / 2;
const CY = HEIGHT / 2;
const RADIUS = 92;
const RINGS = [0.25, 0.5, 0.75, 1];

const fmt = (v: number | null) => (v === null ? "no value" : String(+v.toPrecision(3)));

/** One polygon per molecule over the endpoints scaled to 0-1, further out being better. */
export function RadarChart({ axes, series, total }: Props) {
  const [hidden, setHidden] = useState<Set<number>>(() => new Set());
  const n = axes.length;
  const ring = (r: number) =>
    axes
      .map((_, i) => radarPoint(i, n, r, CX, CY, RADIUS))
      .map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`)
      .join(" ");

  const toggle = (index: number) =>
    setHidden((previous) => {
      const next = new Set(previous);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });

  return (
    <div className="flex flex-wrap items-start gap-x-4 gap-y-2 px-3 py-2">
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="w-full max-w-[380px] shrink-0" role="img" aria-label="Radar of the predicted properties">
        {RINGS.map((r) => (
          <polygon key={r} points={ring(r)} fill="none" stroke="var(--border-strong)" strokeWidth={r === 1 ? 1 : 0.6} strokeDasharray={r === 1 ? undefined : "2 3"} />
        ))}
        {axes.map((axis, i) => {
          const [x, y] = radarPoint(i, n, 1, CX, CY, RADIUS);
          const [lx, ly] = radarPoint(i, n, 1.12, CX, CY, RADIUS);
          const anchor = Math.abs(lx - CX) < 4 ? "middle" : lx > CX ? "start" : "end";
          return (
            <g key={axis.endpoint}>
              <line x1={CX} y1={CY} x2={x} y2={y} stroke="var(--border-strong)" strokeWidth={0.6} />
              <text x={lx} y={ly} dy="0.35em" textAnchor={anchor} fontSize={10} fill="var(--text-muted)">
                <title>{`${axis.label}: ${describeRule(axis.rule)}`}</title>
                {axis.label}
                {axis.rule.relative ? "*" : ""}
              </text>
            </g>
          );
        })}
        {series.map((s, index) => {
          if (hidden.has(index)) return null;
          const colour = COLOURS[index % COLOURS.length];
          const points = s.scores.map((score, i) => radarPoint(i, n, score ?? 0, CX, CY, RADIUS));
          return (
            <g key={index}>
              <polygon
                points={points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ")}
                fill={colour}
                fillOpacity={0.08}
                stroke={colour}
                strokeWidth={1.5}
                strokeLinejoin="round"
              />
              {points.map(([x, y], i) => (
                <circle key={i} cx={x} cy={y} r={2.5} fill={s.scores[i] === null ? "var(--surface-raised)" : colour} stroke={colour}>
                  <title>{`${s.label} · ${axes[i].label}: ${fmt(s.raw[i])}${s.scores[i] === null ? "" : ` → ${s.scores[i]?.toFixed(2)}`}`}</title>
                </circle>
              ))}
            </g>
          );
        })}
        <text x={CX + 3} y={CY - RADIUS / 2} fontSize={8} fill="var(--text-faint)">
          0.5
        </text>
      </svg>

      <div className="min-w-40 flex-1 space-y-2 text-[11px]">
        <ul className="space-y-0.5">
          {series.map((s, index) => (
            <li key={index}>
              <button
                type="button"
                onClick={() => toggle(index)}
                aria-pressed={!hidden.has(index)}
                title={s.smiles ?? s.label}
                className={`hover:bg-inset flex w-full cursor-pointer items-center gap-2 rounded px-1 py-0.5 text-left ${hidden.has(index) ? "opacity-40" : ""}`}
              >
                <span aria-hidden="true" className="h-0.5 w-3 shrink-0 rounded-full" style={{ background: COLOURS[index % COLOURS.length] }} />
                <span className="min-w-0 truncate font-mono text-[10px]">{s.label}</span>
              </button>
            </li>
          ))}
        </ul>
        {total > series.length && <p className="text-faint text-[10px]">First {series.length} of {total} molecules.</p>}
        <details className="text-faint text-[10px]">
          <summary className="hover:text-body cursor-pointer">How the axes are scaled</summary>
          <p className="mt-1">Each endpoint is mapped to 0–1, further out being more favourable.</p>
          <dl className="mt-1 space-y-0.5">
            {axes.map((axis) => (
              <div key={axis.endpoint}>
                <dt className="text-muted inline font-medium">{axis.label}: </dt>
                <dd className="inline">{describeRule(axis.rule)}</dd>
              </div>
            ))}
          </dl>
        </details>
      </div>
    </div>
  );
}
