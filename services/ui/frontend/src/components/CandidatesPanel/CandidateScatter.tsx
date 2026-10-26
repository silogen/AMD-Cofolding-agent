// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { type Candidate, candidateValue } from "../../lib/candidates";

interface Props {
  candidates: Candidate[];
  columns: { key: string; label: string }[];
  x: string;
  y: string;
  onAxes: (x: string, y: string) => void;
  selected: Set<string>;
  detail: string | null;
  onPick: (smiles: string) => void;
}

const WIDTH = 560;
const HEIGHT = 340;
const PAD = { left: 52, right: 16, top: 14, bottom: 40 };

const GATE_COLOR = { pass: "#3ccf91", fail: "#ff4d4f", unknown: "var(--text-faint)" } as const;

/** Round ticks over [lo, hi]. */
function ticks(lo: number, hi: number, count = 5): number[] {
  const span = hi - lo || 1;
  const raw = span / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((f) => f * magnitude).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) out.push(Number(t.toPrecision(10)));
  return out;
}

function domain(values: number[]): [number, number] {
  if (!values.length) return [0, 1];
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  if (lo === hi) {
    lo -= 0.5;
    hi += 0.5;
  }
  const pad = (hi - lo) * 0.06;
  return [lo - pad, hi + pad];
}

const tickText = (value: number) => (Math.abs(value) >= 100 ? value.toFixed(0) : String(Number(value.toPrecision(3))));

/**
 * Two numeric columns against each other, each point coloured by the
 * confidence gate, so a high score on an unconfident pose stands out.
 */
export function CandidateScatter({ candidates, columns, x, y, onAxes, selected, detail, onPick }: Props) {
  const points = candidates
    .map((c) => ({ c, vx: candidateValue(c, x), vy: candidateValue(c, y) }))
    .filter((p): p is { c: Candidate; vx: number; vy: number } => p.vx !== undefined && p.vy !== undefined);
  const missing = candidates.length - points.length;

  const [x0, x1] = domain(points.map((p) => p.vx));
  const [y0, y1] = domain(points.map((p) => p.vy));
  const sx = (v: number) => PAD.left + ((v - x0) / (x1 - x0)) * (WIDTH - PAD.left - PAD.right);
  const sy = (v: number) => HEIGHT - PAD.bottom - ((v - y0) / (y1 - y0)) * (HEIGHT - PAD.top - PAD.bottom);
  const label = (key: string) => columns.find((c) => c.key === key)?.label ?? key;

  const select = "bg-sunken border-subtle rounded-md border px-2 py-1 text-xs";

  if (columns.length < 2) {
    return (
      <p className="text-muted px-3 py-8 text-center text-xs">
        A scatter plot needs two numeric columns; these candidates have {columns.length === 1 ? "one" : "none"}.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-2 p-3">
      <div className="flex flex-wrap items-center gap-2 text-[11px]">
        <span className="text-muted">X</span>
        <select aria-label="X axis" className={select} value={x} onChange={(e) => onAxes(e.target.value, y)}>
          {columns.map((c) => (
            <option key={c.key} value={c.key}>
              {c.label}
            </option>
          ))}
        </select>
        <span className="text-muted">Y</span>
        <select aria-label="Y axis" className={select} value={y} onChange={(e) => onAxes(x, e.target.value)}>
          {columns.map((c) => (
            <option key={c.key} value={c.key}>
              {c.label}
            </option>
          ))}
        </select>
        <span className="text-faint ml-auto flex items-center gap-3">
          {(["pass", "fail", "unknown"] as const).map((status) => (
            <span key={status} className="inline-flex items-center gap-1">
              <span aria-hidden="true" className="size-2 rounded-full" style={{ background: GATE_COLOR[status] }} />
              {status === "unknown" ? "not gated" : `gate ${status}`}
            </span>
          ))}
        </span>
      </div>

      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="bg-sunken border-subtle w-full rounded-lg border"
        role="img"
        aria-label={`${label(y)} against ${label(x)} for ${points.length} candidates`}
      >
        {ticks(x0, x1).map((t) => (
          <g key={`x${t}`}>
            <line x1={sx(t)} x2={sx(t)} y1={PAD.top} y2={HEIGHT - PAD.bottom} stroke="var(--border-subtle)" />
            <text x={sx(t)} y={HEIGHT - PAD.bottom + 14} textAnchor="middle" fontSize={10} fill="var(--text-faint)">
              {tickText(t)}
            </text>
          </g>
        ))}
        {ticks(y0, y1).map((t) => (
          <g key={`y${t}`}>
            <line x1={PAD.left} x2={WIDTH - PAD.right} y1={sy(t)} y2={sy(t)} stroke="var(--border-subtle)" />
            <text x={PAD.left - 6} y={sy(t) + 3} textAnchor="end" fontSize={10} fill="var(--text-faint)">
              {tickText(t)}
            </text>
          </g>
        ))}
        <text x={(PAD.left + WIDTH - PAD.right) / 2} y={HEIGHT - 6} textAnchor="middle" fontSize={11} fill="var(--text-muted)">
          {label(x)}
        </text>
        <text
          transform={`translate(12 ${(PAD.top + HEIGHT - PAD.bottom) / 2}) rotate(-90)`}
          textAnchor="middle"
          fontSize={11}
          fill="var(--text-muted)"
        >
          {label(y)}
        </text>
        {points.map(({ c, vx, vy }) => {
          const active = c.smiles === detail;
          return (
            <circle
              key={c.smiles}
              cx={sx(vx)}
              cy={sy(vy)}
              r={active ? 7 : 5.5}
              fill={GATE_COLOR[c.gate.status]}
              fillOpacity={0.85}
              stroke={active || selected.has(c.smiles) ? "var(--color-accent)" : "var(--surface-raised)"}
              strokeWidth={active ? 2.5 : 1.5}
              className="cursor-pointer"
              onClick={() => onPick(c.smiles)}
            >
              <title>
                {`${c.name ?? c.smiles}\n${label(x)} ${vx.toFixed(3)} · ${label(y)} ${vy.toFixed(3)}\ngate ${c.gate.status}`}
              </title>
            </circle>
          );
        })}
      </svg>
      {missing > 0 && (
        <p className="text-faint text-[11px]">
          {missing} candidate{missing === 1 ? " has" : "s have"} no value on one of these axes and {missing === 1 ? "is" : "are"} not
          plotted.
        </p>
      )}
    </div>
  );
}
