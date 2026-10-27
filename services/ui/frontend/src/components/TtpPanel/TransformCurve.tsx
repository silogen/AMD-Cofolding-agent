// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useState, type PointerEvent } from "react";
import { curveDomain, rangeFields, transformCurve, type Transform } from "../../lib/ttp";

interface Props {
  transform: Transform;
  width?: number;
  height?: number;
  /** The objective's weight relative to the largest one, 0-1: sets how strongly the curve is drawn and the bar on the right. */
  weight?: number;
  /** Without tick labels, for lists. */
  compact?: boolean;
  /** Drag the thresholds on the curve. */
  onRange?: (patch: { low?: number; high?: number }) => void;
}

const PAD = 6;
const BAR = 5;
const tick = (v: number) => String(+v.toPrecision(3));

type Drag = { key: "low" | "high"; from: number; to: number };

/** The score a transform gives each raw value, with the low/high window shaded. */
export function TransformCurve({ transform, width = 200, height = 56, weight = 1, compact = false, onRange }: Props) {
  const [drag, setDrag] = useState<Drag | null>(null);
  const domain = drag ?? curveDomain(transform);
  const points = transformCurve(transform, 64, domain);
  if (points.length === 0) {
    return (
      <span className="bg-sunken text-faint grid shrink-0 place-items-center rounded-md text-[10px]" style={{ width, height }}>
        no curve for {transform.type}
      </span>
    );
  }
  const { from, to } = domain;
  const right = width - PAD - BAR - 3;
  const sx = (x: number) => PAD + ((x - from) / (to - from || 1)) * (right - PAD);
  const sy = (y: number) => height - PAD - y * (height - 2 * PAD);
  const line = points.map((p) => `${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(" ");
  const area = `${sx(from).toFixed(1)},${sy(0)} ${line} ${sx(to).toFixed(1)},${sy(0)}`;
  const low = typeof transform.low === "number" ? transform.low : null;
  const fields = rangeFields(transform.type).map((f) => f.key);
  const high = fields.includes("high") && typeof transform.high === "number" ? transform.high : null;
  const handles = onRange ? ([low !== null && fields.includes("low") && "low", high !== null && "high"].filter(Boolean) as ("low" | "high")[]) : [];
  const w = Math.max(0, Math.min(1, weight));
  const labelsHeight = compact ? 0 : 10;

  const valueAt = (event: PointerEvent<SVGSVGElement>, d: { from: number; to: number }) => {
    const box = event.currentTarget.getBoundingClientRect();
    const x = ((event.clientX - box.left) / box.width) * width;
    const raw = d.from + ((x - PAD) / (right - PAD)) * (d.to - d.from);
    const step = 10 ** Math.floor(Math.log10(Math.abs(d.to - d.from) / 100 || 1));
    return +(Math.round(raw / step) * step).toPrecision(6);
  };

  const onDown = (event: PointerEvent<SVGSVGElement>) => {
    if (!onRange || handles.length === 0) return;
    const value = valueAt(event, domain);
    const key = handles.length === 1 || high === null || Math.abs(value - (low ?? 0)) <= Math.abs(value - high) ? handles[0] : "high";
    event.currentTarget.setPointerCapture(event.pointerId);
    setDrag({ key, from, to });
    move(key, value);
  };

  const move = (key: "low" | "high", value: number) => {
    if (!onRange) return;
    if (key === "low") onRange({ low: high !== null ? Math.min(value, high) : value });
    else onRange({ high: low !== null ? Math.max(value, low) : value });
  };

  return (
    <svg
      viewBox={`0 0 ${width} ${height + labelsHeight}`}
      width={width}
      height={height + labelsHeight}
      className={`shrink-0 select-none ${handles.length ? "cursor-ew-resize touch-none" : ""}`}
      role="img"
      aria-label={`${transform.type} transform`}
      onPointerDown={onDown}
      onPointerMove={(event) => drag && move(drag.key, valueAt(event, drag))}
      onPointerUp={() => setDrag(null)}
      onPointerCancel={() => setDrag(null)}
    >
      {low !== null && high !== null && (
        <rect x={sx(Math.min(low, high))} y={PAD} width={Math.max(1, Math.abs(sx(high) - sx(low)))} height={height - 2 * PAD} fill="var(--color-table)" opacity={0.1} />
      )}
      <path d={`M${PAD} ${PAD}V${height - PAD}H${right}`} fill="none" stroke="var(--border-strong)" />
      <polygon points={area} fill="var(--color-table)" opacity={0.06 + 0.2 * w} />
      <polyline points={line} fill="none" stroke="var(--color-table)" strokeWidth={compact ? 1.5 : 2} strokeLinejoin="round" opacity={0.35 + 0.65 * w} />
      {handles.map((key) => {
        const x = sx((key === "low" ? low : high) ?? 0);
        return (
          <g key={key} opacity={drag && drag.key !== key ? 0.5 : 1}>
            <line x1={x} x2={x} y1={PAD} y2={height - PAD} stroke="var(--color-table)" strokeDasharray="2 2" />
            <circle cx={x} cy={height - PAD} r={3.5} fill="var(--surface-raised)" stroke="var(--color-table)" strokeWidth={1.5} />
          </g>
        );
      })}
      <rect x={width - PAD - BAR} y={PAD} width={BAR} height={height - 2 * PAD} rx={1.5} fill="var(--surface-inset)">
        <title>{`weight ${Math.round(w * 100)}% of the largest`}</title>
      </rect>
      <rect x={width - PAD - BAR} y={PAD + (1 - w) * (height - 2 * PAD)} width={BAR} height={w * (height - 2 * PAD)} rx={1.5} fill="var(--color-table)" />
      {!compact && (
        <>
          <text x={PAD} y={height + 8} fontSize={8} fill="var(--text-faint)">
            {tick(from)}
          </text>
          <text x={right} y={height + 8} fontSize={8} fill="var(--text-faint)" textAnchor="end">
            {tick(to)}
          </text>
          {low !== null && (
            <text x={sx(low)} y={height + 8} fontSize={8} fill="var(--text-muted)" textAnchor="middle">
              {tick(low)}
            </text>
          )}
          {high !== null && high !== low && (
            <text x={sx(high)} y={height + 8} fontSize={8} fill="var(--text-muted)" textAnchor="middle">
              {tick(high)}
            </text>
          )}
        </>
      )}
    </svg>
  );
}
