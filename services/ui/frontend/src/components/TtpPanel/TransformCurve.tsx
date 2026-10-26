// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { transformCurve, type Transform } from "../../lib/ttp";

interface Props {
  transform: Transform;
  width?: number;
  height?: number;
}

const PAD = 6;
const tick = (v: number) => String(+v.toPrecision(3));

/** The score a transform gives each raw value, with the low/high window shaded. */
export function TransformCurve({ transform, width = 200, height = 56 }: Props) {
  const points = transformCurve(transform);
  if (points.length === 0) {
    return (
      <span className="bg-sunken text-faint grid shrink-0 place-items-center rounded-md text-[10px]" style={{ width, height }}>
        no curve for {transform.type}
      </span>
    );
  }
  const from = points[0].x;
  const to = points[points.length - 1].x;
  const sx = (x: number) => PAD + ((x - from) / (to - from || 1)) * (width - 2 * PAD);
  const sy = (y: number) => height - PAD - y * (height - 2 * PAD);
  const line = points.map((p) => `${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(" ");
  const low = typeof transform.low === "number" ? transform.low : null;
  const high = typeof transform.high === "number" ? transform.high : null;

  return (
    <svg viewBox={`0 0 ${width} ${height + 10}`} width={width} height={height + 10} className="shrink-0" role="img" aria-label={`${transform.type} transform`}>
      {low !== null && high !== null && (
        <rect
          x={sx(Math.min(low, high))}
          y={PAD}
          width={Math.max(1, Math.abs(sx(high) - sx(low)))}
          height={height - 2 * PAD}
          fill="var(--color-table)"
          opacity={0.12}
        />
      )}
      <path d={`M${PAD} ${PAD}V${height - PAD}H${width - PAD}`} fill="none" stroke="var(--border-strong)" />
      <polyline points={line} fill="none" stroke="var(--color-table)" strokeWidth={2} strokeLinejoin="round" />
      <text x={PAD} y={height + 8} fontSize={8} fill="var(--text-faint)">
        {tick(from)}
      </text>
      <text x={width - PAD} y={height + 8} fontSize={8} fill="var(--text-faint)" textAnchor="end">
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
    </svg>
  );
}
