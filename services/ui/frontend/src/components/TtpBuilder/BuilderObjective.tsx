// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useState } from "react";
import { CURVE_LABELS, TRANSFORM_TYPES, objectiveSource, rangeFields, retypeObjective, setRange, type Objective, type Transform } from "../../lib/ttp";
import { TransformCurve } from "../TtpPanel/TransformCurve";

interface Props {
  objective: Objective;
  onChange: (patch: Partial<Objective>) => void;
  onRename: (name: string) => void;
  onRemove: () => void;
  weights: { min: number; max: number; step: number };
  /** Share of the total weight, 0-1. */
  share: number;
}

const FIELD = "bg-sunken border-subtle focus:border-strong rounded-md border px-1.5 py-0.5 text-[11px] outline-none";
const fmt = (value: unknown) => (typeof value === "number" && Number.isFinite(value) ? String(+value.toPrecision(6)) : "");
const BOUND: Record<string, string> = { right_step: "≥", left_step: "≤", exponential_decay: "k" };

/** One objective of a new profile: its name, curve, window and weight. */
export function BuilderObjective({ objective, onChange, onRename, onRemove, weights, share }: Props) {
  const transform = objective.transform ?? null;
  const fields = transform ? rangeFields(transform.type).map((f) => f.key) : [];
  const reference = Array.isArray(objective.params?.smiles) ? String(objective.params.smiles[0] ?? "") : "";
  const known = (TRANSFORM_TYPES as readonly string[]).includes(transform?.type ?? "");
  const relative = weights.max > 0 ? objective.weight / weights.max : 1;
  const colour = objectiveSource(objective) === "native" ? "var(--color-ok)" : "var(--color-structure)";
  const setTransform = (next: Transform | null) => onChange({ transform: next });
  const field = (key: string) =>
    transform && <NumberField value={transform[key]} onChange={(value) => setTransform(setRange(transform, { [key]: value }))} label={`${objective.name} ${key}`} />;

  return (
    <li className="border-subtle bg-raised flex items-start gap-3 rounded-lg border px-2.5 py-2">
      <div className="min-w-0 flex-1 space-y-1.5">
        <div className="flex items-center gap-1.5">
          <span aria-hidden="true" className="size-1.5 shrink-0 rounded-full" style={{ background: colour }} />
          <input
            value={objective.name}
            onChange={(event) => onRename(event.target.value)}
            aria-label="Objective name"
            className="hover:bg-sunken focus:bg-sunken min-w-0 flex-1 rounded px-1 py-0.5 text-xs font-semibold outline-none"
          />
          <button type="button" onClick={onRemove} aria-label={`Remove ${objective.name}`} title="Remove" className="text-faint hover:text-body cursor-pointer px-1 text-sm leading-none">
            ×
          </button>
        </div>
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px]">
          <select
            value={transform?.type ?? ""}
            onChange={(event) => setTransform(retypeObjective(objective, event.target.value || null))}
            aria-label={`${objective.name} curve`}
            className={`${FIELD} max-w-44`}
          >
            <option value="">Score as is (0–1)</option>
            {TRANSFORM_TYPES.filter((t) => t !== "value_mapping" || transform?.type === "value_mapping").map((type) => (
              <option key={type} value={type}>
                {CURVE_LABELS[type]}
              </option>
            ))}
            {transform && !known && <option value={transform.type}>{transform.type}</option>}
          </select>
          {transform && fields.length === 2 && (
            <span className="text-muted flex items-center gap-1 font-mono">
              [{field(fields[0])},{field(fields[1])}]
            </span>
          )}
          {transform && fields.length === 1 && (
            <span className="text-muted flex items-center gap-1">
              {BOUND[transform.type] ?? fields[0]}
              {field(fields[0])}
            </span>
          )}
        </div>
        {objective.component === "TanimotoSimilarity" && (
          <input
            value={reference}
            onChange={(event) => onChange({ params: { ...(objective.params ?? {}), smiles: event.target.value.trim() ? [event.target.value.trim()] : [] } })}
            placeholder="Reference SMILES"
            aria-label={`${objective.name} reference SMILES`}
            spellCheck={false}
            className={`${FIELD} w-full font-mono text-[10px]`}
          />
        )}
        <label className="flex items-center gap-2">
          <span className="text-muted w-10 shrink-0 text-[11px]">weight</span>
          <input
            type="range"
            min={weights.min}
            max={weights.max}
            step={weights.step}
            value={Math.min(weights.max, objective.weight)}
            onChange={(event) => onChange({ weight: Number(event.target.value) })}
            aria-label={`${objective.name} weight`}
            className="min-w-0 flex-1 accent-[var(--color-table)]"
          />
          <span className="w-8 shrink-0 text-right font-mono text-[11px] tabular-nums">{objective.weight.toFixed(2)}</span>
          <span className="text-faint w-8 shrink-0 text-right text-[10px] tabular-nums" title="Share of the total weight">
            {Math.round(share * 100)}%
          </span>
        </label>
      </div>
      {transform ? (
        <TransformCurve
          transform={transform}
          width={176}
          height={50}
          weight={relative}
          onRange={fields.includes("low") ? (patch) => setTransform(setRange(transform, patch)) : undefined}
        />
      ) : (
        <span aria-hidden="true" className="w-[176px] shrink-0" />
      )}
    </li>
  );
}

/** A number input that follows outside changes (a dragged threshold) while keeping half-typed text like "-" or "2.". */
function NumberField({ value, onChange, label }: { value: unknown; onChange: (value: number) => void; label: string }) {
  const [text, setText] = useState(fmt(value));
  const [seen, setSeen] = useState(value);
  if (value !== seen) {
    setSeen(value);
    setText(fmt(value));
  }
  return (
    <input
      type="text"
      inputMode="decimal"
      value={text}
      onChange={(event) => {
        setText(event.target.value);
        const parsed = Number.parseFloat(event.target.value);
        if (Number.isFinite(parsed)) {
          setSeen(parsed);
          onChange(parsed);
        }
      }}
      aria-label={label}
      spellCheck={false}
      className={`${FIELD} w-14 font-mono tabular-nums`}
    />
  );
}
