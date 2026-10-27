// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useState } from "react";
import {
  CURVE_LABELS,
  TRANSFORM_TYPES,
  advancedParams,
  rangeFields,
  retypeObjective,
  setRange,
  sourceLabel,
  objectiveSource,
  type Objective,
  type Transform,
} from "../../lib/ttp";
import { TransformCurve } from "./TransformCurve";

interface Props {
  objective: Objective;
  onChange: (patch: Partial<Objective>) => void;
  /** Makes the name editable. */
  onRename?: (name: string) => void;
  onRemove?: () => void;
  weights: { min: number; max: number; step: number };
  /** Share of the total weight, 0-1. */
  share?: number;
  /** The stored version, for "was" hints. */
  original?: Objective;
  changed?: boolean;
}

const FIELD = "bg-sunken border-subtle focus:border-strong rounded-md border px-1.5 py-0.5 text-[11px] outline-none";
const fmt = (value: unknown) => (typeof value === "number" && Number.isFinite(value) ? String(+value.toPrecision(6)) : "");

/** One objective: its curve type, window and weight, with the curve following every edit. */
export function ObjectiveCard({ objective, onChange, onRename, onRemove, weights, share, original, changed }: Props) {
  const [more, setMore] = useState(false);
  const transform = objective.transform ?? null;
  const fields = transform ? rangeFields(transform.type) : [];
  const extra = transform ? advancedParams(transform.type) : [];
  const params = objective.service_params ?? {};
  const selector = [params.endpoint, params.target, params.interaction_type].filter((v): v is string => typeof v === "string" && v !== "");
  const what = [sourceLabel(objective), objective.component, ...selector, objective.depends_on && `after ${objective.depends_on}`].filter(Boolean).join(" · ");
  const reference = Array.isArray(objective.params?.smiles) ? String(objective.params.smiles[0] ?? "") : "";
  const known = (TRANSFORM_TYPES as readonly string[]).includes(transform?.type ?? "");
  const relative = weights.max > 0 ? objective.weight / weights.max : 1;
  const colour = objectiveSource(objective) === "native" ? "var(--color-ok)" : "var(--color-structure)";
  const hasMore = extra.length > 0 || Boolean(objective.comment);

  const setTransform = (next: Transform | null) => onChange({ transform: next });
  const was = (key: string) =>
    original?.transform && typeof original.transform[key] === "number" && original.transform[key] !== transform?.[key] ? `was ${fmt(original.transform[key])}` : null;

  return (
    <li className="border-subtle bg-raised rounded-lg border">
      <div className="flex items-start gap-3 px-2.5 py-2">
        <div className="min-w-0 flex-1 space-y-1.5">
          <div className="flex items-center gap-1.5">
            <span aria-hidden="true" className="size-1.5 shrink-0 rounded-full" style={{ background: changed ? "var(--color-accent)" : colour }} />
            {onRename ? (
              <input
                value={objective.name}
                onChange={(event) => onRename(event.target.value)}
                aria-label="Objective name"
                className="hover:bg-sunken focus:bg-sunken min-w-0 flex-1 rounded px-1 py-0.5 text-xs font-semibold outline-none"
              />
            ) : (
              <span className="min-w-0 flex-1 truncate px-1 text-xs font-semibold">{objective.name}</span>
            )}
            <span className="text-faint hidden max-w-48 truncate text-[10px] sm:block" title={what}>
              {what}
            </span>
            {onRemove && (
              <button type="button" onClick={onRemove} aria-label={`Remove ${objective.name}`} title="Remove" className="text-faint hover:text-body cursor-pointer px-1 text-sm leading-none">
                ×
              </button>
            )}
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
            {transform &&
              fields.map((field) => (
                <label key={field.key} className="text-muted flex items-center gap-1">
                  {field.label}
                  <NumberField
                    value={transform[field.key]}
                    onChange={(value) => setTransform(setRange(transform, { [field.key]: value }))}
                    label={`${objective.name} ${field.key}`}
                  />
                  {was(field.key) && <span className="text-faint font-mono text-[10px]">{was(field.key)}</span>}
                </label>
              ))}
            {hasMore && (
              <button type="button" onClick={() => setMore((v) => !v)} aria-expanded={more} className="text-faint hover:text-body cursor-pointer text-[10px]">
                {more ? "less ▴" : "more ▾"}
              </button>
            )}
          </div>

          {objective.component === "TanimotoSimilarity" && (
            <label className="text-muted flex items-center gap-2 text-[11px]">
              reference
              <input
                value={reference}
                onChange={(event) => onChange({ params: { ...(objective.params ?? {}), smiles: event.target.value.trim() ? [event.target.value.trim()] : [] } })}
                placeholder="SMILES to stay close to"
                spellCheck={false}
                className={`${FIELD} min-w-0 flex-1 font-mono text-[10px]`}
              />
            </label>
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
            {share !== undefined && (
              <span className="text-faint w-8 shrink-0 text-right text-[10px] tabular-nums" title="Share of the total weight">
                {Math.round(share * 100)}%
              </span>
            )}
          </label>
        </div>

        {transform ? (
          <TransformCurve
            transform={transform}
            width={176}
            height={50}
            weight={relative}
            onRange={fields.some((f) => f.key === "low") ? (patch) => setTransform(setRange(transform, patch)) : undefined}
          />
        ) : (
          <span className="bg-sunken text-faint grid h-[60px] w-[176px] shrink-0 place-items-center rounded-md px-3 text-center text-[10px] leading-snug">
            The score is already 0–1 and used as is.
          </span>
        )}
      </div>

      {more && (
        <div className="border-subtle bg-sunken flex flex-wrap items-center gap-x-3 gap-y-1 rounded-b-lg border-t px-2.5 py-1.5 text-[11px]">
          {transform && extra.map((key) => (
            <label key={key} className="text-muted flex items-center gap-1 font-mono">
              {key === "k" ? "steepness k" : key}
              <NumberField value={transform[key]} onChange={(value) => setTransform({ ...transform, [key]: value })} label={`${objective.name} ${key}`} />
              {was(key) && <span className="text-faint text-[10px]">{was(key)}</span>}
            </label>
          ))}
          {transform && <span className="text-faint font-mono text-[10px]">{transform.type}</span>}
          {objective.comment && <p className="text-faint w-full text-[10px]">{objective.comment}</p>}
        </div>
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
