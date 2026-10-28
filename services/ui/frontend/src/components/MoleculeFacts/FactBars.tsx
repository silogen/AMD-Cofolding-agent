// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { ReactNode } from "react";
import type { Bar, BarFact, BarGroup } from "../../lib/factBars";
import type { RelatedFile } from "../../lib/structureFiles";

const BAR_COLORS: Record<Bar["tone"], string> = { ok: "var(--color-ok)", warn: "var(--color-warn)", plain: "var(--color-structure)" };

/**
 * Groups of values side by side once the enclosing `@container` is wide
 * enough: `left` stacked in the first column, `right` in the second.
 */
export function FactColumns({ left, right }: { left: (BarGroup & { action?: ReactNode })[]; right: BarGroup[] }) {
  if (!left.length && !right.length) return null;
  const column = (groups: (BarGroup & { action?: ReactNode })[]) =>
    groups.length > 0 && (
      <div className="space-y-3">
        {groups.map((group) => (
          <FactList key={group.title} {...group} />
        ))}
      </div>
    );
  return (
    <div className="grid items-start gap-x-8 gap-y-3 text-[11px] @xl:grid-cols-2">
      {column(left)}
      {column(right)}
    </div>
  );
}

/** The files of a structure as one row of download links, then cofold and predictions next to the ligand's properties. */
export function DetailsColumns({ files, cofold, predictions, properties }: { files: RelatedFile[]; cofold: BarFact[]; predictions: BarGroup[]; properties: BarFact[] }) {
  return (
    <div className="space-y-3 text-[11px]">
      {files.length > 0 && (
        <ul aria-label="Files" className="flex flex-wrap gap-1.5">
          {files.map((file) => (
            <li key={file.url}>
              <a
                href={file.url}
                download={file.name}
                title={file.name}
                className="bg-sunken hover:bg-inset text-body flex items-baseline gap-1.5 rounded-full px-2.5 py-0.5 no-underline"
              >
                {file.label}
                <span className="text-faint text-[10px]">{file.type}</span>
              </a>
            </li>
          ))}
        </ul>
      )}
      <FactColumns left={[...(cofold.length ? [{ title: "Cofold", facts: cofold }] : []), ...predictions]} right={properties.length ? [{ title: "Properties", facts: properties }] : []} />
    </div>
  );
}

/** One value per row: label, a short bar, the value and its unit. */
export function FactList({ title, facts, action }: BarGroup & { action?: ReactNode }) {
  return (
    <section aria-label={title} className="max-w-xs">
      <h4 className="text-faint mb-0.5 flex items-center gap-2 text-[10px] font-semibold">
        <span className="min-w-0 truncate">{title}</span>
        {action}
      </h4>
      <dl>
        {facts.map((fact) => (
          <div key={fact.label} className="grid grid-cols-[7rem_minmax(0,1fr)] items-center gap-2 leading-5" title={fact.title}>
            <dt className="text-muted truncate">{fact.label}</dt>
            <dd className="flex min-w-0 items-center gap-2">
              <span aria-hidden="true" className={`h-1 w-12 shrink-0 overflow-hidden rounded-full ${fact.bar ? "bg-inset" : ""}`}>
                {fact.bar && <span className="block h-full rounded-full" style={{ width: `${Math.round(fact.bar.fraction * 100)}%`, background: BAR_COLORS[fact.bar.tone] }} />}
              </span>
              <span className="truncate font-mono tabular-nums" style={fact.tone === "warn" ? { color: "var(--color-warn)" } : undefined}>
                {fact.value ?? "—"}
                {fact.unit && <span className="text-faint ml-0.5 font-sans">{fact.unit}</span>}
              </span>
            </dd>
          </div>
        ))}
      </dl>
    </section>
  );
}
