// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { RelatedFile } from "../../lib/structureFiles";
import type { Bar, BarFact, BarGroup } from "./detailFacts";

const BAR_COLORS: Record<Bar["tone"], string> = { ok: "var(--color-ok)", warn: "var(--color-warn)", plain: "var(--color-structure)" };

/** The files of a structure as one row of download links, then cofold and predictions next to the ligand's properties. */
export function DetailsColumns({ files, cofold, predictions, properties }: { files: RelatedFile[]; cofold: BarFact[]; predictions: BarGroup[]; properties: BarFact[] }) {
  const left = [...(cofold.length ? [{ title: "Cofold", facts: cofold }] : []), ...predictions];
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
      {(left.length > 0 || properties.length > 0) && (
        <div className="grid items-start gap-x-8 gap-y-3 @xl:grid-cols-2">
          {left.length > 0 && (
            <div className="space-y-3">
              {left.map((group) => (
                <FactList key={group.title} title={group.title} facts={group.facts} />
              ))}
            </div>
          )}
          {properties.length > 0 && <FactList title="Properties" facts={properties} />}
        </div>
      )}
    </div>
  );
}

function FactList({ title, facts }: BarGroup) {
  return (
    <section aria-label={title} className="max-w-xs">
      <h4 className="text-faint mb-0.5 text-[10px] font-semibold">{title}</h4>
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
