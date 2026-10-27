// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useMemo, useRef, useState } from "react";
import type { Theme } from "../../hooks/useTheme";
import { useRDKit } from "../../hooks/useRDKit";
import { download } from "../../lib/api";
import type { Artifacts } from "../../lib/artifacts";
import { buildCandidates } from "../../lib/candidates";
import { fileName, toCsv } from "../../lib/csv";
import { descriptors, withMol } from "../../lib/rdkit";
import {
  admetPrompt,
  mergeShortlist,
  parseShortlistFile,
  removeStar,
  shortlistColumns,
  shortlistJson,
  shortlistPrompt,
  shortlistSdf,
  shortlistSeeds,
  shortlistSheet,
  shortlistValues,
  useShortlist,
  type KnownValues,
} from "../../lib/shortlist";
import { latestTtp, toolEvents } from "../../lib/ttp";
import { DownloadIcon, TextButton } from "../DataTable/ExportButtons";
import { MolDepiction } from "../MolDepiction/MolDepiction";
import { SmilesText } from "../MolDepiction/SmilesText";
import { RunCard } from "../TtpBuilder/RunFields";

interface Props {
  artifacts: Artifacts;
  theme: Theme;
  sessionId: string | null;
  onOpenMolecule: (smiles: string) => void;
  onAsk: (prompt: string) => void;
}

const DESCRIPTORS = ["MW", "cLogP", "TPSA"];
/** Score columns shown in the list; every known value goes into the exports. */
const SHOWN_SCORES = 3;

/** The starred molecules of the conversation, with the best scores known for each. */
export function ShortlistPanel({ artifacts, theme, sessionId, onOpenMolecule, onAsk }: Props) {
  const entries = useShortlist(sessionId ?? "new");
  const { rdkit } = useRDKit();
  const [seeding, setSeeding] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const candidates = useMemo(() => new Map(buildCandidates(artifacts).map((c) => [c.smiles, c])), [artifacts]);
  const sessionTtpId = useMemo(() => latestTtp(toolEvents(artifacts.steps)).id, [artifacts.steps]);

  const known = useMemo<KnownValues>(() => {
    return (smiles) => {
      const out: Record<string, number | undefined> = {};
      const c = candidates.get(smiles);
      if (c?.score !== undefined) out.Score = c.score;
      Object.assign(out, c?.components);
      const d = rdkit ? descriptors(rdkit, smiles) : null;
      if (d) Object.assign(out, { MW: d.mw, cLogP: d.clogp, TPSA: d.tpsa });
      return out;
    };
  }, [candidates, rdkit]);

  const columns = useMemo(() => {
    const all = shortlistColumns(entries, known);
    const scores = all.filter((c) => !DESCRIPTORS.includes(c) && !/\(raw\)$/.test(c)).slice(0, SHOWN_SCORES);
    return [...scores, ...DESCRIPTORS.filter((d) => all.includes(d))];
  }, [entries, known]);

  const session = sessionId ?? "new";
  const importFile = (text: string | null) => {
    if (text === null) {
      setNote("That file could not be read");
      return;
    }
    try {
      const { added, skipped } = mergeShortlist(session, parseShortlistFile(text));
      setNote(added === 0 ? `Already starred: ${skipped} molecule${skipped === 1 ? "" : "s"}` : `${added} imported${skipped ? `, ${skipped} already starred` : ""}`);
    } catch (error) {
      setNote(error instanceof Error ? error.message : "That file could not be imported");
    }
  };

  if (entries.length === 0) {
    return (
      <div className="flex-1 px-8 py-12 text-center">
        <p className="text-faint text-xs leading-relaxed">Star molecules in Results, Candidates or the chat to collect them here.</p>
        <p className="mt-3">
          <ImportButton onFile={importFile} />
        </p>
        {note && <p className="text-faint mt-1 text-[10px]">{note}</p>}
      </div>
    );
  }

  const name = `shortlist ${session}`;
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="text-faint flex shrink-0 flex-wrap items-center gap-1 px-3 py-1.5 text-[11px]">
        <span className="mr-auto min-w-0 truncate">
          {entries.length} starred
          {note && <span className="ml-2">· {note}</span>}
        </span>
        <TextButton
          title="Download the shortlist as CSV, with every known score"
          onClick={() => {
            const { columns: cols, rows } = shortlistSheet(entries, known);
            download(fileName(name, "csv"), toCsv(cols, rows), "text/csv");
          }}
        >
          <DownloadIcon /> CSV
        </TextButton>
        <TextButton
          title={rdkit ? "Download as SDF with scores as data fields" : "RDKit is loading"}
          disabled={!rdkit}
          onClick={() => rdkit && download(fileName(name, "sdf"), shortlistSdf(entries, (s) => withMol(rdkit, s, (mol) => mol.get_molblock()), known), "chemical/x-mdl-sdfile")}
        >
          <DownloadIcon /> SDF
        </TextButton>
        <TextButton
          title="Download the shortlist as JSON, to keep it or to import it in another browser"
          onClick={() => download(fileName(name, "json"), shortlistJson(entries, session), "application/json")}
        >
          <DownloadIcon /> JSON
        </TextButton>
        <ImportButton onFile={importFile} />
        <TextButton title="Start a Mol2Mol run from these molecules" onClick={() => setSeeding(!seeding)}>
          Use as seeds…
        </TextButton>
        <TextButton title="Put an ADMET prediction request for these SMILES in the composer" onClick={() => onAsk(admetPrompt(entries))}>
          ADMET
        </TextButton>
        <TextButton title="Put a prompt with these SMILES in the composer" onClick={() => onAsk(shortlistPrompt(entries))}>
          Ask the agent…
        </TextButton>
      </div>
      {seeding && (
        <div className="shrink-0 px-3 pb-2">
          <RunCard preferTtpId={sessionTtpId} seeds={shortlistSeeds(entries)} onAsk={onAsk} onClose={() => setSeeding(false)} />
        </div>
      )}
      <div className="min-h-0 flex-1 overflow-auto">
        <table className="w-full border-collapse text-xs">
          <thead className="bg-sunken text-muted sticky top-0 z-10 text-[10px]">
            <tr>
              <th scope="col" className="border-subtle border-b px-2 py-1.5 text-left font-medium">
                Molecule
              </th>
              {columns.map((c) => (
                <th key={c} scope="col" className="border-subtle max-w-24 truncate border-b px-2 py-1.5 text-right font-medium" title={c}>
                  {c}
                </th>
              ))}
              <th scope="col" className="border-subtle w-6 border-b" aria-label="Remove" />
            </tr>
          </thead>
          <tbody>
            {entries.map((e) => {
              const values = shortlistValues(e, known);
              return (
                <tr key={e.smiles} className="group hover:bg-sunken">
                  <td className="border-subtle border-b px-2 py-1">
                    <span className="flex items-center gap-2">
                      <button type="button" onClick={() => onOpenMolecule(e.smiles)} className="hover:bg-inset h-12 w-16 shrink-0 cursor-pointer rounded" title="Open the 2D drawing">
                        <MolDepiction smiles={e.smiles} theme={theme} width={128} height={96} className="size-full" />
                      </button>
                      <span className="flex min-w-0 flex-col">
                        <SmilesText smiles={e.smiles} label={e.label} className="max-w-28 text-[10px] @xl:max-w-44" />
                        {e.source && <span className="text-faint truncate text-[10px]">{e.source}</span>}
                      </span>
                    </span>
                  </td>
                  {columns.map((c) => (
                    <td key={c} className="border-subtle border-b px-2 py-1 text-right font-mono text-[11px] tabular-nums">
                      {values[c] === undefined ? <span className="text-faint">—</span> : values[c].toFixed(DESCRIPTORS.includes(c) ? 1 : 2)}
                    </td>
                  ))}
                  <td className="border-subtle border-b pr-1 text-right">
                    <button
                      type="button"
                      onClick={() => removeStar(session, e.smiles)}
                      aria-label={`Remove ${e.label ?? e.smiles} from the shortlist`}
                      title="Remove"
                      className="text-faint hover:text-body cursor-pointer px-1 opacity-0 group-hover:opacity-100 focus-visible:opacity-100 touch:opacity-100"
                    >
                      ×
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/** Reads a shortlist JSON file the user picks; the panel merges it. */
function ImportButton({ onFile }: { onFile: (text: string | null) => void }) {
  const input = useRef<HTMLInputElement>(null);
  return (
    <>
      <TextButton title="Import a shortlist JSON file, adding the molecules that are not starred yet" onClick={() => input.current?.click()}>
        Import JSON…
      </TextButton>
      <input
        ref={input}
        type="file"
        accept="application/json,.json"
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0];
          // Cleared so picking the same file again fires another change.
          event.target.value = "";
          if (file) void file.text().then(onFile, () => onFile(null));
        }}
      />
    </>
  );
}
