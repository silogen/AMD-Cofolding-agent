// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useMemo, useRef, useState } from "react";
import type { Theme } from "../../hooks/useTheme";
import { useRDKit } from "../../hooks/useRDKit";
import { download } from "../../lib/api";
import type { Artifacts } from "../../lib/artifacts";
import { candidatesOf } from "../../lib/candidates";
import { fileName, toCsv, toSdf } from "../../lib/csv";
import type { MoleculeContext } from "../../lib/moleculePage";
import {
  columnRange,
  filterMolecules,
  listColumns,
  listMolecules,
  moleculeSheet,
  sortMolecules,
  starValues,
  tidyNumber,
  type ListColumn,
  type ListedMolecule,
} from "../../lib/moleculeList";
import { canonical, descriptors, withMol } from "../../lib/rdkit";
import { admetPrompt, mergeShortlist, parseShortlistFile, shortlistJson, shortlistPrompt, shortlistSeeds, useShortlist } from "../../lib/shortlist";
import { latestTtp, toolEvents } from "../../lib/ttp";
import { DownloadIcon, TextButton } from "../DataTable/ExportButtons";
import { MoleculeTable, type MoleculeCell, type MoleculeColumn, type MoleculeRow } from "../MoleculeTable/MoleculeTable";
import { StarButton } from "../Shortlist/StarButton";
import { RunCard } from "../TtpBuilder/RunFields";

interface Props {
  artifacts: Artifacts;
  theme: Theme;
  sessionId: string | null;
  onOpenMolecule: (smiles: string, context?: MoleculeContext) => void;
  onAsk?: (text: string) => void;
}

/** Rows drawn at first and added by "Show more"; drawings are the costly part. */
const PAGE = 50;
const TONES: Record<ListColumn["tone"], string> = {
  score: "var(--color-molecule)",
  cofold: "var(--color-structure)",
  admet: "var(--color-table)",
  property: "var(--text-muted)",
};
/** Origins shown in a row; the rest are in its tooltip. */
const ORIGINS = 2;
const NAME: MoleculeColumn = { key: "name", label: "Name", title: "Its name and where it came from", text: true };

type Row = ListedMolecule & MoleculeRow;

/** A row's cells: its name over its origins, then each value over a hairline of where it stands. */
function cellsOf(row: ListedMolecule, columns: readonly ListColumn[], ranges: ReadonlyMap<string, [number, number] | null>): Record<string, MoleculeCell> {
  const origins = row.origins.slice(0, ORIGINS).join(" · ") + (row.origins.length > ORIGINS ? ` +${row.origins.length - ORIGINS}` : "");
  const cells: Record<string, MoleculeCell> = { name: { text: row.name ?? "", sub: origins, title: row.tools.join(", ") || undefined } };
  for (const c of columns) {
    const value = row.values[c.key];
    if (value === undefined) continue;
    const range = ranges.get(c.key);
    const share = c.bar === "fraction" ? value : range ? (value - range[0]) / (range[1] - range[0]) : null;
    cells[c.key] = { text: tidyNumber(value), score: share === null ? null : Math.max(0.06, share) };
  }
  return cells;
}

/**
 * Every molecule of the conversation once: scored, generated, looked up,
 * predicted or starred, merged by canonical SMILES, with what is known of
 * each as columns. A click opens the molecule's page.
 */
export function MoleculesPanel({ artifacts, theme, sessionId, onOpenMolecule, onAsk }: Props) {
  const session = sessionId ?? "new";
  const starred = useShortlist(session);
  const candidates = candidatesOf(artifacts);
  const { rdkit } = useRDKit(candidates.length > 0 || artifacts.properties.length > 0 || starred.length > 0);

  const rows = useMemo(() => {
    const keys = new Map<string, string>();
    const key = (smiles: string) => {
      let out = keys.get(smiles);
      if (out === undefined) keys.set(smiles, (out = (rdkit && canonical(rdkit, smiles)) || smiles));
      return out;
    };
    return listMolecules({
      candidates,
      properties: artifacts.properties,
      starred,
      key,
      descriptors: rdkit ? (smiles) => descriptors(rdkit, smiles) ?? undefined : undefined,
    });
  }, [candidates, artifacts.properties, starred, rdkit]);
  const columns = useMemo(() => listColumns(rows, artifacts.properties), [rows, artifacts.properties]);
  const tableColumns = useMemo<MoleculeColumn[]>(
    () => [NAME, ...columns.map((c) => ({ key: c.key, label: c.label, unit: c.unit ?? undefined, title: c.title, strong: c.key === "score", colour: TONES[c.tone] }))],
    [columns],
  );
  const ranges = useMemo(() => new Map(columns.map((c) => [c.key, c.bar === "range" ? columnRange(rows, c.key) : null])), [columns, rows]);

  const [query, setQuery] = useState("");
  const [starredOnly, setStarredOnly] = useState(false);
  const [sort, setSort] = useState<{ key: string; desc: boolean } | null>(null);
  const [limit, setLimit] = useState(PAGE);
  const [note, setNote] = useState<string | null>(null);
  const [seeding, setSeeding] = useState(false);
  const sessionTtpId = useMemo(() => latestTtp(toolEvents(artifacts.steps)).id, [artifacts.steps]);

  const sortKey = sort?.key ?? (columns.some((c) => c.key === "score") ? "score" : null);
  const shown = useMemo(
    () => sortMolecules(filterMolecules(rows, query, starredOnly), sortKey, sort?.desc ?? true),
    [rows, query, starredOnly, sortKey, sort],
  );
  const page = useMemo<Row[]>(() => shown.slice(0, limit).map((r) => ({ ...r, cells: cellsOf(r, columns, ranges) })), [shown, limit, columns, ranges]);

  if (rows.length === 0) {
    return <p className="text-faint flex-1 px-8 py-12 text-center text-xs">No molecules yet. The ones the agent scores, generates or looks up gather here.</p>;
  }

  const sortBy = (key: string) => {
    if (sortKey !== key) setSort({ key, desc: key !== "name" });
    else if ((sort?.desc ?? true) === (key !== "name")) setSort({ key, desc: key === "name" });
    else setSort(null);
  };
  const name = starredOnly ? `starred molecules ${session}` : `molecules ${session}`;
  const scope = shown.length === rows.length ? `all ${rows.length} molecules` : `the ${shown.length} shown`;
  const importFile = (text: string | null) => {
    if (text === null) return setNote("That file could not be read");
    try {
      const { added, skipped } = mergeShortlist(session, parseShortlistFile(text));
      setNote(added === 0 ? `Already starred: ${skipped}` : `${added} starred${skipped ? `, ${skipped} already were` : ""}`);
    } catch (error) {
      setNote(error instanceof Error ? error.message : "That file could not be imported");
    }
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="border-subtle flex shrink-0 flex-wrap items-center gap-1.5 border-b px-3 py-2">
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search"
          aria-label="Search molecules by name, SMILES or origin"
          className="bg-sunken border-subtle focus:border-strong w-40 min-w-0 flex-1 rounded-md border px-2 py-1 text-xs outline-none"
        />
        <button
          type="button"
          aria-pressed={starredOnly}
          onClick={() => setStarredOnly((on) => !on)}
          title={starredOnly ? "Show every molecule" : "Show the starred molecules only"}
          className={`inline-flex cursor-pointer items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] touch:min-h-11 ${
            starredOnly ? "text-body" : "border-subtle text-muted hover:text-body"
          }`}
          style={starredOnly ? { borderColor: "var(--color-warn)" } : undefined}
        >
          <span aria-hidden="true" style={{ color: "var(--color-warn)" }}>
            ★
          </span>
          Starred
          <span className="text-faint tabular-nums">{starred.length}</span>
        </button>
        <span className="ml-auto flex items-center gap-0.5">
          <TextButton
            title={`Download ${scope} as CSV`}
            disabled={shown.length === 0}
            onClick={() => {
              const { columns: cols, rows: cells } = moleculeSheet(shown, columns);
              download(fileName(name, "csv"), toCsv(cols, cells), "text/csv");
            }}
          >
            <DownloadIcon /> CSV
          </TextButton>
          <TextButton
            title={rdkit ? `Download ${scope} as SDF, the numbers as data fields` : "RDKit is loading"}
            disabled={!rdkit || shown.length === 0}
            onClick={() => {
              if (!rdkit) return;
              const sheet = moleculeSheet(shown, columns);
              const sdf = toSdf(shown.map((r, i) => ({ molblock: withMol(rdkit, r.smiles, (mol) => mol.get_molblock()), title: r.name ?? undefined, fields: sheet.rows[i] })));
              download(fileName(name, "sdf"), sdf, "chemical/x-mdl-sdfile");
            }}
          >
            <DownloadIcon /> SDF
          </TextButton>
        </span>
      </div>

      {starredOnly && (
        <div className="text-faint flex shrink-0 flex-wrap items-center gap-0.5 px-3 py-1 text-[11px]">
          <span className="mr-auto min-w-0 truncate">{note ?? "Stars are kept in this browser."}</span>
          {onAsk && starred.length > 0 && (
            <>
              <TextButton title="Start a Mol2Mol run from the starred molecules" onClick={() => setSeeding(!seeding)}>
                Use as seeds…
              </TextButton>
              <TextButton title="Put a cofolding request for the starred molecules in the composer" onClick={() => onAsk(shortlistPrompt(starred))}>
                Cofold
              </TextButton>
              <TextButton title="Put an ADMET request for the starred molecules in the composer" onClick={() => onAsk(admetPrompt(starred))}>
                ADMET
              </TextButton>
            </>
          )}
          {starred.length > 0 && (
            <TextButton title="Back up the stars as JSON, to import them in another browser" onClick={() => download(fileName(`shortlist ${session}`, "json"), shortlistJson(starred, session), "application/json")}>
              <DownloadIcon /> JSON
            </TextButton>
          )}
          <ImportButton onFile={importFile} />
        </div>
      )}
      {starredOnly && seeding && onAsk && starred.length > 0 && (
        <div className="shrink-0 px-3 pb-2">
          <RunCard preferTtpId={sessionTtpId} seeds={shortlistSeeds(starred)} onAsk={onAsk} onClose={() => setSeeding(false)} />
        </div>
      )}

      <div className="flex min-h-0 flex-1 flex-col p-2">
        <MoleculeTable
          columns={tableColumns}
          rows={page}
          theme={theme}
          sort={sortKey ? { key: sortKey, desc: sort?.desc ?? true } : undefined}
          onSort={sortBy}
          onOpen={(row) => onOpenMolecule(row.smiles, row.run ? { run: row.run, name: row.name } : undefined)}
          actions={(row) => <StarButton smiles={row.smiles} label={row.name} source={row.run ?? row.tools[0] ?? "molecules"} values={() => starValues(row)} session={session} />}
          empty={starredOnly && !query.trim() ? "Nothing starred yet." : "No molecule matches."}
          maxHeight={null}
          footer={
            shown.length > limit && (
              <p className="py-2 text-center">
                <TextButton title={`${shown.length - limit} more molecules`} onClick={() => setLimit((n) => n + PAGE)}>
                  Show more · {shown.length - limit}
                </TextButton>
              </p>
            )
          }
        />
      </div>
    </div>
  );
}

/** Reads a shortlist JSON file the user picks. */
function ImportButton({ onFile }: { onFile: (text: string | null) => void }) {
  const input = useRef<HTMLInputElement>(null);
  return (
    <>
      <TextButton title="Import starred molecules from a shortlist JSON file" onClick={() => input.current?.click()}>
        Import…
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
