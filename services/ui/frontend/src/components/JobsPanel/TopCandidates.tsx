// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useMemo, useState } from "react";
import type { Theme } from "../../hooks/useTheme";
import { candidateScores, type Candidate } from "../../lib/jobs";
import type { JobStructure } from "../../lib/jobStructures";
import { candidateEndpoints, sortedIndices, TOTAL } from "../../lib/runAnalysis";
import { endpointColours } from "../RunAnalysis/plotUtils";
import { endpointCells, endpointColumn } from "../MoleculeTable/columns";
import { MoleculeTable, type MoleculeRow } from "../MoleculeTable/MoleculeTable";
import { StarButton } from "../Shortlist/StarButton";

interface Props {
  candidates: readonly Candidate[];
  /** Cofolded complexes by SMILES. */
  complexes: ReadonlyMap<string, JobStructure> | null;
  jobId: string;
  session: string;
  theme: Theme;
  onOpen?: (smiles: string) => void;
  onOpenStructure: (entry: JobStructure) => void;
}

interface Row extends MoleculeRow {
  candidate: Candidate;
}

/** A run's best molecules, drawn as the Analysis table draws its selection. */
export function TopCandidates({ candidates, complexes, jobId, session, theme, onOpen, onOpenStructure }: Props) {
  const [sort, setSort] = useState<{ key: string; desc: boolean }>({ key: TOTAL, desc: true });
  const data = useMemo(() => candidateEndpoints(candidates), [candidates]);
  const columns = useMemo(() => {
    const colours = endpointColours(data.endpoints);
    return data.endpoints.map((e) => endpointColumn(e, colours.get(e.key)));
  }, [data]);
  const rows = useMemo(() => {
    const order = sortedIndices({ n: candidates.length, values: data.values, step: new Int32Array(candidates.length) }, null, sort.key, sort.desc);
    return order.map((i): Row => {
      const c = candidates[i];
      const complex = complexes?.get(c.smiles);
      return {
        candidate: c,
        smiles: c.smiles,
        cells: endpointCells(data.endpoints, data, i),
        complex: complex ? `Show the Boltz complex${complex.step !== null ? ` from step ${complex.step}` : ""} in the viewer` : undefined,
      };
    });
  }, [candidates, complexes, data, sort]);

  return (
    <MoleculeTable
      columns={columns}
      rows={rows}
      theme={theme}
      sort={sort}
      onSort={(key) => setSort(sort.key === key ? { key, desc: !sort.desc } : { key, desc: true })}
      onOpen={onOpen && ((row) => onOpen(row.smiles))}
      onOpenStructure={(row) => {
        const entry = complexes?.get(row.smiles);
        if (entry) onOpenStructure(entry);
      }}
      actions={(row) => <StarButton smiles={row.smiles} source={jobId} values={() => candidateScores(row.candidate)} session={session} />}
    />
  );
}
