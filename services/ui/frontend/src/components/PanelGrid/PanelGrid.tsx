// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { Panel, StructurePanel } from "../../types/api";
import type { Theme } from "../../hooks/useTheme";
import { StructureCard } from "../StructureCard/StructureCard";
import { MoleculeCard } from "../MoleculeCard/MoleculeCard";
import { DataTable } from "../DataTable/DataTable";

interface Props {
  panels: Panel[];
  theme: Theme;
  onOpenStructure: (panel: StructurePanel) => void;
  openStructurePath: string | null;
  onOpenMolecule?: (smiles: string) => void;
}

/** What tool calls produced: complexes, then molecules, then tables. */
export function PanelGrid({ panels, theme, onOpenStructure, openStructurePath, onOpenMolecule }: Props) {
  const structures = panels.filter((panel): panel is StructurePanel => panel.kind === "structure");
  const molecules = panels.filter((panel) => panel.kind === "molecule");
  const tables = panels.filter((panel) => panel.kind === "table");

  return (
    <div className="space-y-3">
      {structures.length > 0 && (
        <div className="grid gap-3 sm:grid-cols-2">
          {structures.map((panel) => (
            <StructureCard
              key={panel.path}
              panel={panel}
              theme={theme}
              onOpen={onOpenStructure}
              isOpen={openStructurePath === panel.path}
            />
          ))}
        </div>
      )}

      {molecules.length > 0 && (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {molecules.map((panel) => (
            <MoleculeCard key={panel.smiles} panel={panel} theme={theme} onOpen={onOpenMolecule} />
          ))}
        </div>
      )}

      {tables.map((panel, index) => (
        <DataTable key={index} panel={panel} />
      ))}
    </div>
  );
}
