// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useState, type ReactNode } from "react";
import type { InteractionsPanel, MoleculePanel, Panel, PropertiesPanel, StructurePanel, TablePanel } from "../../types/api";
import type { Theme } from "../../hooks/useTheme";
import { StructureCard } from "../StructureCard/StructureCard";
import { MoleculeCard } from "../MoleculeCard/MoleculeCard";
import { DataTable } from "../DataTable/DataTable";
import { InteractionsCard } from "../InteractionsCard/InteractionsCard";
import { PropertiesCard } from "../PropertiesCard/PropertiesCard";

interface Props {
  panels: Panel[];
  theme: Theme;
  openStructurePath: string | null;
  onOpenMolecule?: (smiles: string) => void;
  onShowProperties?: () => void;
}

/** What tool calls produced: complexes and their contacts, predictions, molecules, then tables. */
export function PanelGrid({ panels, theme, openStructurePath, onOpenMolecule, onShowProperties }: Props) {
  const of = <K extends Panel["kind"]>(kind: K) => panels.filter((panel) => panel.kind === kind) as Extract<Panel, { kind: K }>[];
  const structures: StructurePanel[] = of("structure");
  const interactions: InteractionsPanel[] = of("interactions");
  const properties: PropertiesPanel[] = of("properties");
  const molecules: MoleculePanel[] = of("molecule");
  const tables: TablePanel[] = of("table");

  return (
    <div className="space-y-3">
      {structures.length > 0 && (
        <div className="grid gap-3 sm:grid-cols-2">
          {structures.slice(0, 6).map((panel) => (
            <StructureCard key={panel.path} panel={panel} theme={theme} isOpen={openStructurePath === panel.path} />
          ))}
        </div>
      )}
      {structures.length > 6 && (
        <p className="text-muted text-[11px]">{structures.length - 6} more structures are in the workbench viewer.</p>
      )}

      {interactions.map((panel, index) => (
        <InteractionsCard key={`${panel.structurePath}-${index}`} panel={panel} />
      ))}

      {properties.map((panel, index) => (
        <PropertiesCard key={index} panel={panel} theme={theme} limit={5} onOpen={onShowProperties} onOpenMolecule={onOpenMolecule} />
      ))}

      {molecules.length > 0 && (
        <Folded label={`${molecules.length} molecule${molecules.length === 1 ? "" : "s"}`}>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {molecules.map((panel) => (
              <MoleculeCard key={panel.smiles} panel={panel} theme={theme} onOpen={onOpenMolecule} />
            ))}
          </div>
        </Folded>
      )}

      {tables.length > 0 && (
        <Folded label={`${tables.length} table${tables.length === 1 ? "" : "s"}`}>
          <div className="space-y-3">
            {tables.map((panel, index) => (
              <DataTable key={index} panel={panel} />
            ))}
          </div>
        </Folded>
      )}
    </div>
  );
}

/** Results that the workbench also lists, one click away instead of in the answer's way. */
function Folded({ label, children }: { label: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="text-muted hover:text-body flex cursor-pointer items-center gap-1.5 text-[11px] font-medium touch:min-h-11"
      >
        <span aria-hidden="true" className={`inline-block transition-transform ${open ? "rotate-90" : ""}`}>
          ›
        </span>
        {label}
      </button>
      {open && <div className="mt-2">{children}</div>}
    </div>
  );
}
