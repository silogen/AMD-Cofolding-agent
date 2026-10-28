// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useMemo, useState, type ReactNode } from "react";
import type { InteractionsPanel, MoleculePanel, Panel, PropertiesPanel, StructurePanel } from "../../types/api";
import type { Theme } from "../../hooks/useTheme";
import { mergeTurnPanels } from "../../lib/artifacts";
import { StructureCard } from "../StructureCard/StructureCard";
import { MoleculeCard } from "../MoleculeCard/MoleculeCard";
import { InteractionsCard } from "../InteractionsCard/InteractionsCard";
import { PropertiesCard } from "../PropertiesCard/PropertiesCard";

interface Props {
  panels: Panel[];
  theme: Theme;
  openStructurePath: string | null;
  onOpenMolecule?: (smiles: string) => void;
  onShowProperties?: () => void;
}

/** What tool calls produced: complexes and their contacts, predictions, then molecules. Tables feed the candidates, not the chat. */
export function PanelGrid({ panels, theme, openStructurePath, onOpenMolecule, onShowProperties }: Props) {
  const merged = useMemo(() => mergeTurnPanels(panels), [panels]);
  const of = <K extends Panel["kind"]>(kind: K) => merged.filter((panel) => panel.kind === kind) as Extract<Panel, { kind: K }>[];
  const structures: StructurePanel[] = of("structure");
  const interactions: InteractionsPanel[] = of("interactions");
  const properties: PropertiesPanel[] = of("properties");
  const molecules: MoleculePanel[] = of("molecule");

  return (
    <div className="@container space-y-3">
      {structures.length > 0 && (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(min(100%,16rem),20rem))] gap-3">
          {structures.slice(0, 6).map((panel) => (
            <StructureCard key={panel.path} panel={panel} theme={theme} isOpen={openStructurePath === panel.path} />
          ))}
        </div>
      )}
      {structures.length > 6 && (
        <p className="text-muted text-[11px]">{structures.length - 6} more structures are in the workbench viewer.</p>
      )}

      {interactions.map((panel, index) => (
        <InteractionsCard key={panel.structurePath ?? `${panel.smiles}-${index}`} panel={panel} />
      ))}

      {properties.map((panel, index) => (
        <PropertiesCard key={index} panel={panel} theme={theme} limit={5} onOpen={onShowProperties} onOpenMolecule={onOpenMolecule} />
      ))}

      {molecules.length > 0 && (
        <Folded label={`${molecules.length} molecule${molecules.length === 1 ? "" : "s"}`}>
          <div className="grid gap-3 @md:grid-cols-2 @3xl:grid-cols-3">
            {molecules.map((panel) => (
              <MoleculeCard key={panel.smiles} panel={panel} theme={theme} onOpen={onOpenMolecule} />
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
