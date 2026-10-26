// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useState, type ReactNode } from "react";
import type { StructurePanel } from "../../types/api";
import type { Theme } from "../../hooks/useTheme";
import type { Artifacts } from "../../lib/artifacts";
import { clock, duration } from "../../lib/format";
import { descriptors } from "../../lib/rdkit";
import { StructureViewer, type ViewStyle } from "../StructureViewer/StructureViewer";
import { DataTable } from "../DataTable/DataTable";
import { CopyButton } from "../CopyButton/CopyButton";
import { MolDepiction } from "../MolDepiction/MolDepiction";
import { useRDKit } from "../../hooks/useRDKit";
import { summarise } from "../../lib/format";

export type WorkbenchTab = "structure" | "molecules" | "tables" | "activity";
export type WorkbenchLayout = "sidebar" | "sheet";

interface Props {
  layout: WorkbenchLayout;
  tab: WorkbenchTab;
  onTabChange: (tab: WorkbenchTab) => void;
  artifacts: Artifacts;
  theme: Theme;
  activeStructure: StructurePanel | null;
  onSelectStructure: (panel: StructurePanel) => void;
  style: ViewStyle;
  onStyleChange: (style: ViewStyle) => void;
  selectedSmiles: string | null;
  onSelectSmiles: (smiles: string | null) => void;
  /** Put a prompt in the composer. */
  onAsk: (text: string) => void;
  onClose: () => void;
}

const TABS: { id: WorkbenchTab; label: string; colour: string }[] = [
  { id: "structure", label: "Structure", colour: "var(--color-structure)" },
  { id: "molecules", label: "Molecules", colour: "var(--color-molecule)" },
  { id: "tables", label: "Tables", colour: "var(--color-table)" },
  { id: "activity", label: "Activity", colour: "var(--color-accent)" },
];

/**
 * Everything the conversation produced, in one place: the complexes, the
 * molecules, the result tables and the log of tool calls. The chat says
 * what happened; the workbench is where the results are worked with.
 */
export function Workbench(props: Props) {
  const { layout, tab, onTabChange, artifacts, onClose } = props;
  const isSheet = layout === "sheet";

  useEffect(() => {
    if (!isSheet) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = previous;
    };
  }, [isSheet, onClose]);

  const counts: Record<WorkbenchTab, number> = {
    structure: artifacts.structures.length,
    molecules: artifacts.molecules.length,
    tables: artifacts.tables.length,
    activity: artifacts.steps.length,
  };

  const shell = isSheet
    ? "fixed inset-0 z-50 flex flex-col bg-raised"
    : "flex w-[26rem] xl:w-[34rem] 2xl:w-[40rem] shrink-0 flex-col border-l border-subtle bg-raised";

  return (
    <aside className={shell} role={isSheet ? "dialog" : undefined} aria-modal={isSheet || undefined} aria-label="Workbench">
      <div className="border-subtle flex shrink-0 items-center gap-1 border-b px-2 pt-2">
        <div role="tablist" aria-label="Workbench" className="flex min-w-0 flex-1 gap-0.5 overflow-x-auto">
          {TABS.map((entry) => {
            const selected = entry.id === tab;
            return (
              <button
                key={entry.id}
                type="button"
                role="tab"
                aria-selected={selected}
                onClick={() => onTabChange(entry.id)}
                className={`-mb-px flex cursor-pointer items-center gap-1.5 border-b-2 px-3 py-2 text-xs font-medium whitespace-nowrap ${
                  selected ? "text-body" : "text-muted hover:text-body border-transparent"
                }`}
                style={selected ? { borderColor: "var(--color-accent)" } : undefined}
              >
                <span aria-hidden="true" className="size-1.5 rounded-full" style={{ background: entry.colour }} />
                {entry.label}
                {counts[entry.id] > 0 && <span className="bg-inset text-faint rounded-full px-1.5 text-[10px]">{counts[entry.id]}</span>}
              </button>
            );
          })}
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close the workbench"
          title="Close the workbench"
          className="text-muted hover:text-body mb-1 grid size-8 shrink-0 cursor-pointer place-items-center rounded-lg text-base touch:size-11"
        >
          ×
        </button>
      </div>

      <div className="flex min-h-0 flex-1 flex-col">
        {tab === "structure" && <StructureTab {...props} />}
        {tab === "molecules" && <MoleculesTab {...props} />}
        {tab === "tables" && <TablesTab {...props} />}
        {tab === "activity" && <ActivityTab {...props} />}
      </div>
    </aside>
  );
}

const STYLES: { value: ViewStyle; label: string; hint: string }[] = [
  { value: "cartoon", label: "Cartoon", hint: "Protein fold with the ligand as sticks" },
  { value: "surface", label: "Surface", hint: "Binding pocket shape around the ligand" },
  { value: "ligand", label: "Ligand", hint: "The designed molecule on its own" },
];

function StructureTab({ artifacts, activeStructure, onSelectStructure, style, onStyleChange, theme }: Props) {
  const list = artifacts.structures;
  if (!activeStructure) {
    return <Empty>No structure yet. When the agent cofolds a complex with Boltz-2, it appears here.</Empty>;
  }
  return (
    <>
      <div role="group" aria-label="View style" className="border-subtle flex shrink-0 gap-1 border-b px-3 py-2">
        {STYLES.map((option) => (
          <button
            key={option.value}
            type="button"
            title={option.hint}
            aria-pressed={style === option.value}
            onClick={() => onStyleChange(option.value)}
            className="flex-1 cursor-pointer rounded-md px-2.5 py-1 text-[11px] font-medium transition-colors lg:flex-none touch:min-h-11"
            style={
              style === option.value
                ? { background: "var(--color-accent)", color: "white" }
                : { background: "var(--surface-inset)", color: "var(--text-muted)" }
            }
          >
            {option.label}
          </button>
        ))}
        <a
          href={activeStructure.url}
          download={activeStructure.name}
          className="text-muted hover:text-body ml-auto self-center text-[11px] font-medium"
        >
          Download
        </a>
      </div>

      <StructureViewer key={activeStructure.path} panel={activeStructure} theme={theme} style={style} className="min-h-0 flex-1" />

      <div className="border-subtle shrink-0 border-t px-3 py-2">
        <p className="truncate text-[11px] font-medium" title={activeStructure.path}>
          {activeStructure.name}
        </p>
        <p className="text-faint truncate text-[11px]">from {activeStructure.tool}</p>
      </div>

      {list.length > 1 && (
        <ul className="border-subtle max-h-44 shrink-0 overflow-y-auto border-t p-2">
          {list.map(({ item, at }) => (
            <li key={item.path}>
              <button
                type="button"
                onClick={() => onSelectStructure(item)}
                aria-current={activeStructure.path === item.path}
                className={`flex w-full cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-left text-[11px] touch:min-h-11 ${
                  activeStructure.path === item.path ? "bg-sunken font-semibold" : "hover:bg-sunken"
                }`}
                title={item.path}
              >
                <span className="min-w-0 flex-1 truncate">{item.name}</span>
                <span className="text-faint shrink-0">{clock(at)}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </>
  );
}

function MoleculesTab({ artifacts, theme, selectedSmiles, onSelectSmiles, onAsk }: Props) {
  const [query, setQuery] = useState("");
  const shown = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return artifacts.molecules;
    return artifacts.molecules.filter(
      ({ item, tool }) => item.smiles.toLowerCase().includes(needle) || (item.label ?? "").toLowerCase().includes(needle) || tool.includes(needle),
    );
  }, [artifacts.molecules, query]);

  if (artifacts.molecules.length === 0) {
    return <Empty>No molecules yet. Candidates, seeds and compounds the agent looks up are collected here.</Empty>;
  }

  return (
    <>
      <div className="border-subtle shrink-0 border-b px-3 py-2">
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Filter by SMILES, name or tool"
          aria-label="Filter molecules"
          className="bg-sunken border-subtle w-full rounded-lg border px-2.5 py-1.5 text-xs outline-none focus:border-strong"
        />
      </div>
      {selectedSmiles && (
        <MoleculeDetail smiles={selectedSmiles} artifacts={artifacts} theme={theme} onClose={() => onSelectSmiles(null)} onAsk={onAsk} />
      )}
      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        <ul className="grid grid-cols-2 gap-2 xl:grid-cols-3">
          {shown.map(({ item, tool }) => (
            <li key={item.smiles}>
              <button
                type="button"
                onClick={() => onSelectSmiles(item.smiles)}
                aria-pressed={item.smiles === selectedSmiles}
                title={item.smiles}
                className={`bg-sunken border-subtle w-full cursor-pointer overflow-hidden rounded-lg border text-left ${
                  item.smiles === selectedSmiles ? "border-strong ring-2 ring-[var(--color-accent)]" : "hover:border-strong"
                }`}
              >
                <MolDepiction smiles={item.smiles} theme={theme} width={200} height={140} className="h-28 w-full p-1" />
                <span className="border-subtle block truncate border-t px-2 py-1 text-[10px]">
                  <span className="font-semibold">{item.label ?? "Molecule"}</span>
                  <span className="text-faint"> · {tool}</span>
                </span>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </>
  );
}

/** Rows of every result table that mention this SMILES. */
function rowsFor(smiles: string, artifacts: Artifacts) {
  const found: { tool: string; row: Record<string, unknown> }[] = [];
  for (const { item, tool } of artifacts.tables) {
    for (const row of item.rows) {
      if (Object.values(row).some((value) => value === smiles)) found.push({ tool, row });
    }
  }
  return found;
}

function MoleculeDetail({
  smiles,
  artifacts,
  theme,
  onClose,
  onAsk,
}: {
  smiles: string;
  artifacts: Artifacts;
  theme: Theme;
  onClose: () => void;
  onAsk: (text: string) => void;
}) {
  const { rdkit } = useRDKit();
  const props = useMemo(() => (rdkit ? descriptors(rdkit, smiles) : null), [rdkit, smiles]);
  const rows = useMemo(() => rowsFor(smiles, artifacts), [smiles, artifacts]);

  const facts: [string, string][] = props
    ? [
        ["MW", props.mw.toFixed(1)],
        ["cLogP", props.clogp.toFixed(2)],
        ["TPSA", props.tpsa.toFixed(1)],
        ["HBD / HBA", `${props.hbd} / ${props.hba}`],
        ["Rot. bonds", String(props.rotb)],
        ["Heavy atoms", String(props.heavyAtoms)],
      ]
    : [];

  const asks: [string, string][] = [
    ["Cofold", `Cofold ${smiles} into {TARGET} with Boltz-2 and check the pose with PoseBusters.`],
    ["ADMET", `Predict ADMET properties for ${smiles}.`],
    ["Analogues", `Generate analogues of ${smiles} with Mol2Mol and score them against {TARGET}.`],
  ];

  return (
    <section aria-label="Selected molecule" className="border-subtle bg-sunken max-h-[55%] shrink-0 overflow-y-auto border-b p-3">
      <div className="flex gap-3">
        <div className="bg-raised border-subtle w-44 shrink-0 rounded-lg border p-1">
          <MolDepiction smiles={smiles} theme={theme} width={240} height={180} className="h-32 w-full" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-start gap-2">
            <code className="min-w-0 flex-1 font-mono text-[11px] break-all">{smiles}</code>
            <CopyButton value={smiles} label="Copy SMILES" />
            <button type="button" onClick={onClose} aria-label="Close" className="text-muted hover:text-body cursor-pointer text-base leading-none">
              ×
            </button>
          </div>
          {facts.length > 0 && (
            <dl className="mt-2 grid grid-cols-3 gap-x-3 gap-y-1 text-[11px]">
              {facts.map(([label, value]) => (
                <div key={label}>
                  <dt className="text-faint">{label}</dt>
                  <dd className="font-mono tabular-nums">{value}</dd>
                </div>
              ))}
            </dl>
          )}
          <div className="mt-2 flex flex-wrap gap-1">
            {asks.map(([label, prompt]) => (
              <button
                key={label}
                type="button"
                onClick={() => onAsk(prompt)}
                className="bg-raised border-subtle hover:border-strong cursor-pointer rounded-full border px-2.5 py-0.5 text-[11px]"
              >
                {label}…
              </button>
            ))}
          </div>
        </div>
      </div>
      {rows.length > 0 && (
        <div className="mt-3 space-y-2">
          {rows.slice(0, 4).map(({ tool, row }, index) => (
            <div key={index} className="bg-raised border-subtle rounded-lg border p-2">
              <p className="text-faint mb-1 text-[10px] font-semibold uppercase">{tool}</p>
              <dl className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[11px] sm:grid-cols-3">
                {Object.entries(row)
                  .filter(([, value]) => value !== smiles && value !== null && value !== "")
                  .map(([key, value]) => (
                    <div key={key} className="min-w-0">
                      <dt className="text-faint truncate">{key}</dt>
                      <dd className="truncate font-mono tabular-nums">
                        {typeof value === "number" && !Number.isInteger(value) ? value.toFixed(3) : String(value)}
                      </dd>
                    </div>
                  ))}
              </dl>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function TablesTab({ artifacts }: Props) {
  if (artifacts.tables.length === 0) {
    return <Empty>No tables yet. Scores, top candidates and predictions appear here.</Empty>;
  }
  return (
    <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-3">
      {artifacts.tables.map(({ item, tool, at }, index) => (
        <section key={index}>
          <h3 className="text-muted mb-1 flex gap-2 text-[11px] font-medium">
            <span className="font-mono">{tool}</span>
            <span className="text-faint ml-auto">{clock(at)}</span>
          </h3>
          <DataTable panel={item} />
        </section>
      ))}
    </div>
  );
}

function ActivityTab({ artifacts }: Props) {
  const [filter, setFilter] = useState<"all" | "failed">("all");
  const steps = filter === "failed" ? artifacts.steps.filter((step) => step.result?.isError) : artifacts.steps;
  const total = artifacts.steps.reduce((sum, step) => sum + (step.result?.elapsedMs ?? 0), 0);
  const failed = artifacts.steps.filter((step) => step.result?.isError).length;

  if (artifacts.steps.length === 0) {
    return <Empty>No tool calls yet. Every call the agent makes is logged here with its duration.</Empty>;
  }

  return (
    <>
      <div className="border-subtle text-muted flex shrink-0 items-center gap-3 border-b px-3 py-2 text-[11px]">
        <span>
          {artifacts.steps.length} calls · {duration(total)} in tools
        </span>
        <div role="group" aria-label="Filter" className="ml-auto flex gap-1">
          {(["all", "failed"] as const).map((value) => (
            <button
              key={value}
              type="button"
              aria-pressed={filter === value}
              onClick={() => setFilter(value)}
              className="cursor-pointer rounded-md px-2 py-0.5 font-medium"
              style={filter === value ? { background: "var(--color-accent)", color: "white" } : { background: "var(--surface-inset)" }}
            >
              {value === "all" ? "All" : `Failed (${failed})`}
            </button>
          ))}
        </div>
      </div>
      <ol className="min-h-0 flex-1 overflow-y-auto">
        {steps.map((step) => {
          const state = !step.result ? "running" : step.result.isError ? "failed" : "ok";
          return (
            <li key={step.id} className="border-subtle flex items-center gap-2 border-b px-3 py-1.5 text-[11px]">
              <span
                aria-label={state}
                className={`size-2 shrink-0 rounded-full ${state === "running" ? "animate-pulse" : ""}`}
                style={{ background: state === "ok" ? "var(--color-molecule)" : "var(--color-accent)" }}
              />
              <span className="text-faint shrink-0 font-mono">{clock(step.at)}</span>
              <span className="shrink-0 font-mono font-semibold">{step.name}</span>
              <span className="text-faint min-w-0 flex-1 truncate font-mono" title={summarise(step.arguments)}>
                {summarise(step.arguments)}
              </span>
              <span className="text-faint shrink-0 font-mono">
                {step.result?.elapsedMs != null ? duration(step.result.elapsedMs) : state === "running" ? "…" : ""}
              </span>
            </li>
          );
        })}
      </ol>
    </>
  );
}

function Empty({ children }: { children: ReactNode }) {
  return <p className="text-muted flex-1 px-6 py-10 text-center text-xs">{children}</p>;
}
