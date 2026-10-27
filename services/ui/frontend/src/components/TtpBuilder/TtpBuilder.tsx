// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { ApiError, callTool } from "../../lib/api";
import {
  BOLTZ_NAME,
  PRESETS,
  asTtpList,
  buildTtp,
  createdTtp,
  emptyTargetFacts,
  formFromTtp,
  formProblems,
  hasCofolding,
  isAccession,
  mergeUniprot,
  normaliseSequence,
  objectiveCatalogue,
  parseComponents,
  parseServices,
  payloadError,
  presetObjectives,
  rememberCreated,
  renameObjective,
  sequenceProblem,
  serviceObjective,
  slugify,
  uniqueName,
  weightRangeOf,
  withCofolding,
  type CatalogueEntry,
  type Objective,
  type PresetId,
  type ServiceInfo,
  type TargetFacts,
  type Ttp,
  type TtpForm,
} from "../../lib/ttp";
import {
  DEFAULT_RUN,
  MAX_BATCH,
  MAX_STEPS,
  loadRunSettings,
  modelLabel,
  parseMol2MolModels,
  runCost,
  runProblems,
  saveRunSettings,
  ttpRequest,
  type PriorModel,
  type RunSettings,
} from "../../lib/ttpRun";
import { ObjectiveCard } from "../TtpPanel/ObjectiveCard";

export interface CreatedTtp {
  id: string;
  name: string;
}

interface Props {
  onClose: () => void;
  /** Called once create_ttp has stored the profile, with the text for the chat: use it, or run REINVENT on it. */
  onCreated: (ttp: CreatedTtp, request: string) => void;
  /** Start from a copy of this stored TTP. */
  initial?: Ttp;
}

interface Row {
  key: number;
  objective: Objective;
}

interface Catalog {
  services: Record<string, ServiceInfo>;
  components: string[];
  ids: string[];
  models: PriorModel[];
  loaded: boolean;
}

type Lookup = { state: "idle" } | { state: "busy" } | { state: "error"; message: string } | { state: "found"; facts: TargetFacts };
type Submit =
  | { state: "idle" }
  | { state: "busy"; label: string }
  | { state: "error"; message: string; forbidden: boolean };

const ORGANISMS = ["human", "mouse", "rat", ""];
const message = (err: unknown) => (err instanceof Error ? err.message : String(err));
const int = (n: number) => n.toLocaleString("en");

let nextKey = 1;
const rows = (objectives: Objective[]): Row[] => objectives.map((objective) => ({ key: nextKey++, objective }));

/** A form to define a target product profile; the chat then gets the request to use it, or to run REINVENT on it. */
export function TtpBuilder({ onClose, onCreated, initial }: Props) {
  const [start] = useState(() => (initial ? formFromTtp(initial) : null));
  const [catalog, setCatalog] = useState<Catalog>({ services: {}, components: [], ids: [], models: [], loaded: false });
  const [name, setName] = useState(start?.name ?? "");
  const [id, setId] = useState<string | null>(start?.id ?? null);
  const [description, setDescription] = useState(start?.description ?? "");
  const [sequence, setSequence] = useState(start?.sequence ?? "");
  const [pdbId, setPdbId] = useState(start?.pdbId ?? "");
  const [aggregation, setAggregation] = useState<TtpForm["aggregation"]>(start?.aggregation ?? "geometric_mean");
  const [objectives, setObjectives] = useState<Row[]>(() => rows(start?.objectives ?? presetObjectives("druglike", {})));
  const [preset, setPreset] = useState<PresetId | null>(start ? null : "druglike");
  const [query, setQuery] = useState("");
  const [organism, setOrganism] = useState("human");
  const [lookup, setLookup] = useState<Lookup>({ state: "idle" });
  const [attempted, setAttempted] = useState(false);
  const [submit, setSubmit] = useState<Submit>({ state: "idle" });
  const [copied, setCopied] = useState(false);
  const [run, setRun] = useState<RunSettings>(() => loadRunSettings());
  const [options, setOptions] = useState(false);

  useEffect(() => {
    let cancelled = false;
    Promise.allSettled([callTool("list_external_services"), callTool("list_scoring_components"), callTool("list_ttps"), callTool("list_available_models")]).then(
      ([services, components, ttps, models]) => {
        if (cancelled) return;
        setCatalog({
          services: services.status === "fulfilled" ? parseServices(services.value.result) : {},
          components: components.status === "fulfilled" ? parseComponents(components.value.result) : [],
          ids: ttps.status === "fulfilled" ? asTtpList(ttps.value.result).map((t) => t.id) : [],
          models: models.status === "fulfilled" ? parseMol2MolModels(models.value.result) : [],
          loaded: true,
        });
      },
    );
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const form: TtpForm = {
    name,
    id: id ?? slugify(name),
    description,
    sequence,
    pdbId,
    aggregation,
    objectives: objectives.map((r) => r.objective),
    constraints: start?.constraints,
  };
  const problems = formProblems(form, catalog.ids);
  const runIssues = run.enabled ? runProblems(run) : [];
  const catalogue = useMemo(() => objectiveCatalogue(catalog.services, catalog.components), [catalog]);
  const residues = normaliseSequence(sequence);
  const cofold = hasCofolding(form.objectives);
  const boltzMissing = catalog.loaded && Object.keys(catalog.services).length > 0 && !catalog.services.boltz;
  const weights = weightRangeOf(form.objectives);
  const total = form.objectives.reduce((sum, o) => sum + Math.max(0, o.weight), 0) || 1;
  const installed = catalog.models.filter((m) => m.available);
  const mol2mol = installed.length ? installed : [{ key: DEFAULT_RUN.model, label: modelLabel(DEFAULT_RUN.model), available: true }];
  const effective: RunSettings = { ...run, model: mol2mol.some((m) => m.key === run.model) ? run.model : mol2mol[0].key };
  const cost = runCost(run, cofold);
  const busy = submit.state === "busy";

  const setList = (list: Objective[]) => setObjectives((previous) => list.map((objective, i) => ({ key: previous[i]?.key ?? nextKey++, objective })));
  const replaceList = (list: Objective[]) => setObjectives(rows(list));
  const updateRun = (patch: Partial<RunSettings>) => {
    const next = { ...run, ...patch };
    setRun(next);
    saveRunSettings(next);
  };

  const applyPreset = (value: PresetId) => {
    setPreset(value);
    replaceList(withCofolding(presetObjectives(value, catalog.services), catalog.services, cofold));
  };

  const add = (entry: CatalogueEntry) => {
    const taken = form.objectives.map((o) => o.name);
    const added: Objective[] = [];
    const objective = entry.make();
    if (objective.depends_on === BOLTZ_NAME && !cofold) {
      added.push(serviceObjective(catalog.services, "boltz"));
      taken.push(BOLTZ_NAME);
    } else if (objective.depends_on === BOLTZ_NAME) {
      objective.depends_on = form.objectives.find((o) => o.service === "boltz")?.name ?? BOLTZ_NAME;
    }
    added.push({ ...objective, name: uniqueName(objective.name, taken) });
    setObjectives((previous) => [...previous, ...rows(added)]);
  };

  const remove = (index: number) => {
    const target = form.objectives[index];
    if (target?.service === "boltz") replaceList(withCofolding(form.objectives, catalog.services, false));
    else setObjectives((previous) => previous.filter((_, i) => i !== index));
  };

  const search = async () => {
    const text = query.trim();
    if (!text) return;
    setLookup({ state: "busy" });
    try {
      const response = await callTool("fetch_target_info", { query: isAccession(text) || !organism ? text : `${text} ${organism}` });
      const error = payloadError(response.result);
      const facts = mergeUniprot(emptyTargetFacts(), response.result);
      if (error || !facts.sequence) {
        setLookup({ state: "error", message: error ?? "UniProt returned no sequence for that query." });
        return;
      }
      setLookup({ state: "found", facts });
      setSequence(facts.sequence);
      setPdbId("");
      if (!name.trim() && facts.gene) setName(`${facts.gene.toUpperCase()} ${cofold ? "cofolding" : (PRESETS.find((p) => p.id === preset)?.label.toLowerCase() ?? "profile")}`);
    } catch (err) {
      setLookup({ state: "error", message: message(err) });
    }
  };

  const createOnly = async (): Promise<CreatedTtp | null> => {
    const response = await callTool("create_ttp", { ttp: buildTtp(form) });
    const created = createdTtp(response.result);
    if ("error" in created) {
      setSubmit({ state: "error", message: created.error, forbidden: false });
      return null;
    }
    rememberCreated(created.id);
    return created;
  };

  const failed = (err: unknown) => setSubmit({ state: "error", message: message(err), forbidden: err instanceof ApiError && err.status === 403 });

  const create = async () => {
    setAttempted(true);
    if (problems.length || runIssues.length) return;
    setSubmit({ state: "busy", label: "Creating…" });
    try {
      const created = await createOnly();
      if (created) onCreated(created, ttpRequest(created, effective));
    } catch (err) {
      failed(err);
    }
  };

  const copyJson = async () => {
    try {
      await navigator.clipboard.writeText(JSON.stringify(buildTtp(form), null, 2));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  };

  const groups = [...new Set(catalogue.map((e) => e.group))];
  const shown = attempted ? [...problems, ...runIssues] : [];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
      <div
        role="dialog"
        aria-modal="true"
        aria-label={initial ? `Duplicate ${initial.name}` : "New target product profile"}
        className="bg-raised border-subtle shadow-card flex max-h-[92vh] w-full max-w-2xl flex-col rounded-xl border text-xs"
      >
        <header className="border-subtle flex items-center gap-2 border-b px-4 py-2.5">
          <div className="min-w-0 flex-1">
            <h2 className="text-sm font-semibold">{initial ? `Duplicate ${initial.name}` : "New target product profile"}</h2>
            <p className="text-faint text-[11px]">What REINVENT optimises for: a target, the objectives and how much each counts.</p>
          </div>
          <button type="button" onClick={onClose} aria-label="Close" className="text-faint hover:text-body cursor-pointer px-1 text-lg leading-none">
            ×
          </button>
        </header>

          <div className="min-h-0 flex-1 space-y-5 overflow-y-auto px-4 py-4">
            <Labelled label="Name" hint={form.id ? `id ${form.id}` : undefined}>
              <input value={name} onChange={(event) => setName(event.target.value)} placeholder="e.g. BRD4 cofolding lead optimisation" autoFocus className={BLOCK} />
            </Labelled>

            <Section title="Target" hint={cofold ? "required for cofolding" : "optional without cofolding"}>
              <form
                className="flex flex-wrap gap-2"
                onSubmit={(event) => {
                  event.preventDefault();
                  void search();
                }}
              >
                <input
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Gene or UniProt accession, e.g. BRD4 or O60885"
                  aria-label="Gene or UniProt accession"
                  className={`${INPUT} min-w-48 flex-1`}
                />
                <select value={organism} onChange={(event) => setOrganism(event.target.value)} aria-label="Organism" className={INPUT} disabled={isAccession(query)}>
                  {ORGANISMS.map((o) => (
                    <option key={o} value={o}>
                      {o || "any organism"}
                    </option>
                  ))}
                </select>
                <button type="submit" disabled={!query.trim() || lookup.state === "busy"} className={SECONDARY}>
                  {lookup.state === "busy" ? "Looking up…" : "Look up"}
                </button>
              </form>
              {lookup.state === "error" && <p style={{ color: "var(--color-accent)" }}>{lookup.message}</p>}
              {lookup.state === "found" && (
                <div className="bg-sunken border-subtle rounded-lg border px-3 py-2">
                  <p className="font-medium">
                    {lookup.facts.name ?? lookup.facts.entryName}
                    <span className="text-faint font-normal">
                      {[lookup.facts.gene, lookup.facts.organism, lookup.facts.uniprot, `${lookup.facts.sequenceLength ?? lookup.facts.sequence?.length} aa`].filter(Boolean).map((part) => ` · ${part}`)}
                    </span>
                  </p>
                  {lookup.facts.pdbIds.length > 0 && (
                    <div className="mt-1 flex flex-wrap items-center gap-1">
                      <span className="text-faint text-[10px]">PDB</span>
                      {lookup.facts.pdbIds.slice(0, 8).map((pdb) => (
                        <button
                          key={pdb}
                          type="button"
                          onClick={() => setPdbId(pdbId === pdb ? "" : pdb)}
                          aria-pressed={pdbId === pdb}
                          className={`cursor-pointer rounded-md border px-1.5 py-px font-mono text-[10px] ${pdbId === pdb ? "border-strong text-body" : "border-subtle text-muted hover:text-body"}`}
                        >
                          {pdb}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              )}
              <textarea
                value={sequence}
                onChange={(event) => setSequence(event.target.value)}
                rows={2}
                spellCheck={false}
                aria-label="Protein sequence"
                placeholder="…or paste the protein sequence (FASTA is fine)"
                className={`${BLOCK} resize-y font-mono text-[10px] break-all`}
              />
              {residues && <p className="text-faint text-[10px]">{sequenceProblem(residues) ?? `${residues.length} residues`}</p>}
            </Section>

            <div className="border-subtle flex items-start gap-3 rounded-lg border px-3 py-2.5">
              <div className="min-w-0 flex-1">
                <p className="font-semibold">Cofold with Boltz-2</p>
                <p className="text-faint text-[11px] leading-snug">
                  {boltzMissing
                    ? "Boltz-2 is not configured on this server."
                    : cofold
                      ? "Each molecule is cofolded with the target: Boltz-2 binding affinity, then PoseBusters checks the pose. Much slower."
                      : "Off: molecules are scored on ligand properties only. Turn on to score binding to the target."}
                </p>
                {cofold && !residues && (
                  <p className="mt-0.5 text-[11px]" style={{ color: "var(--color-warn)" }}>
                    Needs the target's protein sequence: look it up above.
                  </p>
                )}
              </div>
              <Switch
                on={cofold}
                label="Cofold with Boltz-2"
                disabled={!catalog.loaded || (boltzMissing && !cofold)}
                onChange={(on) => replaceList(withCofolding(form.objectives, catalog.services, on))}
              />
            </div>

            <Section title="Objectives" hint="each scored 0–1, then combined by weight">
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-faint text-[11px]">Start from</span>
                {PRESETS.filter((p) => p.id !== "cofolding").map((p) => (
                  <button
                    key={p.id}
                    type="button"
                    onClick={() => applyPreset(p.id)}
                    aria-pressed={preset === p.id}
                    title={p.hint}
                    className={`cursor-pointer rounded-full border px-2 py-px text-[11px] ${preset === p.id ? "border-strong bg-sunken text-body font-medium" : "border-subtle text-muted hover:text-body"}`}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
              {objectives.length === 0 && <p className="text-faint">Pick a starting set above, or add objectives one by one below.</p>}
              <ul className="space-y-1.5">
                {objectives.map((row, index) => (
                  <ObjectiveCard
                    key={row.key}
                    objective={row.objective}
                    weights={weights}
                    share={Math.max(0, row.objective.weight) / total}
                    onChange={(patch) => setList(form.objectives.map((o, i) => (i === index ? { ...o, ...patch } : o)))}
                    onRename={(value) => setList(renameObjective(form.objectives, index, value))}
                    onRemove={() => remove(index)}
                  />
                ))}
              </ul>
              <select
                value=""
                onChange={(event) => {
                  const entry = catalogue.find((e) => e.key === event.target.value);
                  if (entry) add(entry);
                }}
                aria-label="Add an objective"
                className={INPUT}
              >
                <option value="">+ Add an objective…</option>
                {groups.map((group) => (
                  <optgroup key={group} label={group}>
                    {catalogue
                      .filter((e) => e.group === group)
                      .map((e) => (
                        <option key={e.key} value={e.key}>
                          {e.label}
                        </option>
                      ))}
                  </optgroup>
                ))}
              </select>
              {!catalog.loaded && <span className="text-faint ml-2">Loading the services…</span>}
            </Section>

            <section className="border-subtle space-y-2.5 rounded-lg border px-3 py-2.5">
              <div className="flex items-start gap-3">
                <div className="min-w-0 flex-1">
                  <p className="font-semibold">Run REINVENT</p>
                  <p className="text-faint text-[11px] leading-snug">
                    {run.enabled ? "The chat gets a request to start a run with these settings." : "Off: the chat just gets the TTP to use."}
                  </p>
                </div>
                <Switch on={run.enabled} label="Run REINVENT" onChange={(enabled) => updateRun({ enabled })} />
              </div>
              {run.enabled && (
                <>
              <div className="flex flex-wrap items-end gap-3">
                <Labelled label="Steps">
                  <input
                    type="number"
                    min={1}
                    max={MAX_STEPS}
                    step={50}
                    value={run.maxSteps}
                    onChange={(event) => updateRun({ maxSteps: Math.round(Number(event.target.value)) })}
                    className={`${INPUT} w-20 font-mono tabular-nums`}
                  />
                </Labelled>
                <Labelled label="Batch size">
                  <input
                    type="number"
                    min={1}
                    max={MAX_BATCH}
                    step={8}
                    value={run.batchSize}
                    onChange={(event) => updateRun({ batchSize: Math.round(Number(event.target.value)) })}
                    className={`${INPUT} w-20 font-mono tabular-nums`}
                  />
                </Labelled>
                <p className="text-faint pb-1 text-[11px]">
                  = {int(cost.molecules)} molecules{cofold ? `, each cofolded` : ""}
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-muted text-[11px]">Prior</span>
                <div role="radiogroup" aria-label="Prior" className="bg-sunken border-subtle inline-flex rounded-md border p-0.5">
                  {(
                    [
                      ["reinvent", "Reinvent · de novo"],
                      ["mol2mol", "Mol2Mol · from seeds"],
                    ] as const
                  ).map(([value, label]) => (
                    <button
                      key={value}
                      type="button"
                      role="radio"
                      aria-checked={run.prior === value}
                      onClick={() => updateRun({ prior: value })}
                      className={`cursor-pointer rounded px-2 py-0.5 text-[11px] ${run.prior === value ? "bg-raised text-body shadow-card font-medium" : "text-muted hover:text-body"}`}
                    >
                      {label}
                    </button>
                  ))}
                </div>
                {run.prior === "mol2mol" && mol2mol.length > 1 && (
                  <select value={effective.model} onChange={(event) => updateRun({ model: event.target.value })} aria-label="Mol2Mol prior" className={INPUT}>
                    {mol2mol.map((m) => (
                      <option key={m.key} value={m.key}>
                        {m.label}
                      </option>
                    ))}
                  </select>
                )}
              </div>
              {run.prior === "mol2mol" && (
                <textarea
                  value={run.seeds}
                  onChange={(event) => updateRun({ seeds: event.target.value })}
                  rows={2}
                  spellCheck={false}
                  aria-label="Seed SMILES"
                  placeholder="Seed SMILES, one per line: the molecules Mol2Mol starts from"
                  className={`${BLOCK} resize-y font-mono text-[10px]`}
                />
              )}
                </>
              )}
            </section>

            <div>
              <button type="button" onClick={() => setOptions((v) => !v)} aria-expanded={options} className="text-muted hover:text-body cursor-pointer text-[11px] font-medium">
                {options ? "▾" : "▸"} More options <span className="text-faint font-normal">id, description, PDB, how scores combine</span>
              </button>
              {options && (
                <div className="mt-2 space-y-2">
                  <div className="grid gap-2 sm:grid-cols-[12rem_1fr]">
                    <Labelled label="Id" hint="lowercase, - and _">
                      <input value={form.id} onChange={(event) => setId(event.target.value)} placeholder="brd4-cofolding" className={`${BLOCK} font-mono`} spellCheck={false} />
                    </Labelled>
                    <Labelled label="Description" hint="optional">
                      <input value={description} onChange={(event) => setDescription(event.target.value)} placeholder="What this profile is for" className={BLOCK} />
                    </Labelled>
                  </div>
                  <div className="flex flex-wrap items-end gap-3">
                    <Labelled label="PDB id" hint="optional">
                      <input value={pdbId} onChange={(event) => setPdbId(event.target.value)} placeholder="e.g. 3MXF" className={`${INPUT} w-24 font-mono`} spellCheck={false} />
                    </Labelled>
                    <Labelled label="Combine scores">
                      <select value={aggregation} onChange={(event) => setAggregation(event.target.value as TtpForm["aggregation"])} className={INPUT}>
                        <option value="geometric_mean">Geometric mean</option>
                        <option value="arithmetic_mean">Arithmetic mean</option>
                      </select>
                    </Labelled>
                    <p className="text-faint min-w-48 flex-1 pb-1 text-[11px]">
                      {aggregation === "geometric_mean"
                        ? "A molecule must do reasonably on every objective: one zero sinks the total."
                        : "Objectives trade off: a strong score can make up for a weak one."}
                    </p>
                  </div>
                  {(start?.constraints?.length ?? 0) > 0 && <p className="text-faint text-[11px]">Keeps the {start?.constraints?.length} structural constraint(s) of the original.</p>}
                </div>
              )}
            </div>
          </div>

        {submit.state === "error" && (
          <div className="mx-4 mb-2">
            {submit.forbidden ? (
              <Alert
                title="Creating profiles from the UI is not enabled on this server yet."
                detail={`The gateway does not allow create_ttp as a direct tool (${submit.message}). Copy the JSON and ask the agent to create it.`}
              />
            ) : (
              <Alert title="The server did not accept the profile." detail={submit.message} />
            )}
          </div>
        )}

        <footer className="border-subtle flex flex-wrap items-center gap-2 border-t px-4 py-2.5">
              <div className="min-w-0 flex-1 text-[11px]">
                {shown.length > 0 ? (
                  <ul style={{ color: "var(--color-accent)" }}>
                    {shown.slice(0, 3).map((p) => (
                      <li key={p}>{p}</li>
                    ))}
                    {shown.length > 3 && <li>…and {shown.length - 3} more.</li>}
                  </ul>
                ) : (
                  <span className="text-faint">{problems.length === 0 ? "Ready to create." : `${problems.length} thing${problems.length === 1 ? "" : "s"} left to fill in.`}</span>
                )}
              </div>
              <button type="button" onClick={() => void copyJson()} title="Copy the create_ttp JSON" className={GHOST}>
                {copied ? "Copied" : "Copy JSON"}
              </button>
              <button type="button" onClick={onClose} className={GHOST}>
                Cancel
              </button>
              <button
                type="button"
                onClick={() => void create()}
                disabled={busy}
                title={run.enabled ? "Save the profile and put the run request in the chat" : "Save the profile and put it in the chat"}
                className={PRIMARY}
                style={{ background: "var(--color-accent)" }}
              >
                {busy ? submit.label : "Create TTP"}
              </button>
        </footer>
      </div>
    </div>
  );
}

const INPUT = "bg-sunken border-subtle focus:border-strong rounded-md border px-2 py-1 text-xs outline-none";
const BLOCK = `${INPUT} w-full`;
const SECONDARY = "bg-inset text-body hover:bg-sunken border-subtle cursor-pointer rounded-md border px-2.5 py-1 font-medium disabled:cursor-default disabled:opacity-50 touch:min-h-11";
const GHOST = "text-muted hover:text-body hover:bg-inset cursor-pointer rounded-md px-2 py-1 font-medium disabled:opacity-50 touch:min-h-11";
const PRIMARY = "cursor-pointer rounded-md px-3 py-1 font-semibold text-white transition-opacity hover:opacity-90 disabled:cursor-default disabled:opacity-50 touch:min-h-11";

function Section({ title, hint, children }: { title: string; hint?: string; children: ReactNode }) {
  return (
    <section className="space-y-2">
      <h3 className="flex items-baseline gap-2 text-[11px] font-semibold">
        {title}
        {hint && <span className="text-faint font-normal">{hint}</span>}
      </h3>
      {children}
    </section>
  );
}

function Labelled({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="block space-y-0.5">
      <span className="text-muted flex gap-2 text-[11px]">
        {label}
        {hint && <span className="text-faint font-mono">{hint}</span>}
      </span>
      {children}
    </label>
  );
}

function Switch({ on, label, disabled, onChange }: { on: boolean; label: string; disabled?: boolean; onChange: (on: boolean) => void }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!on)}
      className="relative mt-0.5 h-5 w-9 shrink-0 cursor-pointer rounded-full transition-colors disabled:cursor-default disabled:opacity-50"
      style={{ background: on ? "var(--color-structure)" : "var(--border-strong)" }}
    >
      <span className="absolute top-0.5 size-4 rounded-full bg-white shadow transition-all" style={{ left: on ? "1.125rem" : "0.125rem" }} />
    </button>
  );
}

function Alert({ title, detail }: { title: string; detail: string }) {
  return (
    <div
      className="rounded-lg border px-3 py-2"
      style={{ borderColor: "color-mix(in oklab, var(--color-accent) 35%, transparent)", background: "color-mix(in oklab, var(--color-accent) 8%, transparent)" }}
    >
      <p className="font-medium">{title}</p>
      <p className="text-muted mt-0.5 font-mono text-[10px] break-words whitespace-pre-wrap">{detail}</p>
    </div>
  );
}
