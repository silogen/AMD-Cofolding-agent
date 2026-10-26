// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { ApiError, callTool } from "../../lib/api";
import {
  BOLTZ_NAME,
  PRESETS,
  TRANSFORM_TYPES,
  asTtpList,
  buildTtp,
  createdTtp,
  emptyTargetFacts,
  formProblems,
  isAccession,
  mergeUniprot,
  normaliseSequence,
  objectiveCatalogue,
  objectiveSource,
  parseComponents,
  parseServices,
  payloadError,
  presetObjectives,
  renameObjective,
  retype,
  sequenceProblem,
  serviceObjective,
  slugify,
  transformParams,
  uniqueName,
  type CatalogueEntry,
  type Objective,
  type PresetId,
  type ServiceInfo,
  type TargetFacts,
  type Transform,
  type TtpForm,
} from "../../lib/ttp";
import { TransformCurve } from "../TtpPanel/TransformCurve";

export interface CreatedTtp {
  id: string;
  name: string;
}

interface Props {
  onClose: () => void;
  /** Called once create_ttp has stored the profile. */
  onCreated: (ttp: CreatedTtp) => void;
}

interface Row {
  key: number;
  objective: Objective;
}

interface Catalog {
  services: Record<string, ServiceInfo>;
  components: string[];
  ids: string[];
  loaded: boolean;
}

type Lookup = { state: "idle" } | { state: "busy" } | { state: "error"; message: string } | { state: "found"; facts: TargetFacts };
type Submit = { state: "idle" } | { state: "busy" } | { state: "error"; message: string; forbidden: boolean };

const ORGANISMS = ["human", "mouse", "rat", ""];
const message = (err: unknown) => (err instanceof Error ? err.message : String(err));

let nextKey = 1;
const rows = (objectives: Objective[]): Row[] => objectives.map((objective) => ({ key: nextKey++, objective }));

/** A form to define a target product profile without asking the agent. */
export function TtpBuilder({ onClose, onCreated }: Props) {
  const [catalog, setCatalog] = useState<Catalog>({ services: {}, components: [], ids: [], loaded: false });
  const [name, setName] = useState("");
  const [id, setId] = useState<string | null>(null);
  const [description, setDescription] = useState("");
  const [sequence, setSequence] = useState("");
  const [pdbId, setPdbId] = useState("");
  const [aggregation, setAggregation] = useState<TtpForm["aggregation"]>("geometric_mean");
  const [objectives, setObjectives] = useState<Row[]>([]);
  const [preset, setPreset] = useState<PresetId | null>(null);
  const [query, setQuery] = useState("");
  const [organism, setOrganism] = useState("human");
  const [lookup, setLookup] = useState<Lookup>({ state: "idle" });
  const [attempted, setAttempted] = useState(false);
  const [submit, setSubmit] = useState<Submit>({ state: "idle" });
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    let cancelled = false;
    Promise.allSettled([callTool("list_external_services"), callTool("list_scoring_components"), callTool("list_ttps")]).then(([services, components, ttps]) => {
      if (cancelled) return;
      setCatalog({
        services: services.status === "fulfilled" ? parseServices(services.value.result) : {},
        components: components.status === "fulfilled" ? parseComponents(components.value.result) : [],
        ids: ttps.status === "fulfilled" ? asTtpList(ttps.value.result).map((t) => t.id) : [],
        loaded: true,
      });
    });
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
  };
  const problems = formProblems(form, catalog.ids);
  const catalogue = useMemo(() => objectiveCatalogue(catalog.services, catalog.components), [catalog]);
  const residues = normaliseSequence(sequence);
  const needsTarget = form.objectives.some((o) => o.service === "boltz");

  const setList = (list: Objective[]) => setObjectives((previous) => list.map((objective, i) => ({ key: previous[i]?.key ?? nextKey++, objective })));

  const applyPreset = (value: PresetId) => {
    setPreset(value);
    setObjectives(rows(presetObjectives(value, catalog.services)));
  };

  const add = (entry: CatalogueEntry) => {
    const taken = form.objectives.map((o) => o.name);
    const added: Objective[] = [];
    const objective = entry.make();
    if (objective.depends_on === BOLTZ_NAME && !form.objectives.some((o) => o.service === "boltz")) {
      added.push(serviceObjective(catalog.services, "boltz"));
      taken.push(BOLTZ_NAME);
    } else if (objective.depends_on === BOLTZ_NAME) {
      objective.depends_on = form.objectives.find((o) => o.service === "boltz")?.name ?? BOLTZ_NAME;
    }
    added.push({ ...objective, name: uniqueName(objective.name, taken) });
    setObjectives((previous) => [...previous, ...rows(added)]);
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
      if (!name.trim() && facts.gene) setName(`${facts.gene.toUpperCase()} ${preset ? PRESETS.find((p) => p.id === preset)?.label : "profile"}`);
    } catch (err) {
      setLookup({ state: "error", message: message(err) });
    }
  };

  const create = async () => {
    setAttempted(true);
    if (problems.length) return;
    setSubmit({ state: "busy" });
    try {
      const response = await callTool("create_ttp", { ttp: buildTtp(form) });
      const created = createdTtp(response.result);
      if ("error" in created) setSubmit({ state: "error", message: created.error, forbidden: false });
      else onCreated(created);
    } catch (err) {
      const forbidden = err instanceof ApiError && err.status === 403;
      setSubmit({ state: "error", message: message(err), forbidden });
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

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
      <div role="dialog" aria-modal="true" aria-label="New target product profile" className="bg-raised border-subtle shadow-card flex max-h-[92vh] w-full max-w-2xl flex-col rounded-xl border text-xs">
        <header className="border-subtle flex items-center gap-2 border-b px-4 py-3">
          <div className="min-w-0 flex-1">
            <h2 className="text-sm font-semibold">New target product profile</h2>
            <p className="text-faint text-[11px]">What REINVENT optimises for: a target, the objectives and how much each counts.</p>
          </div>
          <button type="button" onClick={onClose} aria-label="Close" className="text-faint hover:text-body cursor-pointer px-1 text-lg leading-none">
            ×
          </button>
        </header>

        <div className="min-h-0 flex-1 space-y-5 overflow-y-auto px-4 py-4">
          <Section title="Profile">
            <div className="grid gap-2 sm:grid-cols-[1fr_12rem]">
              <Labelled label="Name">
                <input
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  placeholder="e.g. BRD4 cofolding lead optimisation"
                  autoFocus
                  className={BLOCK}
                />
              </Labelled>
              <Labelled label="Id" hint="lowercase, - and _">
                <input value={form.id} onChange={(event) => setId(event.target.value)} placeholder="brd4-cofolding" className={`${BLOCK} font-mono`} spellCheck={false} />
              </Labelled>
            </div>
            <Labelled label="Description" hint="optional">
              <input value={description} onChange={(event) => setDescription(event.target.value)} placeholder="What this profile is for" className={BLOCK} />
            </Labelled>
          </Section>

          <Section title="Target" hint={needsTarget ? "required for Boltz" : "optional unless you score with Boltz"}>
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
            <Labelled label="Protein sequence" hint={residues ? `${residues.length} residues` : "or paste one (FASTA is fine)"}>
              <textarea
                value={sequence}
                onChange={(event) => setSequence(event.target.value)}
                rows={3}
                spellCheck={false}
                placeholder="MSAESGPGTRLRNLPVMGDGLETSQ…"
                className={`${BLOCK} resize-y font-mono text-[10px] break-all`}
              />
            </Labelled>
            {residues && sequenceProblem(residues) && <p style={{ color: "var(--color-accent)" }}>{sequenceProblem(residues)}</p>}
            <Labelled label="PDB id" hint="optional">
              <input value={pdbId} onChange={(event) => setPdbId(event.target.value)} placeholder="e.g. 3MXF" className={`${INPUT} w-28 font-mono`} spellCheck={false} />
            </Labelled>
          </Section>

          <Section title="Objectives" hint={`${objectives.length} · each scored 0–1, then combined by weight`}>
            <div className="grid gap-2 sm:grid-cols-3">
              {PRESETS.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => applyPreset(p.id)}
                  aria-pressed={preset === p.id}
                  className={`hover:border-strong cursor-pointer rounded-lg border px-3 py-2 text-left ${preset === p.id ? "border-strong bg-sunken" : "border-subtle"}`}
                >
                  <span className="block font-semibold">{p.label}</span>
                  <span className="text-faint block text-[10px] leading-snug">{p.hint}</span>
                </button>
              ))}
            </div>
            {objectives.length === 0 && <p className="text-faint">Start from a preset above, or add objectives one by one below.</p>}
            <ul className="space-y-1.5">
              {objectives.map((row, index) => (
                <ObjectiveEditor
                  key={row.key}
                  objective={row.objective}
                  onChange={(patch) => setList(form.objectives.map((o, i) => (i === index ? { ...o, ...patch } : o)))}
                  onRename={(value) => setList(renameObjective(form.objectives, index, value))}
                  onRemove={() => setObjectives((previous) => previous.filter((_, i) => i !== index))}
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

          <Section title="Combine">
            <label className="text-muted flex flex-wrap items-center gap-2">
              <select value={aggregation} onChange={(event) => setAggregation(event.target.value as TtpForm["aggregation"])} className={INPUT}>
                <option value="geometric_mean">Geometric mean</option>
                <option value="arithmetic_mean">Arithmetic mean</option>
              </select>
              <span className="text-faint text-[11px]">
                {aggregation === "geometric_mean"
                  ? "A molecule must do reasonably on every objective: one zero sinks the total."
                  : "Objectives trade off: a strong score can make up for a weak one."}
              </span>
            </label>
          </Section>
        </div>

        {submit.state === "error" && (
          <div className="mx-4 mb-2 rounded-lg border px-3 py-2" style={{ borderColor: "color-mix(in oklab, var(--color-accent) 35%, transparent)", background: "color-mix(in oklab, var(--color-accent) 8%, transparent)" }}>
            {submit.forbidden ? (
              <>
                <p className="font-medium">Creating profiles from the UI is not enabled on this server yet.</p>
                <p className="text-muted mt-0.5 text-[11px]">
                  The gateway does not allow create_ttp as a direct tool ({submit.message}). Copy the JSON and ask the agent to create it, or try again once the update is deployed.
                </p>
              </>
            ) : (
              <>
                <p className="font-medium">The server did not accept the profile.</p>
                <p className="text-muted mt-0.5 font-mono text-[10px] break-words whitespace-pre-wrap">{submit.message}</p>
              </>
            )}
          </div>
        )}

        <footer className="border-subtle flex flex-wrap items-center gap-2 border-t px-4 py-2.5">
          <div className="min-w-0 flex-1 text-[11px]">
            {attempted && problems.length > 0 ? (
              <ul style={{ color: "var(--color-accent)" }}>
                {problems.slice(0, 3).map((p) => (
                  <li key={p}>{p}</li>
                ))}
                {problems.length > 3 && <li>…and {problems.length - 3} more.</li>}
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
            disabled={submit.state === "busy"}
            className="cursor-pointer rounded-md px-3 py-1 font-semibold text-white transition-opacity hover:opacity-90 disabled:cursor-default disabled:opacity-50 touch:min-h-11"
            style={{ background: "var(--color-accent)" }}
          >
            {submit.state === "busy" ? "Creating…" : "Create profile"}
          </button>
        </footer>
      </div>
    </div>
  );
}

const INPUT = "bg-sunken border-subtle focus:border-strong rounded-md border px-2 py-1 text-xs outline-none";
const BLOCK = `${INPUT} w-full`;
const SECONDARY = "bg-inset text-body hover:bg-sunken border-subtle cursor-pointer rounded-md border px-2.5 py-1 font-medium disabled:cursor-default disabled:opacity-50 touch:min-h-11";
const GHOST = "text-muted hover:text-body hover:bg-inset cursor-pointer rounded-md px-2 py-1 font-medium touch:min-h-11";

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
        {hint && <span className="text-faint">{hint}</span>}
      </span>
      {children}
    </label>
  );
}

function ObjectiveEditor({
  objective,
  onChange,
  onRename,
  onRemove,
}: {
  objective: Objective;
  onChange: (patch: Partial<Objective>) => void;
  onRename: (name: string) => void;
  onRemove: () => void;
}) {
  const [open, setOpen] = useState(false);
  const transform = objective.transform ?? null;
  const source = objectiveSource(objective);
  const colour = source === "native" ? "var(--color-ok)" : "var(--color-structure)";
  const params = objective.service_params ?? {};
  const selector = [params.endpoint, params.target, params.interaction_type].filter((v): v is string => typeof v === "string" && v !== "");
  const what = [objective.component ?? objective.service, ...selector, objective.depends_on && `after ${objective.depends_on}`].filter(Boolean).join(" · ");
  const reference = Array.isArray(objective.params?.smiles) ? String(objective.params.smiles[0] ?? "") : "";

  const setTransform = (next: Transform | null) => onChange({ transform: next });

  return (
    <li className="border-subtle rounded-lg border">
      <div className="flex items-center gap-2 px-2.5 pt-1.5">
        <span aria-hidden="true" className="size-1.5 shrink-0 rounded-full" style={{ background: colour }} title={source} />
        <input
          value={objective.name}
          onChange={(event) => onRename(event.target.value)}
          aria-label="Objective name"
          className="hover:bg-sunken focus:bg-sunken min-w-0 flex-1 rounded px-1 py-0.5 text-xs font-semibold outline-none"
        />
        <span className="text-faint hidden max-w-56 truncate text-[10px] sm:block" title={objective.comment ?? what}>
          {what}
        </span>
        <button type="button" onClick={onRemove} aria-label={`Remove ${objective.name}`} title="Remove" className="text-faint hover:text-body cursor-pointer px-1 text-sm leading-none">
          ×
        </button>
      </div>
      {objective.component === "TanimotoSimilarity" && (
        <label className="flex items-center gap-2 px-2.5 pt-1">
          <span className="text-faint w-12 shrink-0 text-[10px]">reference</span>
          <input
            value={reference}
            onChange={(event) => onChange({ params: { ...(objective.params ?? {}), smiles: event.target.value.trim() ? [event.target.value.trim()] : [] } })}
            placeholder="SMILES to stay close to"
            spellCheck={false}
            className={`${BLOCK} py-0.5 font-mono text-[10px]`}
          />
        </label>
      )}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 px-2.5 py-1.5">
        <label className="flex min-w-44 flex-1 items-center gap-2">
          <span className="text-faint w-12 shrink-0 text-[10px]">weight</span>
          <input
            type="range"
            min={0.05}
            max={1}
            step={0.05}
            value={Math.min(1, objective.weight)}
            onChange={(event) => onChange({ weight: Number(event.target.value) })}
            aria-label={`${objective.name} weight`}
            className="min-w-0 flex-1 accent-[var(--color-table)]"
          />
          <span className="w-8 shrink-0 text-right font-mono text-[11px] tabular-nums">{objective.weight.toFixed(2)}</span>
        </label>
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          title="How the raw value becomes a 0–1 score"
          className="bg-sunken border-subtle hover:border-strong flex shrink-0 cursor-pointer items-center gap-1.5 rounded-full border px-2 py-px font-mono text-[10px]"
        >
          {transform ? transform.type : "raw score"}
          <span aria-hidden="true">{open ? "▴" : "▾"}</span>
        </button>
      </div>
      {open && (
        <div className="border-subtle bg-sunken flex flex-wrap items-start gap-3 border-t px-2.5 py-2">
          <div className="min-w-0 flex-1 space-y-1.5">
            <label className="text-muted flex items-center gap-2 text-[11px]">
              <span className="w-14 shrink-0">transform</span>
              <select
                value={transform?.type ?? ""}
                onChange={(event) => setTransform(event.target.value ? retype(transform, event.target.value) : null)}
                className="bg-raised border-subtle min-w-0 flex-1 rounded-md border px-1 py-0.5 font-mono text-[11px]"
              >
                <option value="">none (score is already 0–1)</option>
                {TRANSFORM_TYPES.filter((t) => t !== "value_mapping").map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
            </label>
            {transform &&
              transformParams(transform.type).map((param) => (
                <label key={`${transform.type}:${param}`} className="text-muted flex items-center gap-2 text-[11px]">
                  <span className="w-14 shrink-0 font-mono">{param}</span>
                  <input
                    type="number"
                    step="any"
                    defaultValue={typeof transform[param] === "number" ? String(transform[param]) : ""}
                    onChange={(event) => {
                      const value = Number.parseFloat(event.target.value);
                      setTransform({ ...transform, [param]: Number.isFinite(value) ? value : undefined });
                    }}
                    className="bg-raised border-subtle w-24 rounded-md border px-1.5 py-0.5 font-mono text-[11px] tabular-nums"
                  />
                </label>
              ))}
            {objective.comment && <p className="text-faint text-[10px]">{objective.comment}</p>}
          </div>
          {transform && <TransformCurve transform={transform} />}
        </div>
      )}
      {!open && transform && (
        <div className="px-2.5 pb-1.5">
          <TransformCurve transform={transform} width={220} height={28} />
        </div>
      )}
    </li>
  );
}
