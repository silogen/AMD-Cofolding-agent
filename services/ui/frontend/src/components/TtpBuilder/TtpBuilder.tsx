// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { ApiError, callTool } from "../../lib/api";
import { groupByFolder, type Asset } from "../../lib/assets";
import { cleanChain, templateName } from "../../lib/cleanStructure";
import { isPdbId, sessionFromLocation } from "../../lib/rcsb";
import { fetchRcsbCif, sliceRegion, uniprotSpans, type Region, type UniprotSpan } from "../../lib/region";
import { fetchText, isCif, msaMismatch, pdbIdFromPath, readA3mQuery, structureChains, targetFiles, type ProteinChain } from "../../lib/sharedFiles";
import { uploadFile } from "../../lib/uploads";
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
  presetOn,
  rememberCreated,
  renameObjective,
  sequenceProblem,
  serviceObjective,
  slugify,
  togglePreset,
  uniqueName,
  weightRangeOf,
  withCofolding,
  type CatalogueEntry,
  type Objective,
  type ServiceInfo,
  type TargetFacts,
  type TemplateConfig,
  type Ttp,
  type TtpForm,
} from "../../lib/ttp";
import {
  loadRunSettings,
  mol2molChoices,
  parseMol2MolModels,
  runProblems,
  saveRunSettings,
  ttpRequest,
  withOfferedModel,
  type PriorModel,
  type RunSettings,
  type SeedMolecule,
} from "../../lib/ttpRun";
import { BuilderObjective } from "./BuilderObjective";
import { RunFields } from "./RunFields";

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
  /** Molecules of the conversation to offer as Mol2Mol seeds, besides its shortlist. */
  molecules?: SeedMolecule[];
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
  files: { structures: Asset[]; msas: Asset[] } | null;
  loaded: boolean;
}

type Lookup = { state: "idle" } | { state: "busy" } | { state: "error"; message: string } | { state: "found"; facts: TargetFacts };
type Submit =
  | { state: "idle" }
  | { state: "busy"; label: string }
  | { state: "error"; message: string; forbidden: boolean };
type FileRead<T> = { state: "idle" } | { state: "busy"; path: string } | { state: "error"; path: string; message: string } | { state: "ready"; path: string; value: T };

/** A structure read for its chains: an RCSB entry (path "rcsb:1M17") or a shared file. */
interface Loaded {
  /** The PDB id, or the shared file's path. */
  label: string;
  /** The name the parsers go by: its extension decides the format. */
  file: string;
  text: string;
  chains: ProteinChain[];
  spans: Map<string, UniprotSpan>;
}

/** The template of the chosen chain; "kept" is the one a duplicated TTP came with. */
type Template =
  | { state: "off" }
  | { state: "busy"; key: string }
  | { state: "error"; key: string; message: string }
  | { state: "ready"; key: string; config: TemplateConfig; atoms: number | null };

const ORGANISMS = ["human", "mouse", "rat", ""];
const LONG_SEQUENCE = 600;
const REGIONS: { id: Region; label: string }[] = [
  { id: "full", label: "Full sequence" },
  { id: "range", label: "Residues" },
  { id: "pdb", label: "Same as PDB" },
];
const RCSB = "rcsb:";
const message = (err: unknown) => (err instanceof Error ? err.message : String(err));
const aa = (n: number) => `${n.toLocaleString("en")} aa`;
const fileName = (path: string) => path.split("/").pop() ?? path;
const sourceName = (key: string) => (key.startsWith(RCSB) ? key.slice(RCSB.length) : fileName(key));
const chainLabel = (chain: ProteinChain) => `${chain.ids.join(", ")} · ${aa(chain.sequence.length)}`;

let nextKey = 1;
const rows = (objectives: Objective[]): Row[] => objectives.map((objective) => ({ key: nextKey++, objective }));

/** A form to define a target product profile; the chat then gets the request to use it, or to run REINVENT on it. */
export function TtpBuilder({ onClose, onCreated, initial, molecules }: Props) {
  const [start] = useState(() => (initial ? formFromTtp(initial) : null));
  const [catalog, setCatalog] = useState<Catalog>({ services: {}, components: [], ids: [], models: [], files: null, loaded: false });
  const [name, setName] = useState(start?.name ?? "");
  const [id, setId] = useState<string | null>(start?.id ?? null);
  const [description, setDescription] = useState(start?.description ?? "");
  const [sequence, setSequence] = useState(start?.sequence ?? "");
  const [region, setRegion] = useState<Region>(start?.pdbId ? "pdb" : "full");
  const [range, setRange] = useState({ from: "", to: "" });
  const [entry, setEntry] = useState(start?.pdbId ?? "");
  const [msaPath, setMsaPath] = useState(start?.msaPath ?? "");
  const [structure, setStructure] = useState<FileRead<Loaded>>({ state: "idle" });
  const [chainId, setChainId] = useState<string | null>(null);
  const [template, setTemplate] = useState<Template>(() =>
    start?.templates?.length ? { state: "ready", key: "kept", config: start.templates[0], atoms: null } : { state: "off" },
  );
  const [force, setForce] = useState(start?.templates?.[0]?.force ?? false);
  const [threshold, setThreshold] = useState(String(start?.templates?.[0]?.threshold ?? 2));
  const [msa, setMsa] = useState<FileRead<string | null>>(() => (start?.msaPath ? { state: "busy", path: start.msaPath } : { state: "idle" }));
  const [aggregation, setAggregation] = useState<TtpForm["aggregation"]>(start?.aggregation ?? "geometric_mean");
  const [objectives, setObjectives] = useState<Row[]>(() => rows(start?.objectives ?? presetObjectives("druglike", {})));
  const [query, setQuery] = useState("");
  const [organism, setOrganism] = useState("human");
  const [lookup, setLookup] = useState<Lookup>({ state: "idle" });
  const [attempted, setAttempted] = useState(false);
  const [submit, setSubmit] = useState<Submit>({ state: "idle" });
  const [copied, setCopied] = useState(false);
  const [run, setRun] = useState<RunSettings>(() => loadRunSettings());
  const latest = useRef({ structure: "", msa: start?.msaPath ?? "", template: "" });

  useEffect(() => {
    let cancelled = false;
    Promise.allSettled([
      callTool("list_external_services"),
      callTool("list_scoring_components"),
      callTool("list_ttps"),
      callTool("list_available_models"),
      callTool("list_shared_assets"),
    ]).then(([services, components, ttps, models, assets]) => {
      if (cancelled) return;
      setCatalog({
        services: services.status === "fulfilled" ? parseServices(services.value.result) : {},
        components: components.status === "fulfilled" ? parseComponents(components.value.result) : [],
        ids: ttps.status === "fulfilled" ? asTtpList(ttps.value.result).map((t) => t.id) : [],
        models: models.status === "fulfilled" ? parseMol2MolModels(models.value.result) : [],
        files: assets.status === "fulfilled" ? targetFiles(assets.value.result) : { structures: [], msas: [] },
        loaded: true,
      });
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const path = start?.msaPath;
    if (!path) return;
    readA3mQuery(path).then(
      (value) => latest.current.msa === path && setMsa({ state: "ready", path, value }),
      (err: unknown) => latest.current.msa === path && setMsa({ state: "error", path, message: message(err) }),
    );
  }, [start]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const loaded = structure.state === "ready" ? structure.value : null;
  const chains = loaded?.chains ?? [];
  const chain = chains.find((c) => c.ids[0] === chainId) ?? null;
  const templateKey = loaded && chainId ? `${structure.state === "ready" ? structure.path : ""}#${chainId}` : null;
  const templateLive = template.state === "ready" && (template.key === "kept" || template.key === templateKey);
  const templates: TemplateConfig[] =
    template.state === "ready" && templateLive
      ? [
          { ...template.config, force, threshold: force ? Number(threshold) : null },
          ...(template.key === "kept" ? (start?.templates ?? []).slice(1) : []),
        ]
      : [];
  const sourcePdb = structure.state === "idle" ? entry.trim() : structure.path.startsWith(RCSB) ? structure.path.slice(RCSB.length) : (pdbIdFromPath(structure.path) ?? "");
  const form: TtpForm = {
    name,
    id: id ?? slugify(name),
    description,
    sequence,
    pdbId: region === "pdb" ? sourcePdb : "",
    msaPath,
    proteinPdbPath: start?.proteinPdbPath,
    templates,
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
  const mol2mol = mol2molChoices(catalog.models);
  const effective = withOfferedModel(run, mol2mol);
  const busy = submit.state === "busy";
  const presets = PRESETS.filter((p) => p.id !== "cofolding");
  const msaQuery = msa.state === "ready" ? msa.value : null;
  const mismatch = msaMismatch(msaQuery, residues);
  const reference = lookup.state === "found" ? (lookup.facts.uniprotSequence ?? lookup.facts.sequence) : null;
  const accession = lookup.state === "found" ? lookup.facts.uniprot : null;
  const span = chain && loaded ? (loaded.spans.get(chain.ids[0]) ?? null) : null;
  const sameProtein = !span?.accession || !accession || span.accession === accession;
  const cut = reference && range.from && range.to ? sliceRegion(reference, Number(range.from), Number(range.to)) : null;
  const shownChain = region === "pdb" && chain && chain.sequence === residues ? chain : null;
  const shownRange = region === "range" && cut && "sequence" in cut && cut.sequence === residues;

  /** The list as rows, keeping the rows (and their editing state) of objectives that stay. */
  const keepRows = (list: Objective[]) =>
    setObjectives((previous) => list.map((objective) => ({ key: previous.find((r) => r.objective === objective)?.key ?? nextKey++, objective })));
  const setList = (list: Objective[]) => setObjectives((previous) => list.map((objective, i) => ({ key: previous[i]?.key ?? nextKey++, objective })));
  const updateRun = (patch: Partial<RunSettings>) => {
    const next = { ...run, ...patch };
    setRun(next);
    saveRunSettings(next);
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
    if (target?.service === "boltz") keepRows(withCofolding(form.objectives, catalog.services, false));
    else setObjectives((previous) => previous.filter((_, i) => i !== index));
  };

  const makeTemplate = async (key: string, source: Loaded, chainOf: string) => {
    latest.current.template = key;
    setTemplate({ state: "busy", key });
    try {
      const fileName = templateName(source.label, chainOf);
      const cleaned = cleanChain(source.file, source.text, chainOf, fileName.replace(/\.cif$/, ""));
      const file = await uploadFile(sessionFromLocation(), new File([cleaned.text], fileName, { type: "chemical/x-mmcif" }));
      if (latest.current.template !== key) return;
      const path = decodeURIComponent(file.url.replace(/^\/api\/artifacts\//, ""));
      setTemplate({ state: "ready", key, config: { path, chain_id: chainOf }, atoms: cleaned.atoms });
    } catch (err) {
      if (latest.current.template === key) setTemplate({ state: "error", key, message: message(err) });
    }
  };

  const adoptChain = (key: string, source: Loaded, picked: ProteinChain, templated: boolean) => {
    setChainId(picked.ids[0]);
    setSequence(picked.sequence);
    if (templated) void makeTemplate(`${key}#${picked.ids[0]}`, source, picked.ids[0]);
  };

  const loadStructure = async (key: string, file: string, label: string, read: () => Promise<string>) => {
    latest.current.structure = key;
    latest.current.template = "";
    setRegion("pdb");
    setChainId(null);
    setTemplate({ state: "off" });
    setStructure({ state: "busy", path: key });
    try {
      const text = await read();
      if (latest.current.structure !== key) return;
      const source: Loaded = { label, file, text, chains: structureChains(file, text), spans: isCif(file, text) ? uniprotSpans(text) : new Map() };
      setStructure({ state: "ready", path: key, value: source });
      if (source.chains.length === 1) adoptChain(key, source, source.chains[0], false);
    } catch (err) {
      if (latest.current.structure === key) setStructure({ state: "error", path: key, message: message(err) });
    }
  };

  const fetchEntry = (id: string) => {
    const code = id.trim().toUpperCase();
    setEntry(code);
    if (code) void loadStructure(`${RCSB}${code}`, `${code}.cif`, code, () => fetchRcsbCif(code, sessionFromLocation()));
  };

  const pickStructure = (path: string) => {
    if (path) {
      void loadStructure(path, path, path, () => fetchText(path));
      return;
    }
    latest.current.structure = "";
    setStructure({ state: "idle" });
    setChainId(null);
    if (template.state !== "ready" || template.key !== "kept") setTemplate({ state: "off" });
  };

  const toggleTemplate = (on: boolean) => {
    latest.current.template = "";
    if (on && loaded && templateKey && chainId) void makeTemplate(templateKey, loaded, chainId);
    else setTemplate({ state: "off" });
  };

  const applyRange = (from: string, to: string) => {
    setRange({ from, to });
    const next = reference ? sliceRegion(reference, Number(from), Number(to)) : null;
    if (next && "sequence" in next) setSequence(next.sequence);
  };

  const chooseRegion = (next: Region) => {
    setRegion(next);
    if (next === "full" && reference) setSequence(reference);
    else if (next === "range" && reference) {
      const mapped = span && sameProtein && span.end <= reference.length ? span : null;
      applyRange(range.from || String(mapped?.start ?? 1), range.to || String(mapped?.end ?? reference.length));
    } else if (next === "pdb" && chain) setSequence(chain.sequence);
  };

  const pickMsa = async (path: string) => {
    latest.current.msa = path;
    setMsaPath(path);
    if (!path) {
      setMsa({ state: "idle" });
      return;
    }
    setMsa({ state: "busy", path });
    try {
      const value = await readA3mQuery(path);
      if (latest.current.msa !== path) return;
      setMsa({ state: "ready", path, value });
      if (value) setSequence((previous) => (normaliseSequence(previous) ? previous : value));
    } catch (err) {
      if (latest.current.msa === path) setMsa({ state: "error", path, message: message(err) });
    }
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
      setRange({ from: "", to: "" });
      if (region !== "pdb" || !chain) {
        setRegion("full");
        setSequence(facts.uniprotSequence ?? facts.sequence);
      }
      const purpose = cofold ? "cofolding" : (presets.find((p) => presetOn(form.objectives, p.id, catalog.services))?.label.toLowerCase() ?? "profile");
      if (!name.trim() && facts.gene) setName(`${facts.gene.toUpperCase()} ${purpose}`);
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
  const files = catalog.files;

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
            <div className="space-y-2">
              <div className="grid gap-2 sm:grid-cols-[1fr_12rem]">
                <Labelled label="Name">
                  <input value={name} onChange={(event) => setName(event.target.value)} placeholder="e.g. BRD4 cofolding lead optimisation" autoFocus className={BLOCK} />
                </Labelled>
                <Labelled label="Id">
                  <input value={form.id} onChange={(event) => setId(event.target.value)} placeholder="brd4-cofolding" className={`${BLOCK} font-mono`} spellCheck={false} />
                </Labelled>
              </div>
              <input value={description} onChange={(event) => setDescription(event.target.value)} placeholder="Description (optional)" aria-label="Description" className={BLOCK} />
            </div>

            <Section title="Target">
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
                          onClick={() => fetchEntry(pdb)}
                          aria-pressed={region === "pdb" && form.pdbId === pdb}
                          title={`Fold the construct of ${pdb}`}
                          className={`cursor-pointer rounded-md border px-1.5 py-px font-mono text-[10px] ${region === "pdb" && form.pdbId === pdb ? "border-strong text-body" : "border-subtle text-muted hover:text-body"}`}
                        >
                          {pdb}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              )}

              <div role="radiogroup" aria-label="Region" className="flex flex-wrap items-center gap-1">
                <span className="text-muted mr-1 text-[11px]">Region</span>
                {REGIONS.map((r) => (
                  <Pill
                    key={r.id}
                    on={region === r.id}
                    disabled={r.id === "range" && !reference}
                    title={r.id === "range" && !reference ? "Look up the target first: residues are numbered as in UniProt" : undefined}
                    onClick={() => chooseRegion(r.id)}
                  >
                    {r.label}
                  </Pill>
                ))}
              </div>

              {region === "range" && reference && (
                <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
                  <input
                    type="number"
                    min={1}
                    max={reference.length}
                    value={range.from}
                    onChange={(event) => applyRange(event.target.value, range.to)}
                    aria-label="First residue"
                    className={`${INPUT} w-20 font-mono`}
                  />
                  <span className="text-muted">to</span>
                  <input
                    type="number"
                    min={1}
                    max={reference.length}
                    value={range.to}
                    onChange={(event) => applyRange(range.from, event.target.value)}
                    aria-label="Last residue"
                    className={`${INPUT} w-20 font-mono`}
                  />
                  {cut && "error" in cut ? <span style={{ color: "var(--color-accent)" }}>{cut.error}</span> : <span className="text-faint">UniProt numbering</span>}
                </div>
              )}

              {region === "pdb" && (
                <div className="flex flex-wrap items-center gap-1.5">
                  <form
                    className="flex items-center gap-1.5"
                    onSubmit={(event) => {
                      event.preventDefault();
                      fetchEntry(entry);
                    }}
                  >
                    <input value={entry} onChange={(event) => setEntry(event.target.value)} placeholder="PDB id, e.g. 1M17" aria-label="PDB id" className={`${INPUT} w-36 font-mono`} spellCheck={false} />
                    <button type="submit" disabled={!isPdbId(entry) || structure.state === "busy"} className={SECONDARY}>
                      Fetch
                    </button>
                  </form>
                  <span className="text-faint text-[11px]">or</span>
                  <div className="min-w-40 flex-1">
                    <FileSelect
                      files={files?.structures}
                      value={structure.state !== "idle" && !structure.path.startsWith(RCSB) ? structure.path : ""}
                      onChange={pickStructure}
                      label="Structure file"
                      empty="A structure on the shared volume"
                      detail={() => null}
                    />
                  </div>
                </div>
              )}
              {region === "pdb" && structure.state === "busy" && <p className="text-faint text-[11px]">{structure.path.startsWith(RCSB) ? "Fetching" : "Reading"} {sourceName(structure.path)}…</p>}
              {region === "pdb" && structure.state === "error" && <p style={{ color: "var(--color-accent)" }}>{structure.message}</p>}
              {region === "pdb" && structure.state === "ready" && chains.length === 0 && <p className="text-faint text-[11px]">No protein chain in {sourceName(structure.path)}.</p>}
              {region === "pdb" && chains.length > 0 && (
                <div role="radiogroup" aria-label="Protein chain" className="flex flex-wrap items-center gap-1">
                  <span className="text-muted mr-1 text-[11px]">Chain</span>
                  {chains.map((c) => (
                    <Pill
                      key={c.ids.join()}
                      on={chain === c}
                      title={c.source === "construct" ? "Deposited construct sequence" : "Residues with coordinates"}
                      onClick={() => structure.state === "ready" && loaded && adoptChain(structure.path, loaded, c, template.state !== "off")}
                    >
                      {chainLabel(c)}
                    </Pill>
                  ))}
                </div>
              )}

              {((region === "pdb" && chain) || templateLive || (template.state !== "off" && template.key === templateKey)) && (
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px]">
                  <span className="flex items-center gap-1.5">
                    <Switch on={template.state !== "off"} label="Use as template" disabled={template.state === "off" && !templateKey} onChange={toggleTemplate} />
                    <span className={template.state === "off" ? "text-muted" : "text-body"}>Use as template</span>
                  </span>
                  {template.state === "busy" && <span className="text-faint">Saving a cleaned copy…</span>}
                  {template.state === "error" && <span style={{ color: "var(--color-accent)" }}>{template.message}</span>}
                  {template.state === "ready" && (
                    <span className="text-faint truncate" title={template.config.path}>
                      {[fileName(template.config.path), template.config.chain_id && `chain ${template.config.chain_id}`, template.atoms !== null && `${template.atoms.toLocaleString("en")} atoms`].filter(Boolean).join(" · ")}
                    </span>
                  )}
                  {template.state === "ready" && (
                    <span className="flex items-center gap-1">
                      <label className="text-muted flex cursor-pointer items-center gap-1">
                        <input type="checkbox" checked={force} onChange={(event) => setForce(event.target.checked)} />
                        Force
                      </label>
                      {force && (
                        <>
                          <input
                            type="number"
                            min={0.1}
                            step={0.5}
                            value={threshold}
                            onChange={(event) => setThreshold(event.target.value)}
                            aria-label="Template threshold (Å)"
                            title="How far the cofold may deviate from the template"
                            className={`${INPUT} w-14 px-1 py-0.5 font-mono text-[11px]`}
                          />
                          <span className="text-muted">Å</span>
                        </>
                      )}
                    </span>
                  )}
                </div>
              )}

              <Labelled label="MSA">
                <FileSelect files={files?.msas} value={msaPath} onChange={(path) => void pickMsa(path)} label="MSA file" empty="Automatic" detail={(a) => (a.queryLength ? aa(a.queryLength) : null)} />
              </Labelled>
              {msa.state === "busy" && <p className="text-faint text-[11px]">Reading {fileName(msa.path)}…</p>}
              {msa.state === "error" && <p style={{ color: "var(--color-accent)" }}>{msa.message}</p>}

              <textarea
                value={sequence}
                onChange={(event) => setSequence(event.target.value)}
                rows={2}
                spellCheck={false}
                aria-label="Protein sequence"
                placeholder="Protein sequence (FASTA is fine)"
                className={`${BLOCK} resize-y font-mono text-[10px] break-all`}
              />
              {residues && (
                <p className="text-faint text-[10px]">
                  {sequenceProblem(residues) ??
                    [
                      aa(residues.length),
                      shownRange && `residues ${range.from}–${range.to}`,
                      shownChain && structure.state === "ready" && `chain ${shownChain.ids[0]} of ${sourceName(structure.path)}`,
                      shownChain && span && `${sameProtein ? "UniProt" : span.accession} ${span.start}–${span.end}`,
                    ]
                      .filter(Boolean)
                      .join(" · ")}
                </p>
              )}
              {cofold && residues.length > LONG_SEQUENCE && !mismatch && (
                <p className="text-[11px]" style={{ color: "var(--color-warn)" }}>
                  {region === "full" ? "Slow to cofold at this length: fold residues or a PDB chain instead." : "Slow to cofold at this length."}
                </p>
              )}
              {mismatch && msaQuery && (
                <p className="flex flex-wrap items-baseline gap-x-2 text-[11px]" style={{ color: "var(--color-warn)" }}>
                  {mismatch}
                  <button type="button" onClick={() => setSequence(msaQuery)} className="text-body cursor-pointer underline underline-offset-2">
                    Use the MSA's sequence
                  </button>
                </p>
              )}
              {start?.proteinPdbPath && <p className="text-faint text-[10px]">Keeps the original's protein_pdb_path, {start.proteinPdbPath}.</p>}
            </Section>

            <div className="border-subtle flex items-start gap-3 rounded-lg border px-3 py-2.5">
              <div className="min-w-0 flex-1">
                <p className="font-semibold">Cofold</p>
                {(boltzMissing || cofold) && (
                  <p className="text-faint text-[11px] leading-snug">{boltzMissing ? "Not configured on this server." : "Much slower: every molecule is cofolded with the target."}</p>
                )}
                {cofold && !residues && (
                  <p className="mt-0.5 text-[11px]" style={{ color: "var(--color-warn)" }}>
                    Needs the target's protein sequence.
                  </p>
                )}
              </div>
              <Switch
                on={cofold}
                label="Cofold"
                disabled={!catalog.loaded || (boltzMissing && !cofold)}
                onChange={(on) => keepRows(withCofolding(form.objectives, catalog.services, on))}
              />
            </div>

            <Section
              title="Objectives"
              aside={
                <select
                  value={aggregation}
                  onChange={(event) => setAggregation(event.target.value as TtpForm["aggregation"])}
                  aria-label="Combine scores"
                  title={aggregation === "geometric_mean" ? "Every objective counts: one zero sinks the total." : "Objectives trade off: a strong score can make up for a weak one."}
                  className={`${INPUT} py-0.5 text-[11px]`}
                >
                  <option value="geometric_mean">Geometric mean</option>
                  <option value="arithmetic_mean">Arithmetic mean</option>
                </select>
              }
            >
              <div className="flex flex-wrap items-center gap-1.5">
                {presets.map((p) => {
                  const on = presetOn(form.objectives, p.id, catalog.services);
                  return (
                    <button
                      key={p.id}
                      type="button"
                      onClick={() => keepRows(togglePreset(form.objectives, p.id, catalog.services, !on))}
                      aria-pressed={on}
                      title={p.hint}
                      className={`cursor-pointer rounded-full border px-2 py-px text-[11px] ${on ? "border-strong bg-sunken text-body font-medium" : "border-subtle text-muted hover:text-body"}`}
                    >
                      {p.label}
                    </button>
                  );
                })}
              </div>
              <ul className="space-y-1.5">
                {objectives.map((row, index) => (
                  <BuilderObjective
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
              {(start?.constraints?.length ?? 0) > 0 && <p className="text-faint text-[11px]">Keeps the {start?.constraints?.length} structural constraint(s) of the original.</p>}
            </Section>

            <section className="border-subtle space-y-2.5 rounded-lg border px-3 py-2.5">
              <div className="flex items-center gap-3">
                <p className="min-w-0 flex-1 font-semibold">Run REINVENT</p>
                <Switch on={run.enabled} label="Run REINVENT" onChange={(enabled) => updateRun({ enabled })} />
              </div>
              {run.enabled && <RunFields run={run} models={mol2mol} cofold={cofold} molecules={molecules} onChange={updateRun} />}
            </section>
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
                  problems.length > 0 && <span className="text-faint">{`${problems.length} thing${problems.length === 1 ? "" : "s"} left to fill in.`}</span>
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

/** A shared file picker, grouped by top-level folder. */
function FileSelect({
  files,
  value,
  onChange,
  label,
  empty = "None",
  detail,
}: {
  files?: Asset[];
  value: string;
  onChange: (path: string) => void;
  label: string;
  empty?: string;
  detail: (asset: Asset) => string | null;
}) {
  const folders = groupByFolder(files ?? []);
  const listed = (files ?? []).some((a) => a.path === value);
  const names = (files ?? []).map((a) => a.name);
  const shown = (a: Asset) => (names.indexOf(a.name) !== names.lastIndexOf(a.name) ? `${a.dir.split("/").pop()}/${a.name}` : a.name);
  return (
    <select value={value} onChange={(event) => onChange(event.target.value)} aria-label={label} disabled={!files || (!files.length && !value)} className={`${BLOCK} truncate`}>
      <option value="">{!files ? "Loading…" : files.length ? empty : "None on the shared volume"}</option>
      {value && !listed && <option value={value}>{fileName(value)}</option>}
      {folders.map((f) => (
        <optgroup key={f.folder} label={f.folder}>
          {f.assets.map((a) => (
            <option key={a.path} value={a.path} title={a.path}>
              {[shown(a), detail(a)].filter(Boolean).join(" · ")}
            </option>
          ))}
        </optgroup>
      ))}
    </select>
  );
}

function Pill({ on, disabled, title, onClick, children }: { on: boolean; disabled?: boolean; title?: string; onClick: () => void; children: ReactNode }) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={on}
      disabled={disabled}
      title={title}
      onClick={onClick}
      className={`cursor-pointer rounded-full border px-2 py-px text-[11px] disabled:cursor-default disabled:opacity-50 ${on ? "border-strong bg-sunken text-body font-medium" : "border-subtle text-muted hover:text-body"}`}
    >
      {children}
    </button>
  );
}

function Section({ title, aside, children }: { title: string; aside?: ReactNode; children: ReactNode }) {
  return (
    <section className="space-y-2">
      <h3 className="flex items-center gap-2 text-[11px] font-semibold">
        <span className="flex-1">{title}</span>
        {aside}
      </h3>
      {children}
    </section>
  );
}

function Labelled({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block min-w-0 space-y-0.5">
      <span className="text-muted flex gap-2 text-[11px]">{label}</span>
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
