// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { ApiError, callTool } from "../../lib/api";
import { groupByFolder, type Asset } from "../../lib/assets";
import { cleanChain, templateName } from "../../lib/cleanStructure";
import { isPdbId, sessionFromLocation } from "../../lib/rcsb";
import { sharedPath, sharedRcsbCif, sliceRegion, uploadStructure, type Region, type SharedStructure } from "../../lib/region";
import { fetchText, msaMismatch, pdbIdFromPath, readA3mQuery, structureChains, targetFiles, type ProteinChain } from "../../lib/sharedFiles";
import { uploadFile } from "../../lib/uploads";
import {
  asTtp,
  asTtpList,
  buildTtp,
  createdTimes,
  createdTtp,
  defaultObjectives,
  dropObjectives,
  emptyTargetFacts,
  formFromTtp,
  formProblems,
  hasCofolding,
  isAccession,
  mergeUniprot,
  normaliseSequence,
  objectiveCatalogue,
  objectiveGroups,
  parseComponents,
  parseResult,
  parseServices,
  payloadError,
  rememberCreated,
  renameObjective,
  sequenceProblem,
  slugify,
  sortTtps,
  toggleObjective,
  weightRangeOf,
  type Objective,
  type ServiceInfo,
  type TargetFacts,
  type TemplateConfig,
  type Ttp,
  type TtpForm,
  type TtpSummary,
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
import { ObjectivePicker } from "./ObjectivePicker";
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
  /** The stored TTPs, most recently created first. */
  ttps: TtpSummary[];
  models: PriorModel[];
  files: { structures: Asset[]; msas: Asset[] } | null;
  loaded: boolean;
}

type Loading = { state: "idle" } | { state: "busy"; id: string } | { state: "error"; message: string };

type Lookup = { state: "idle" } | { state: "busy" } | { state: "error"; message: string } | { state: "found"; facts: TargetFacts };
type Submit =
  | { state: "idle" }
  | { state: "busy"; label: string }
  | { state: "error"; message: string; forbidden: boolean };
type FileRead<T> = { state: "idle" } | { state: "busy"; path: string } | { state: "error"; path: string; message: string } | { state: "ready"; path: string; value: T };

/** The structure the protein comes from, read for its chains; `path` of its FileRead is where it sits on the shared volume. */
interface Loaded {
  /** The PDB id, or the file's name: what templates are named after. */
  label: string;
  /** Fetched from the PDB, rather than picked or uploaded. */
  fromPdb: boolean;
  pdb: string | null;
  text: string;
  chains: ProteinChain[];
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
];
const RCSB = "rcsb:";
const UPLOAD = "upload:";
const message = (err: unknown) => (err instanceof Error ? err.message : String(err));
const aa = (n: number) => `${n.toLocaleString("en")} aa`;
const fileName = (path: string) => path.split("/").pop() ?? path;
const sourceName = (key: string) => (key.startsWith(RCSB) ? key.slice(RCSB.length) : key.startsWith(UPLOAD) ? key.slice(UPLOAD.length) : fileName(key));
const chainLabel = (chain: ProteinChain) => `${chain.ids.join(", ")} · ${aa(chain.sequence.length)}`;
const loaded = (shared: SharedStructure, label: string, fromPdb: boolean, pdb: string | null): Loaded => ({
  label,
  fromPdb,
  pdb,
  text: shared.text,
  chains: structureChains(shared.path, shared.text),
});

let nextKey = 1;
const rows = (objectives: Objective[]): Row[] => objectives.map((objective) => ({ key: nextKey++, objective }));

/** A form to define a Therapeutic Target Profile; the chat then gets the request to use it, or to run REINVENT on it. */
export function TtpBuilder({ initial, ...props }: Props) {
  const [catalog, setCatalog] = useState<Catalog>({ services: {}, components: [], ttps: [], models: [], files: null, loaded: false });
  const [source, setSource] = useState<{ ttp?: Ttp; key: number }>({ ttp: initial, key: 0 });
  const [loading, setLoading] = useState<Loading>({ state: "idle" });
  const latestLoad = useRef("");

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
        ttps: ttps.status === "fulfilled" ? sortTtps(asTtpList(ttps.value.result), createdTimes()) : [],
        models: models.status === "fulfilled" ? parseMol2MolModels(models.value.result) : [],
        files: assets.status === "fulfilled" ? targetFiles(assets.value.result) : { structures: [], msas: [] },
        loaded: true,
      });
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const load = async (id: string) => {
    latestLoad.current = id;
    setLoading({ state: "busy", id });
    try {
      const response = await callTool("get_ttp", { ttp_id: id });
      if (latestLoad.current !== id) return;
      const error = payloadError(response.result);
      const ttp = error ? null : asTtp(parseResult(response.result));
      if (!ttp) {
        setLoading({ state: "error", message: error ?? `Could not read the TTP "${id}".` });
        return;
      }
      setLoading({ state: "idle" });
      setSource((previous) => ({ ttp, key: previous.key + 1 }));
    } catch (err) {
      if (latestLoad.current === id) setLoading({ state: "error", message: message(err) });
    }
  };

  const saved = <SavedTtps ttps={catalog.ttps} loaded={catalog.loaded} loading={loading} onPick={(id) => void load(id)} />;
  return <BuilderForm key={source.key} {...props} initial={source.ttp} catalog={catalog} saved={saved} />;
}

/** Start from a stored TTP: a compact list that fills the form with a copy of the one picked. */
function SavedTtps({ ttps, loaded, loading, onPick }: { ttps: TtpSummary[]; loaded: boolean; loading: Loading; onPick: (id: string) => void }) {
  if (loaded && !ttps.length) return null;
  return (
    <span className="flex shrink-0 items-center gap-1.5">
      {loading.state === "error" && (
        <span className="max-w-40 truncate text-[11px]" style={{ color: "var(--color-accent)" }} title={loading.message}>
          {loading.message}
        </span>
      )}
      <select
        value=""
        onChange={(event) => event.target.value && onPick(event.target.value)}
        disabled={!loaded || loading.state === "busy"}
        aria-label="Start from a saved TTP"
        title="Fill the form with a copy of a saved TTP"
        className={`${INPUT} text-muted max-w-44 truncate py-0.5 text-[11px]`}
      >
        <option value="">{!loaded ? "Loading…" : loading.state === "busy" ? "Loading TTP…" : "Start from saved…"}</option>
        {ttps.map((t) => (
          <option key={t.id} value={t.id}>
            {t.name} · v{t.version}
          </option>
        ))}
      </select>
    </span>
  );
}

interface FormProps extends Omit<Props, "initial"> {
  initial?: Ttp;
  catalog: Catalog;
  /** The saved-TTP picker, shown in the header. */
  saved: ReactNode;
}

function BuilderForm({ onClose, onCreated, initial, molecules, catalog, saved }: FormProps) {
  const ids = catalog.ttps.map((t) => t.id);
  const [start] = useState(() => (initial ? formFromTtp(initial, ids) : null));
  const [name, setName] = useState(start?.name ?? "");
  const [id, setId] = useState<string | null>(start?.id ?? null);
  const [sequence, setSequence] = useState(start?.sequence ?? "");
  const [region, setRegion] = useState<Region>("full");
  const [range, setRange] = useState({ from: "", to: "" });
  const [entry, setEntry] = useState("");
  const [msaPath, setMsaPath] = useState(start?.msaPath ?? "");
  const [structure, setStructure] = useState<FileRead<Loaded>>(() => (start?.structurePath ? { state: "busy", path: start.structurePath } : { state: "idle" }));
  const [chainId, setChainId] = useState<string | null>(start?.structureChain ?? null);
  const [template, setTemplate] = useState<Template>(() =>
    start?.templates?.length ? { state: "ready", key: "kept", config: start.templates[0], atoms: null } : { state: "off" },
  );
  const [force, setForce] = useState(start?.templates?.[0]?.force ?? false);
  const [threshold, setThreshold] = useState(String(start?.templates?.[0]?.threshold ?? 2));
  const [msa, setMsa] = useState<FileRead<string | null>>(() => (start?.msaPath ? { state: "busy", path: start.msaPath } : { state: "idle" }));
  const [objectives, setObjectives] = useState<Row[]>(() => rows(start?.objectives ?? defaultObjectives(objectiveCatalogue(catalog.services, catalog.components))));
  /** The objectives were picked by hand (or came with a stored TTP): the services loading late leaves them be. */
  const [handPicked, setHandPicked] = useState(Boolean(start));
  const [defaultsFor, setDefaultsFor] = useState(catalog);
  const [lookupOpen, setLookupOpen] = useState(Boolean(start?.sequence));
  const [query, setQuery] = useState("");
  const [organism, setOrganism] = useState("human");
  const [lookup, setLookup] = useState<Lookup>({ state: "idle" });
  const [attempted, setAttempted] = useState(false);
  const [submit, setSubmit] = useState<Submit>({ state: "idle" });
  const [copied, setCopied] = useState(false);
  const [run, setRun] = useState<RunSettings>(() => loadRunSettings());
  const latest = useRef({ structure: start?.structurePath ?? "", msa: start?.msaPath ?? "", template: "" });

  useEffect(() => {
    const path = start?.msaPath;
    if (!path) return;
    readA3mQuery(path).then(
      (value) => latest.current.msa === path && setMsa({ state: "ready", path, value }),
      (err: unknown) => latest.current.msa === path && setMsa({ state: "error", path, message: message(err) }),
    );
  }, [start]);

  useEffect(() => {
    const path = start?.structurePath;
    if (!path) return;
    fetchText(path).then(
      (text) => latest.current.structure === path && setStructure({ state: "ready", path, value: loaded({ path, text }, path, false, start?.pdbId || pdbIdFromPath(path)) }),
      (err: unknown) => latest.current.structure === path && setStructure({ state: "error", path, message: message(err) }),
    );
  }, [start]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const source = structure.state === "ready" ? structure.value : null;
  const chains = source?.chains ?? [];
  const chain = (chainId && chains.find((c) => c.ids.includes(chainId))) || (chains.length === 1 ? chains[0] : null);
  const chosenChain = chain ? (chainId && chain.ids.includes(chainId) ? chainId : chain.ids[0]) : null;
  const structured = structure.state !== "idle";
  const templateKey = source && chosenChain ? `${structure.state === "ready" ? structure.path : ""}#${chosenChain}` : null;
  const templateLive = template.state === "ready" && (template.key === "kept" || template.key === templateKey);
  const templates: TemplateConfig[] =
    template.state === "ready" && templateLive
      ? [
          { ...template.config, force, threshold: force ? Number(threshold) : null },
          ...(template.key === "kept" ? (start?.templates ?? []).slice(1) : []),
        ]
      : [];
  const keptPdb = start?.pdbId && !start.structurePath && sequence === start.sequence ? start.pdbId : "";
  const form: TtpForm = {
    name,
    id: id ?? slugify(name),
    description: start?.description ?? "",
    sequence: structured ? "" : sequence,
    structurePath: structure.state === "ready" ? structure.path : "",
    structureChain: chosenChain,
    pdbId: structured ? (source?.pdb ?? "") : keptPdb,
    msaPath,
    proteinPdbPath: start?.proteinPdbPath,
    templates,
    aggregation: "geometric_mean",
    objectives: objectives.map((r) => r.objective),
    constraints: start?.constraints,
  };
  const proteinProblems =
    structure.state === "busy"
      ? [`${sourceName(structure.path)} is still loading.`]
      : structure.state === "error"
        ? [structure.message]
        : source && !chains.length
          ? [`No protein chain in ${sourceName(source.label)}.`]
          : source && !chain
            ? [`Pick a chain of ${sourceName(source.label)}.`]
            : [];
  const problems = [...proteinProblems, ...formProblems(form, ids)];
  const runIssues = run.enabled ? runProblems(run) : [];
  const catalogue = useMemo(() => objectiveCatalogue(catalog.services, catalog.components), [catalog]);
  const groups = useMemo(() => objectiveGroups(catalogue), [catalogue]);
  if (defaultsFor !== catalog) {
    setDefaultsFor(catalog);
    if (!handPicked) setObjectives(rows(defaultObjectives(catalogue)));
  }
  const residues = structured ? (chain?.sequence ?? "") : normaliseSequence(sequence);
  const cofold = hasCofolding(form.objectives);
  const weights = weightRangeOf(form.objectives);
  const total = form.objectives.reduce((sum, o) => sum + Math.max(0, o.weight), 0) || 1;
  const mol2mol = mol2molChoices(catalog.models);
  const effective = withOfferedModel(run, mol2mol);
  const busy = submit.state === "busy";
  const msaQuery = msa.state === "ready" ? msa.value : null;
  const mismatch = msaMismatch(msaQuery, residues);
  const reference = lookup.state === "found" ? (lookup.facts.uniprotSequence ?? lookup.facts.sequence) : null;
  const cut = reference && range.from && range.to ? sliceRegion(reference, Number(range.from), Number(range.to)) : null;
  const shownRange = region === "range" && cut && "sequence" in cut && cut.sequence === residues;
  const showSequence = !structured && lookupOpen;

  /** The list as rows, keeping the rows of objectives that stay. */
  const keepRows = (list: Objective[]) => {
    setHandPicked(true);
    setObjectives((previous) => list.map((objective) => previous.find((r) => r.objective === objective) ?? { key: nextKey++, objective }));
  };
  const editRows = (list: Objective[]) => {
    setHandPicked(true);
    setObjectives((previous) => list.map((objective, i) => ({ key: previous[i]?.key ?? nextKey++, objective })));
  };
  const updateRun = (patch: Partial<RunSettings>) => {
    const next = { ...run, ...patch };
    setRun(next);
    saveRunSettings(next);
  };

  const makeTemplate = async (key: string, from: Loaded, chainOf: string) => {
    latest.current.template = key;
    setTemplate({ state: "busy", key });
    try {
      const file = templateName(from.label, chainOf);
      const cleaned = cleanChain(key.split("#")[0], from.text, chainOf, file.replace(/\.cif$/, ""));
      const uploaded = await uploadFile(sessionFromLocation(), new File([cleaned.text], file, { type: "chemical/x-mmcif" }));
      if (latest.current.template !== key) return;
      setTemplate({ state: "ready", key, config: { path: sharedPath(uploaded), chain_id: chainOf }, atoms: cleaned.atoms });
    } catch (err) {
      if (latest.current.template === key) setTemplate({ state: "error", key, message: message(err) });
    }
  };

  const pickChain = (picked: ProteinChain) => {
    setChainId(picked.ids[0]);
    if (template.state !== "off" && structure.state === "ready" && source) void makeTemplate(`${structure.path}#${picked.ids[0]}`, source, picked.ids[0]);
  };

  const loadStructure = async (key: string, read: () => Promise<SharedStructure>, label: string, fromPdb: boolean, pdb: string | null) => {
    latest.current.structure = key;
    latest.current.template = "";
    setChainId(null);
    if (template.state !== "ready" || template.key !== "kept") setTemplate({ state: "off" });
    setStructure({ state: "busy", path: key });
    try {
      const shared = await read();
      if (latest.current.structure !== key) return;
      latest.current.structure = shared.path;
      setStructure({ state: "ready", path: shared.path, value: loaded(shared, label, fromPdb, pdb) });
    } catch (err) {
      if (latest.current.structure === key) setStructure({ state: "error", path: key, message: message(err) });
    }
  };

  const fetchEntry = (code: string) => {
    const pdb = code.trim().toUpperCase();
    setEntry(pdb);
    if (isPdbId(pdb)) void loadStructure(`${RCSB}${pdb}`, () => sharedRcsbCif(pdb, sessionFromLocation()), pdb, true, pdb);
  };

  const pickStructure = (path: string) => {
    setEntry("");
    if (path) void loadStructure(path, async () => ({ path, text: await fetchText(path) }), path, false, pdbIdFromPath(path));
    else clearStructure();
  };

  const upload = (file: File) => {
    setEntry("");
    void loadStructure(`${UPLOAD}${file.name}`, () => uploadStructure(file, sessionFromLocation()), file.name, false, pdbIdFromPath(file.name));
  };

  const clearStructure = () => {
    latest.current.structure = "";
    latest.current.template = "";
    setStructure({ state: "idle" });
    setChainId(null);
    setEntry("");
    if (template.state !== "ready" || template.key !== "kept") setTemplate({ state: "off" });
  };

  const toggleTemplate = (on: boolean) => {
    latest.current.template = "";
    if (on && source && templateKey && chosenChain) void makeTemplate(templateKey, source, chosenChain);
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
    else if (next === "range" && reference) applyRange(range.from || "1", range.to || String(reference.length));
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
      if (value && !structured) setSequence((previous) => (normaliseSequence(previous) ? previous : value));
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
      setRegion("full");
      setSequence(facts.uniprotSequence ?? facts.sequence);
      if (!name.trim() && facts.gene) setName(`${facts.gene.toUpperCase()} ${cofold ? "cofolding" : "lead optimisation"}`);
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

  const shown = attempted ? [...problems, ...runIssues] : [];
  const files = catalog.files;
  const picked = structure.state === "idle" || (structure.state === "ready" ? structure.value.fromPdb : structure.path.startsWith(RCSB) || structure.path.startsWith(UPLOAD)) ? "" : structure.path;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Therapeutic Target Profile"
        className="bg-raised border-subtle shadow-card flex max-h-[92vh] w-full max-w-2xl flex-col rounded-xl border text-xs"
      >
        <header className="border-subtle flex items-center gap-2 border-b px-4 py-2.5">
          <h2 className="min-w-0 flex-1 truncate text-sm font-semibold">
            Therapeutic Target Profile
            {initial && <span className="text-faint font-normal"> · copy of {initial.name}</span>}
          </h2>
          {saved}
          <button type="button" onClick={onClose} aria-label="Close" className="text-faint hover:text-body cursor-pointer px-1 text-lg leading-none">
            ×
          </button>
        </header>

        <div className="min-h-0 flex-1 space-y-5 overflow-y-auto px-4 py-4">
          <div className="grid gap-2 sm:grid-cols-[1fr_12rem]">
            <Labelled label="Name">
              <input value={name} onChange={(event) => setName(event.target.value)} placeholder="e.g. BRD4 cofolding lead optimisation" autoFocus className={BLOCK} />
            </Labelled>
            <Labelled label="Id">
              <input value={form.id} onChange={(event) => setId(event.target.value)} placeholder="brd4-cofolding" className={`${BLOCK} font-mono`} spellCheck={false} />
            </Labelled>
          </div>

          <Section title="Protein">
            <div className="flex flex-wrap items-center gap-1.5">
              <div className="min-w-44 flex-1">
                <FileSelect files={files?.structures} value={picked} onChange={pickStructure} label="Structure file" empty="Structure file" detail={() => null} />
              </div>
              <label className={SECONDARY} title="Upload a .pdb or .cif file">
                Upload
                <input
                  type="file"
                  accept=".pdb,.ent,.cif,.mmcif"
                  className="hidden"
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    event.target.value = "";
                    if (file) upload(file);
                  }}
                />
              </label>
              <span className="text-faint text-[11px]">or</span>
              <form
                className="flex items-center gap-1.5"
                onSubmit={(event) => {
                  event.preventDefault();
                  fetchEntry(entry);
                }}
              >
                <input
                  value={entry}
                  onChange={(event) => setEntry(event.target.value)}
                  placeholder="PDB id"
                  aria-label="PDB id"
                  className={`${INPUT} w-20 font-mono`}
                  spellCheck={false}
                />
                <button type="submit" disabled={!isPdbId(entry) || structure.state === "busy"} className={SECONDARY}>
                  Fetch
                </button>
              </form>
              {structured && (
                <button type="button" onClick={clearStructure} aria-label="Clear the structure" title="Clear" className="text-faint hover:text-body cursor-pointer px-1 text-sm leading-none">
                  ×
                </button>
              )}
            </div>

            {structure.state === "busy" && <p className="text-faint text-[11px]">{structure.path.startsWith(RCSB) ? "Fetching" : structure.path.startsWith(UPLOAD) ? "Uploading" : "Reading"} {sourceName(structure.path)}…</p>}
            {structure.state === "error" && <p style={{ color: "var(--color-accent)" }}>{structure.message}</p>}
            {source && chains.length === 0 && <p className="text-faint text-[11px]">No protein chain in {sourceName(source.label)}.</p>}
            {chains.length === 1 && chain && (
              <p className="text-muted text-[11px]" title={chain.source === "construct" ? "The deposited construct" : "The residues with coordinates"}>
                Chain {chain.ids.join(", ")} · {aa(chain.sequence.length)}
              </p>
            )}
            {chains.length > 1 && (
              <div role="radiogroup" aria-label="Protein chain" className="flex flex-wrap items-center gap-1">
                <span className="text-muted mr-1 text-[11px]">Chain</span>
                {chains.map((c) => (
                  <Pill key={c.ids.join()} on={chain === c} title={c.source === "construct" ? "The deposited construct" : "The residues with coordinates"} onClick={() => pickChain(c)}>
                    {chainLabel(c)}
                  </Pill>
                ))}
              </div>
            )}

            {(templateKey || templateLive || (template.state !== "off" && template.key === templateKey)) && (
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

            {!structured && !lookupOpen && (
              <button type="button" onClick={() => setLookupOpen(true)} className="text-muted hover:text-body cursor-pointer text-[11px] underline underline-offset-2">
                or look up a target
              </button>
            )}

            {showSequence && (
              <div className="space-y-2">
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
                    autoFocus={!start}
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
                            title={`Fold the construct of ${pdb}`}
                            className="border-subtle text-muted hover:text-body cursor-pointer rounded-md border px-1.5 py-px font-mono text-[10px]"
                          >
                            {pdb}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {reference && (
                  <div role="radiogroup" aria-label="Region" className="flex flex-wrap items-center gap-1">
                    {REGIONS.map((r) => (
                      <Pill key={r.id} on={region === r.id} onClick={() => chooseRegion(r.id)}>
                        {r.label}
                      </Pill>
                    ))}
                    {region === "range" && (
                      <span className="ml-1 flex flex-wrap items-center gap-1.5 text-[11px]">
                        <input
                          type="number"
                          min={1}
                          max={reference.length}
                          value={range.from}
                          onChange={(event) => applyRange(event.target.value, range.to)}
                          aria-label="First residue"
                          className={`${INPUT} w-16 py-0.5 font-mono`}
                        />
                        <span className="text-muted">to</span>
                        <input
                          type="number"
                          min={1}
                          max={reference.length}
                          value={range.to}
                          onChange={(event) => applyRange(range.from, event.target.value)}
                          aria-label="Last residue"
                          className={`${INPUT} w-16 py-0.5 font-mono`}
                        />
                        {cut && "error" in cut && <span style={{ color: "var(--color-accent)" }}>{cut.error}</span>}
                      </span>
                    )}
                  </div>
                )}

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
                  <p className="text-faint text-[10px]">{sequenceProblem(residues) ?? [aa(residues.length), shownRange && `UniProt ${range.from}–${range.to}`].filter(Boolean).join(" · ")}</p>
                )}
              </div>
            )}

            {(source || showSequence || msaPath) && (
              <div className="flex items-center gap-2">
                <span className="text-muted shrink-0 text-[11px]">MSA</span>
                <div className="min-w-0 flex-1">
                  <FileSelect files={files?.msas} value={msaPath} onChange={(path) => void pickMsa(path)} label="MSA file" empty="Automatic" detail={(a) => (a.queryLength ? aa(a.queryLength) : null)} />
                </div>
              </div>
            )}
            {msa.state === "busy" && <p className="text-faint text-[11px]">Reading {fileName(msa.path)}…</p>}
            {msa.state === "error" && <p style={{ color: "var(--color-accent)" }}>{msa.message}</p>}
            {cofold && residues.length > LONG_SEQUENCE && !mismatch && (
              <p className="text-[11px]" style={{ color: "var(--color-warn)" }}>
                {structured ? "Slow to cofold at this length." : "Slow to cofold at this length: fold residues or a structure's chain instead."}
              </p>
            )}
            {mismatch && msaQuery && (
              <p className="flex flex-wrap items-baseline gap-x-2 text-[11px]" style={{ color: "var(--color-warn)" }}>
                {mismatch}
                {!structured && (
                  <button type="button" onClick={() => setSequence(msaQuery)} className="text-body cursor-pointer underline underline-offset-2">
                    Use the MSA's sequence
                  </button>
                )}
              </p>
            )}
            {start?.proteinPdbPath && <p className="text-faint text-[10px]">Keeps the original's protein_pdb_path, {start.proteinPdbPath}.</p>}
          </Section>

          <Section title="Objectives">
            <ObjectivePicker groups={groups} objectives={form.objectives} onToggle={(e, on) => keepRows(toggleObjective(form.objectives, e, on))} />
            {!catalog.loaded && <p className="text-faint text-[11px]">Loading the services…</p>}
            {cofold && <p className="text-faint text-[11px]">Cofolding is much slower: every molecule is cofolded with the protein.</p>}
            {objectives.length > 0 && (
              <ul className="border-subtle divide-y rounded-lg border">
                {objectives.map((row, index) => (
                  <BuilderObjective
                    key={row.key}
                    objective={row.objective}
                    weights={weights}
                    share={Math.max(0, row.objective.weight) / total}
                    onChange={(patch) => editRows(form.objectives.map((o, i) => (i === index ? { ...o, ...patch } : o)))}
                    onRename={(value) => editRows(renameObjective(form.objectives, index, value))}
                    onRemove={() => keepRows(dropObjectives(form.objectives, (o) => o === row.objective))}
                  />
                ))}
              </ul>
            )}
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
            {shown.length > 0 && (
              <ul style={{ color: "var(--color-accent)" }}>
                {shown.slice(0, 3).map((p) => (
                  <li key={p}>{p}</li>
                ))}
                {shown.length > 3 && <li>…and {shown.length - 3} more.</li>}
              </ul>
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

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="space-y-2">
      <h3 className="text-[11px] font-semibold">{title}</h3>
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
