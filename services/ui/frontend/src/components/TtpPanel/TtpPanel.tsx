// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { callTool } from "../../lib/api";
import {
  TRANSFORM_TYPES,
  applyPrompt,
  asTtp,
  asTtpList,
  changes,
  isDirty,
  latestTtp,
  loadDraft,
  makeDraft,
  objectivesOf,
  parseResult,
  patchObjective,
  payloadError,
  removeObjective,
  retype,
  saveDraft,
  targetRange,
  transformParams,
  ttpUpdates,
  weightShares,
  type Objective,
  type ToolEvent,
  type Transform,
  type Ttp,
  type TtpDraft,
  type TtpSection,
  type TtpSummary,
} from "../../lib/ttp";
import { TransformCurve } from "./TransformCurve";

interface Props {
  /** Edits are kept per session, in localStorage. */
  sessionId: string;
  /** The session's tool calls, from `toolEvents(artifacts.steps)`; picks the TTP the session uses. */
  events?: ToolEvent[];
  /** Put a prompt in the composer: "Apply" asks the agent to adopt the edited profile. */
  onAsk: (prompt: string) => void;
  /** Show this stored TTP instead of the session's. */
  ttpId?: string;
}

/** The Target Product Profile: objectives, target windows and weights, editable. */
export function TtpPanel(props: Props) {
  return <TtpEditor key={props.sessionId} {...props} />;
}

type Remote = { id: string; ttp: Ttp | null; error: string | null };
type SaveState = { state: "idle" } | { state: "busy" } | { state: "error"; message: string } | { state: "saved"; version: number };
type ListState = { state: "loading" } | { state: "error"; message: string } | { state: "ready"; items: TtpSummary[] };

const message = (err: unknown) => (err instanceof Error ? err.message : String(err));

function TtpEditor({ sessionId, events = [], onAsk, ttpId }: Props) {
  const fromSession = useMemo(() => latestTtp(events), [events]);
  const [stored, setStored] = useState<TtpDraft | null>(() => loadDraft(sessionId));
  const [list, setList] = useState<ListState>({ state: "loading" });
  const [choice, setChoice] = useState<string | null>(null);
  const [remote, setRemote] = useState<Remote | null>(null);
  const [reload, setReload] = useState(0);
  const [save, setSave] = useState<SaveState>({ state: "idle" });
  const [revision, setRevision] = useState(0);

  useEffect(() => {
    let cancelled = false;
    callTool("list_ttps")
      .then((response) => !cancelled && setList({ state: "ready", items: asTtpList(response.result) }))
      .catch((err: unknown) => !cancelled && setList({ state: "error", message: message(err) }));
    return () => {
      cancelled = true;
    };
  }, [reload]);

  const firstListed = list.state === "ready" ? (list.items[0]?.id ?? null) : null;
  const selectedId = ttpId ?? choice ?? fromSession.id ?? stored?.base.id ?? firstListed;
  const sessionTtp = fromSession.ttp && fromSession.ttp.id === selectedId ? fromSession.ttp : null;
  const sessionVersion = sessionTtp?.version ?? 0;

  useEffect(() => {
    if (!selectedId) return;
    let cancelled = false;
    callTool("get_ttp", { ttp_id: selectedId })
      .then((response) => {
        if (cancelled) return;
        const error = payloadError(response.result);
        setRemote({ id: selectedId, ttp: error ? null : asTtp(parseResult(response.result)), error });
      })
      .catch((err: unknown) => !cancelled && setRemote({ id: selectedId, ttp: null, error: message(err) }));
    return () => {
      cancelled = true;
    };
  }, [selectedId, reload, sessionVersion]);

  const remoteTtp = remote && remote.id === selectedId ? remote.ttp : null;
  const loading = Boolean(selectedId) && remote?.id !== selectedId;
  const onServer = remoteTtp !== null;
  const candidates = [remoteTtp, sessionTtp, stored && stored.base.id === selectedId ? stored.base : null].filter((t): t is Ttp => t !== null);
  const base = candidates.sort((a, b) => b.version - a.version)[0] ?? null;

  if (!selectedId || (!base && !loading)) {
    return (
      <Shell list={list} selectedId={selectedId} onSelect={setChoice} onRefresh={() => setReload((n) => n + 1)}>
        <p className="text-muted px-6 py-10 text-center text-xs">
          {remote?.error && selectedId
            ? `Could not load TTP "${selectedId}": ${remote.error}`
            : list.state === "error"
              ? `Could not list TTPs: ${list.message}`
              : "No TTP yet. Ask the agent to create a target product profile; its objectives, windows and weights appear here to edit."}
        </p>
      </Shell>
    );
  }
  if (!base) {
    return (
      <Shell list={list} selectedId={selectedId} onSelect={setChoice} onRefresh={() => setReload((n) => n + 1)}>
        <p className="text-muted px-6 py-10 text-center text-xs">Loading {selectedId}…</p>
      </Shell>
    );
  }

  const mine = stored && stored.base.id === base.id ? stored : null;
  const draft = mine?.draft ?? base;
  const dirty = isDirty(base, draft);
  const outdated = dirty && mine !== null && mine.base.version < base.version;
  const diff = changes(base, draft);
  const shares = weightShares(draft);

  const edit = (next: Ttp) => {
    const value = makeDraft(mine?.base ?? base, next, base);
    setStored(value);
    saveDraft(sessionId, value);
    setSave({ state: "idle" });
  };

  const discard = () => {
    setStored(null);
    saveDraft(sessionId, null);
    setRevision((n) => n + 1);
    setSave({ state: "idle" });
  };

  const rebase = () => {
    if (!mine) return;
    const value = makeDraft(base, mine.draft);
    setStored(value);
    saveDraft(sessionId, value);
  };

  const saveDirect = async () => {
    setSave({ state: "busy" });
    try {
      const response = await callTool("update_ttp", { ttp_id: base.id, updates: ttpUpdates(base, draft) });
      const error = payloadError(response.result);
      if (error) {
        setSave({ state: "error", message: error });
        return;
      }
      const payload = parseResult(response.result) as { ttp?: unknown } | null;
      const updated = asTtp(payload?.ttp);
      if (updated) setRemote({ id: base.id, ttp: updated, error: null });
      setStored(null);
      saveDraft(sessionId, null);
      setRevision((n) => n + 1);
      setSave({ state: "saved", version: updated?.version ?? base.version + 1 });
    } catch (err) {
      setSave({ state: "error", message: message(err) });
    }
  };

  const sections: { section: TtpSection; title: string; hint: string }[] = [
    { section: "pre_scoring", title: "Pre-scoring", hint: "fast RDKit descriptors" },
    { section: "scoring", title: "Objectives", hint: draft.scoring.comment ?? "native and external scores" },
  ];

  return (
    <Shell list={list} selectedId={selectedId} onSelect={setChoice} onRefresh={() => setReload((n) => n + 1)}>
      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3">
        <div className="bg-sunken border-subtle flex flex-wrap items-center gap-2 rounded-xl border px-3 py-2">
          <span className="rounded-full border px-1.5 py-px text-[10px] font-semibold" style={{ color: "var(--color-table)", borderColor: "var(--color-table)" }}>
            v{base.version}
          </span>
          <span className="min-w-0 truncate text-xs font-semibold">{base.name}</span>
          <span className="text-faint font-mono text-[10px]">{base.id}</span>
          {!onServer && (
            <span className="text-faint text-[10px]" title={remote?.error ?? undefined}>
              · from this session, not in the TTP store
            </span>
          )}
          <label className="text-muted ml-auto flex items-center gap-1 text-[11px]">
            Aggregation
            <select
              value={draft.scoring.aggregation ?? "geometric_mean"}
              onChange={(event) => edit({ ...draft, scoring: { ...draft.scoring, aggregation: event.target.value as "geometric_mean" | "arithmetic_mean" } })}
              className="bg-raised border-subtle rounded-md border px-1 py-0.5 text-[11px]"
            >
              <option value="geometric_mean">geometric mean</option>
              <option value="arithmetic_mean">arithmetic mean</option>
            </select>
          </label>
          {base.target && (
            <p className="text-faint w-full truncate font-mono text-[10px]">
              target {base.target.protein_sequence.length} aa
              {base.target.pdb_id ? ` · PDB ${base.target.pdb_id}` : ""}
              {base.target.msa_path ? ` · MSA ${base.target.msa_path}` : ""}
            </p>
          )}
        </div>

        {outdated && (
          <p className="bg-sunken border-subtle rounded-lg border px-3 py-2 text-[11px]">
            The stored TTP moved to v{base.version} since you started editing v{mine?.base.version}.{" "}
            <button type="button" onClick={rebase} className="cursor-pointer font-semibold underline">
              Keep my edits on v{base.version}
            </button>{" "}
            or{" "}
            <button type="button" onClick={discard} className="cursor-pointer font-semibold underline">
              discard them
            </button>
            .
          </p>
        )}

        {sections.map(({ section, title, hint }) => {
          const items = objectivesOf(draft, section);
          if (section === "pre_scoring" && items.length === 0 && objectivesOf(base, section).length === 0) return null;
          return (
            <section key={section}>
              <h3 className="text-muted mb-1 flex gap-2 text-[11px] font-medium">
                <span>{title}</span>
                <span className="text-faint truncate">{hint}</span>
                <span className="text-faint ml-auto">{items.length}</span>
              </h3>
              <ul className="space-y-1.5">
                {items.map((objective, index) => {
                  const original = objectivesOf(base, section).find((o) => o.name === objective.name);
                  return (
                    <ObjectiveRow
                      key={`${revision}:${objective.name}`}
                      objective={objective}
                      original={original}
                      share={shares.get(objective.name) ?? 0}
                      changed={diff.changed.includes(objective.name)}
                      onChange={(patch) => edit(patchObjective(draft, section, index, patch))}
                      onRemove={items.length > 1 || section === "pre_scoring" ? () => edit(removeObjective(draft, section, index)) : undefined}
                    />
                  );
                })}
              </ul>
            </section>
          );
        })}

        {diff.removed.length > 0 && <p className="text-faint text-[11px]">Removed: {diff.removed.join(", ")}</p>}

        {(draft.constraints ?? []).length > 0 && (
          <section>
            <h3 className="text-muted mb-1 text-[11px] font-medium">Constraints · SMARTS</h3>
            <div className="flex flex-wrap gap-1">
              {(draft.constraints ?? []).flatMap((constraint, i) =>
                (constraint.smarts ?? []).map((smarts, j) => (
                  <span key={`${i}:${j}`} className="bg-sunken border-subtle rounded-md border px-1.5 py-px font-mono text-[10px]" title={constraint.type}>
                    {smarts}
                  </span>
                )),
              )}
            </div>
          </section>
        )}
      </div>

      <footer className="border-subtle flex shrink-0 flex-wrap items-center gap-2 border-t px-3 py-2 text-[11px]">
        <span className="text-muted min-w-0 flex-1 truncate">
          {save.state === "saved" && !dirty
            ? `Saved as v${save.version}.`
            : save.state === "error"
              ? null
              : dirty
                ? `Edited: ${[...diff.changed, ...diff.removed.map((n) => `−${n}`)].join(", ") || "aggregation"} · kept in this browser`
                : "No edits. Move a weight or a window to start."}
          {save.state === "error" && <span style={{ color: "var(--color-accent)" }}>{save.message}</span>}
        </span>
        <button
          type="button"
          onClick={discard}
          disabled={!dirty}
          className="text-muted hover:text-body hover:bg-inset cursor-pointer rounded-md px-2 py-1 font-medium disabled:cursor-default disabled:opacity-40 touch:min-h-11"
        >
          Discard
        </button>
        {onServer && (
          <button
            type="button"
            onClick={saveDirect}
            disabled={!dirty || save.state === "busy"}
            title="Write the edits to the TTP store with update_ttp"
            className="bg-inset text-muted hover:text-body cursor-pointer rounded-md px-2 py-1 font-medium disabled:cursor-default disabled:opacity-40 touch:min-h-11"
          >
            {save.state === "busy" ? "Saving…" : `Save as v${base.version + 1}`}
          </button>
        )}
        <button
          type="button"
          onClick={() => onAsk(applyPrompt(base, draft))}
          disabled={!dirty}
          title="Ask the agent to update the TTP and use it"
          className="cursor-pointer rounded-md px-2.5 py-1 font-semibold text-white transition-opacity hover:opacity-90 disabled:cursor-default disabled:opacity-40 touch:min-h-11"
          style={{ background: "var(--color-accent)" }}
        >
          Apply
        </button>
      </footer>
    </Shell>
  );
}

function Shell({
  list,
  selectedId,
  onSelect,
  onRefresh,
  children,
}: {
  list: ListState;
  selectedId: string | null;
  onSelect: (id: string) => void;
  onRefresh: () => void;
  children: ReactNode;
}) {
  const items = list.state === "ready" ? list.items : [];
  const options = selectedId && !items.some((t) => t.id === selectedId) ? [{ id: selectedId, name: selectedId, version: 0 }, ...items] : items;
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="border-subtle flex shrink-0 items-center gap-2 border-b px-3 py-2">
        <div className="min-w-0 flex-1">
          <h2 className="text-xs font-semibold">Target product profile</h2>
          <p className="text-faint text-[10px]">
            {list.state === "ready" ? `${items.length} stored TTP${items.length === 1 ? "" : "s"}` : list.state === "loading" ? "Loading…" : "TTP store unavailable"}
          </p>
        </div>
        {options.length > 1 && (
          <select
            value={selectedId ?? ""}
            onChange={(event) => onSelect(event.target.value)}
            aria-label="TTP"
            className="bg-sunken border-subtle max-w-48 rounded-md border px-1.5 py-1 text-[11px]"
          >
            {options.map((t) => (
              <option key={t.id} value={t.id}>
                {t.name}
                {t.version ? ` · v${t.version}` : ""}
              </option>
            ))}
          </select>
        )}
        <button
          type="button"
          onClick={onRefresh}
          className="bg-inset text-muted hover:text-body shrink-0 cursor-pointer rounded-md px-2 py-1 text-[11px] font-medium touch:min-h-11"
        >
          Refresh
        </button>
      </div>
      {children}
    </div>
  );
}

function ObjectiveRow({
  objective,
  original,
  share,
  changed,
  onChange,
  onRemove,
}: {
  objective: Objective;
  original: Objective | undefined;
  share: number;
  changed: boolean;
  onChange: (patch: Partial<Objective>) => void;
  onRemove?: () => void;
}) {
  const [open, setOpen] = useState(false);
  const transform = objective.transform ?? null;
  const params = transform ? transformParams(transform.type) : [];
  const range = targetRange(transform);
  const what = [objective.type, objective.component ?? objective.service, endpointOf(objective), objective.depends_on && `after ${objective.depends_on}`]
    .filter(Boolean)
    .join(" · ");
  const maxWeight = Math.max(2, Math.ceil((original?.weight ?? objective.weight) * 2));

  const setTransform = (next: Transform | null) => onChange({ transform: next });

  return (
    <li className="bg-raised border-subtle shadow-card overflow-hidden rounded-xl border">
      <div className="flex items-center gap-2 px-3 py-1.5">
        <span
          aria-label={changed ? "edited" : undefined}
          className="size-1.5 shrink-0 rounded-full"
          style={{ background: changed ? "var(--color-accent)" : "var(--color-table)" }}
        />
        <div className="min-w-0 flex-1">
          <p className="truncate text-xs font-semibold" title={objective.comment ?? objective.name}>
            {objective.name}
          </p>
          <p className="text-faint truncate text-[10px]">{what}</p>
        </div>
        {range && (
          <span className="text-muted shrink-0 font-mono text-[10px] tabular-nums" title="Target window">
            {range}
          </span>
        )}
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          className="bg-sunken border-subtle hover:border-strong shrink-0 cursor-pointer rounded-full border px-2 py-px font-mono text-[10px]"
          title="Edit the transform"
        >
          {transform?.type ?? "raw"}
        </button>
        {onRemove && (
          <button type="button" onClick={onRemove} aria-label={`Remove ${objective.name}`} title="Remove" className="text-faint hover:text-body cursor-pointer px-1 text-sm leading-none">
            ×
          </button>
        )}
      </div>
      <div className="flex items-center gap-2 px-3 pb-1.5">
        <span className="text-faint w-10 shrink-0 text-[10px]">weight</span>
        <input
          type="range"
          min={0.05}
          max={maxWeight}
          step={0.05}
          value={objective.weight}
          onChange={(event) => onChange({ weight: Number(event.target.value) })}
          aria-label={`${objective.name} weight`}
          className="min-w-0 flex-1 accent-[var(--color-table)]"
        />
        <span className="w-9 shrink-0 text-right font-mono text-[11px] tabular-nums">{objective.weight.toFixed(2)}</span>
        <span className="bg-inset block h-1 w-12 shrink-0 overflow-hidden rounded-full" title={`${(share * 100).toFixed(0)}% of the total weight`}>
          <span className="block h-full rounded-full" style={{ width: `${Math.max(4, share * 100)}%`, background: "var(--color-table)" }} />
        </span>
      </div>
      {open && (
        <div className="border-subtle bg-sunken flex flex-wrap items-start gap-3 border-t px-3 py-2">
          <div className="min-w-0 flex-1 space-y-1.5">
            <label className="text-muted flex items-center gap-2 text-[11px]">
              <span className="w-14 shrink-0">transform</span>
              <select
                value={transform?.type ?? ""}
                onChange={(event) => setTransform(event.target.value ? retype(transform, event.target.value) : null)}
                className="bg-raised border-subtle min-w-0 flex-1 rounded-md border px-1 py-0.5 font-mono text-[11px]"
              >
                <option value="">none (raw score)</option>
                {TRANSFORM_TYPES.map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
                {transform && !(TRANSFORM_TYPES as readonly string[]).includes(transform.type) && <option value={transform.type}>{transform.type}</option>}
              </select>
            </label>
            {transform &&
              params.map((param) => (
                <label key={`${transform.type}:${param}`} className="text-muted flex items-center gap-2 text-[11px]">
                  <span className="w-14 shrink-0 font-mono">{param}</span>
                  <input
                    type="number"
                    step="any"
                    defaultValue={typeof transform[param] === "number" ? String(transform[param]) : ""}
                    onChange={(event) => {
                      const value = Number.parseFloat(event.target.value);
                      if (Number.isFinite(value)) setTransform({ ...transform, [param]: value });
                    }}
                    className="bg-raised border-subtle w-24 rounded-md border px-1.5 py-0.5 font-mono text-[11px] tabular-nums"
                  />
                  {original?.transform && typeof original.transform[param] === "number" && original.transform[param] !== transform[param] && (
                    <span className="text-faint font-mono text-[10px]">was {String(original.transform[param])}</span>
                  )}
                </label>
              ))}
            {transform?.type === "value_mapping" && (
              <p className="text-faint font-mono text-[10px] break-all">{JSON.stringify(transform.mapping ?? {})}</p>
            )}
          </div>
          {transform && <TransformCurve transform={transform} />}
        </div>
      )}
    </li>
  );
}

function endpointOf(objective: Objective): string | null {
  const params = objective.service_params ?? {};
  const value = params.endpoint ?? params.interaction_type ?? params.target;
  return typeof value === "string" ? value : null;
}
