// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import type { SessionSummary } from "../../types/api";
import {
  confirmText,
  deletableIds,
  fetchSessionData,
  formatBytes,
  purgeSession,
  purgeSummary,
  selectionTotals,
  shortDate,
  type DataGroup,
  type DataItem,
  type PurgeResult,
  type SessionData,
} from "../../lib/sessionData";

interface Props {
  session: SessionSummary;
  onClose: () => void;
  /** Removes the conversation from the list; also called once a purge has deleted it. */
  onDelete: () => Promise<void>;
}

type Step = "browse" | "confirm" | "deleting" | "done";

/**
 * What one conversation produced or used, and deleting it with the data
 * picked here. The gateway decides what may go; items it keeps say why.
 */
export function SessionDataDialog({ session, onClose, onDelete }: Props) {
  const [data, setData] = useState<SessionData | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [step, setStep] = useState<Step>("browse");
  const [result, setResult] = useState<PurgeResult | null>(null);
  const title = session.title || "Untitled conversation";

  useEffect(() => {
    let stale = false;
    fetchSessionData(session.id)
      .then((body) => !stale && setData(body))
      .catch((reason: unknown) => !stale && setProblem(reason instanceof Error ? reason.message : String(reason)));
    return () => {
      stale = true;
    };
  }, [session.id]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape" && step !== "deleting") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, step]);

  const busy = session.busy || Boolean(data?.busy);
  const chosen = useMemo(() => (data ? selectionTotals(data, picked) : { count: 0, bytes: 0 }), [data, picked]);
  const shown = data?.groups.filter((group) => group.items.length > 0) ?? [];

  const toggle = (ids: string[], on: boolean) =>
    setPicked((previous) => {
      const next = new Set(previous);
      for (const id of ids) {
        if (on) next.add(id);
        else next.delete(id);
      }
      return next;
    });

  async function remove() {
    setStep("deleting");
    setProblem(null);
    try {
      const outcome = await purgeSession(session.id, [...picked]);
      setResult(outcome);
      setStep("done");
      await onDelete().catch(() => undefined);
    } catch (reason) {
      setProblem(reason instanceof Error ? reason.message : String(reason));
      setStep("browse");
    }
  }

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      onMouseDown={(event) => event.target === event.currentTarget && step !== "deleting" && onClose()}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={`Data of ${title}`}
        className="bg-raised border-subtle shadow-card flex max-h-[88vh] w-full max-w-2xl flex-col rounded-xl border text-xs"
      >
        <header className="border-subtle flex items-center gap-2 border-b px-4 py-2.5">
          <div className="min-w-0 flex-1">
            <h2 className="text-sm font-semibold">Conversation data</h2>
            <p className="text-faint truncate text-[11px]" title={title}>
              {title}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            disabled={step === "deleting"}
            aria-label="Close"
            className="text-muted hover:text-body grid size-8 cursor-pointer place-items-center rounded-lg text-base disabled:opacity-40"
          >
            ×
          </button>
        </header>

        {step === "done" && result ? (
          <Outcome result={result} onClose={onClose} />
        ) : (
          <>
            {data && <Totals data={data} />}
            <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
              {!data && !problem && <p className="text-faint py-6 text-center">Looking up what this conversation produced…</p>}
              {data && shown.length === 0 && (
                <p className="text-faint py-6 text-center">No files, jobs or TTPs are associated with this conversation.</p>
              )}
              {shown.map((group) => (
                <Group key={group.kind} group={group} picked={picked} onToggle={toggle} disabled={step !== "browse"} />
              ))}
            </div>
            {problem && (
              <p role="alert" className="px-4 pb-2 text-[11px]" style={{ color: "var(--color-accent)" }}>
                {problem}
              </p>
            )}
            <footer className="border-subtle flex flex-wrap items-center gap-2 border-t px-4 py-2.5">
              {step === "confirm" && data ? (
                <>
                  <p className="min-w-0 flex-1 text-[11px]">{confirmText(title, data, picked)}</p>
                  <button type="button" onClick={() => setStep("browse")} className="border-subtle hover:bg-sunken cursor-pointer rounded-md border px-3 py-1.5">
                    Cancel
                  </button>
                  <button
                    type="button"
                    onClick={remove}
                    className="cursor-pointer rounded-md px-3 py-1.5 font-semibold text-white hover:opacity-90"
                    style={{ background: "var(--color-accent)" }}
                  >
                    Delete
                  </button>
                </>
              ) : (
                <>
                  <p className="text-faint min-w-0 flex-1 text-[11px]">
                    {busy
                      ? "A turn is running: stop it before deleting this conversation."
                      : step === "deleting"
                        ? "Deleting…"
                        : "Pick the data to delete with the conversation. Running jobs, shared files and TTPs are never deleted."}
                  </p>
                  {data && data.totals.deletableItems > 0 && (
                    <button
                      type="button"
                      disabled={step !== "browse"}
                      onClick={() => toggle(deletableIds(data), chosen.count < data.totals.deletableItems)}
                      className="border-subtle hover:bg-sunken cursor-pointer rounded-md border px-3 py-1.5 disabled:opacity-50"
                    >
                      {chosen.count < data.totals.deletableItems ? "Select all removable" : "Clear selection"}
                    </button>
                  )}
                  <button
                    type="button"
                    disabled={busy || !data || step !== "browse"}
                    onClick={() => setStep("confirm")}
                    className="cursor-pointer rounded-md px-3 py-1.5 font-semibold text-white hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
                    style={{ background: "var(--color-accent)" }}
                  >
                    {chosen.count ? `Delete with ${chosen.count} item${chosen.count === 1 ? "" : "s"} (${formatBytes(chosen.bytes)})` : "Delete conversation only"}
                  </button>
                </>
              )}
            </footer>
          </>
        )}
      </div>
    </div>,
    document.body,
  );
}

function Totals({ data }: { data: SessionData }) {
  const { totals } = data;
  return (
    <dl className="border-subtle bg-sunken grid grid-cols-2 gap-x-4 gap-y-1 border-b px-4 py-2.5 sm:grid-cols-4">
      <Stat label="Items" value={String(totals.items)} />
      <Stat label="On disk" value={formatBytes(totals.bytes)} />
      <Stat label="Removable" value={`${formatBytes(totals.deletableBytes)} · ${totals.deletableItems}`} />
      <Stat label="Conversation record" value={formatBytes(totals.conversationBytes)} />
    </dl>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <dt className="text-faint text-[10px] font-semibold tracking-wide uppercase">{label}</dt>
      <dd className="truncate text-sm font-semibold tabular-nums">{value}</dd>
    </div>
  );
}

function Group({
  group,
  picked,
  onToggle,
  disabled,
}: {
  group: DataGroup;
  picked: ReadonlySet<string>;
  onToggle: (ids: string[], on: boolean) => void;
  disabled: boolean;
}) {
  const removable = group.items.filter((item) => item.deletable).map((item) => item.id);
  const all = removable.length > 0 && removable.every((id) => picked.has(id));
  return (
    <section className="mb-3">
      <h3 className="mb-1 flex items-center gap-2 text-[10px] font-semibold tracking-wide uppercase">
        {removable.length > 0 ? (
          <input
            type="checkbox"
            checked={all}
            disabled={disabled}
            onChange={(event) => onToggle(removable, event.target.checked)}
            aria-label={`Select every removable item in ${group.label}`}
            className="cursor-pointer"
          />
        ) : (
          <span className="inline-block w-[13px]" aria-hidden="true" />
        )}
        <span className="text-muted flex-1">
          {group.label} <span className="text-faint font-normal normal-case">· {group.items.length}</span>
        </span>
        {group.bytes > 0 && <span className="text-faint font-normal tabular-nums">{formatBytes(group.bytes)}</span>}
      </h3>
      <ul className="border-subtle divide-y rounded-lg border">
        {group.items.map((item) => (
          <Row key={item.id} item={item} checked={picked.has(item.id)} disabled={disabled} onToggle={onToggle} />
        ))}
      </ul>
    </section>
  );
}

function Row({
  item,
  checked,
  disabled,
  onToggle,
}: {
  item: DataItem;
  checked: boolean;
  disabled: boolean;
  onToggle: (ids: string[], on: boolean) => void;
}) {
  return (
    <li className="flex items-start gap-2 px-2.5 py-1.5">
      <input
        type="checkbox"
        checked={item.deletable && checked}
        disabled={!item.deletable || disabled}
        onChange={(event) => onToggle([item.id], event.target.checked)}
        aria-label={`Delete ${item.name}`}
        title={item.reason ?? undefined}
        className="mt-0.5 cursor-pointer disabled:cursor-not-allowed"
      />
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline gap-2">
          {item.url ? (
            <a href={item.url} target="_blank" rel="noreferrer" className="truncate font-mono font-semibold hover:underline" title={item.path ?? item.name}>
              {item.name}
            </a>
          ) : (
            <span className="truncate font-mono font-semibold" title={item.path ?? item.name}>
              {item.name}
            </span>
          )}
          {item.detail && <span className="text-faint truncate text-[11px]">{item.detail}</span>}
        </div>
        {item.links.length > 0 && (
          <div className="mt-0.5 flex flex-wrap gap-1">
            {item.links.map((link) => (
              <a
                key={link.url}
                href={link.url}
                target="_blank"
                rel="noreferrer"
                className="bg-sunken text-muted hover:text-body rounded px-1.5 py-px font-mono text-[10px]"
              >
                {link.label}
              </a>
            ))}
          </div>
        )}
        {!item.deletable && item.reason && <p className="text-faint mt-0.5 text-[10px]">{item.reason}</p>}
      </div>
      <div className="text-faint shrink-0 text-right text-[10px] tabular-nums">
        {item.onDisk && <div className="text-muted text-[11px]">{formatBytes(item.bytes)}</div>}
        <div>{shortDate(item.modified)}</div>
      </div>
    </li>
  );
}

function Outcome({ result, onClose }: { result: PurgeResult; onClose: () => void }) {
  const kept = result.kept.filter((entry) => entry.reason !== "Not selected");
  return (
    <>
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
        <p className="text-sm font-semibold" role="status">
          {purgeSummary(result)}
        </p>
        {result.error && (
          <p role="alert" className="mt-1 text-[11px]" style={{ color: "var(--color-accent)" }}>
            The conversation was deleted, but its data could not be: {result.error}
          </p>
        )}
        {result.removed.length > 0 && (
          <section className="mt-3">
            <h3 className="text-faint mb-1 text-[10px] font-semibold tracking-wide uppercase">Removed</h3>
            <ul className="space-y-0.5">
              {result.removed.map((entry) => (
                <li key={entry.id} className="flex gap-2">
                  <span className="min-w-0 flex-1 truncate font-mono">{entry.name}</span>
                  <span className="text-faint tabular-nums">{formatBytes(entry.bytes)}</span>
                </li>
              ))}
            </ul>
          </section>
        )}
        {kept.length > 0 && (
          <section className="mt-3">
            <h3 className="text-faint mb-1 text-[10px] font-semibold tracking-wide uppercase">Kept</h3>
            <ul className="space-y-0.5">
              {kept.map((entry) => (
                <li key={entry.id} className="flex gap-2">
                  <span className="min-w-0 flex-1 truncate font-mono">{entry.name}</span>
                  <span className="text-faint">{entry.reason}</span>
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>
      <footer className="border-subtle flex justify-end border-t px-4 py-2.5">
        <button type="button" onClick={onClose} className="border-subtle hover:bg-sunken cursor-pointer rounded-md border px-3 py-1.5">
          Close
        </button>
      </footer>
    </>
  );
}
