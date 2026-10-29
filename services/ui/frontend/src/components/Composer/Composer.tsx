// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { forwardRef, Fragment, lazy, memo, Suspense, useEffect, useImperativeHandle, useMemo, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import type { Effort, ModelOption } from "../../types/api";
import type { Theme } from "../../hooks/useTheme";
import { useRDKit } from "../../hooks/useRDKit";
import { availableCommands, firstPlaceholder, VIEW_PREFIX, type SlashCommand } from "../../lib/commands";
import { canonical, loadRDKit, withMol } from "../../lib/rdkit";
import { findSmiles, replaceToken } from "../../lib/smiles";
import { onSketchRequest, openSketcher, type SketchRequest } from "../../lib/sketcherBus";
import { ACCEPTED_FILES, fileKind, moleculesFromFile } from "../../lib/dropFiles";
import { modelLabel, pickEffort, useModels } from "../../lib/models";
import type { SeedMolecule } from "../../lib/ttpRun";
import { uploadedStructure, uploadFile } from "../../lib/uploads";
import { addUserStructure } from "../../lib/userStructures";
import { sendToViewer } from "../../lib/viewerBus";
import { MolDepiction } from "../MolDepiction/MolDepiction";
import { AutoButton } from "./AutoButton";

const Sketcher = lazy(() => import("../Sketcher/Sketcher").then((module) => ({ default: module.Sketcher })));
const TtpBuilder = lazy(() => import("../TtpBuilder/TtpBuilder").then((module) => ({ default: module.TtpBuilder })));

interface Props {
  value: string;
  onChange: (text: string) => void;
  onSend: (text: string, effort: Effort | null) => boolean;
  onStop: () => void;
  busy: boolean;
  connected: boolean;
  tools: string[];
  theme: Theme;
  /** The conversation's id, null before it has one; uploads go to its folder. */
  sessionId: string | null;
  /** The model answering, for the model pill. */
  model: string | null;
  /** The gateway's effort, used until the user picks one. */
  defaultEffort: string | null;
  /** The conversation's molecules, offered as seeds in the TTP builder. */
  molecules?: SeedMolecule[];
  /** End of the agent's autonomy budget, ISO 8601 UTC; null when it has none. */
  autoUntil?: string | null;
  /** Give the agent *minutes* on its own, 0 to stop. Without it the button is hidden. */
  onAuto?: (minutes: number) => void;
}

export interface ComposerHandle {
  /** Put *text* in the box, focus it and select its first placeholder. */
  fill: (text: string) => void;
}

type Attachment =
  | { id: number; kind: "molecules"; name: string; smiles: string[] }
  | { id: number; kind: "structure"; name: string; path: string | null; error: string | null };

const EFFORTS: { value: Effort; label: string; hint: string }[] = [
  { value: "low", label: "Low", hint: "Fastest answers" },
  { value: "medium", label: "Medium", hint: "Balanced" },
  { value: "high", label: "High", hint: "Thinks longest" },
];
const EFFORT_VALUES = EFFORTS.map((option) => option.value);
const EFFORT_KEY = "cofolding-effort";
/** Files with up to this many molecules go into the message as SMILES; longer lists stay one attachment. */
const INLINE_MOLECULES = 6;

const effortLabel = (value: Effort) => EFFORTS.find((option) => option.value === value)?.label ?? value;
const effortHint = (value: Effort) => EFFORTS.find((option) => option.value === value)?.hint ?? "";

function isEffort(value: unknown): value is Effort {
  return value === "low" || value === "medium" || value === "high";
}

function storedEffort(): Effort | null {
  try {
    const value = localStorage.getItem(EFFORT_KEY);
    return isEffort(value) ? value : null;
  } catch {
    return null;
  }
}

let attachmentId = 0;

/**
 * The message box.
 *
 * Typing "/" opens the commands, which insert a prompt template with its
 * placeholders selected one after another (Tab jumps to the next one).
 * SMILES in the text show as chips, checked by RDKit before they are sent.
 * Files dropped on the chat or attached are read here: molecule lists become
 * SMILES, structures are uploaded and opened in the viewer. While the agent
 * works the send button becomes Stop: a turn can run for an hour, and
 * stopping must not need a page reload.
 */
export const Composer = memo(forwardRef<ComposerHandle, Props>(function Composer(
  { value, onChange, onSend, onStop, busy, connected, tools, theme, sessionId, model, defaultEffort, molecules, autoUntil, onAuto },
  ref,
) {
  const rootRef = useRef<HTMLDivElement>(null);
  const areaRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const valueRef = useRef(value);
  const dropRef = useRef<(data: DataTransfer) => void>(() => {});
  const [highlight, setHighlight] = useState(0);
  const [dismissed, setDismissed] = useState(false);
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [problem, setProblem] = useState<string | null>(null);
  const [sketch, setSketch] = useState<SketchRequest | null>(null);
  const [ttpOpen, setTtpOpen] = useState(false);
  const [chosenEffort, setChosenEffort] = useState<Effort | null>(storedEffort);
  const [modelMenu, setModelMenu] = useState(false);
  const [dropArea, setDropArea] = useState<DOMRect | null>(null);
  const models = useModels();

  const served = models.state;
  const currentKey = served?.current ?? model;
  // Without the endpoint the ready event still named one model: show that
  // alone, with the thinking levels the UI has always offered.
  const rows: ModelOption[] = served
    ? served.models
    : model
      ? [{ key: model, id: model, label: model.split("/").pop() ?? model, efforts: EFFORT_VALUES, defaultEffort: null }]
      : [];
  const current = rows.find((row) => row.key === currentKey) ?? null;
  const effort = pickEffort(current?.efforts ?? [], chosenEffort, isEffort(defaultEffort) ? defaultEffort : "medium");
  /** The model being loaded, named for the pill; nothing is while it is null. */
  const loading = models.pending ? modelLabel(served, models.pending) : null;
  const canSwitch = served !== null && served.switchable && !loading;
  const commands = useMemo(() => availableCommands(tools), [tools]);
  const slash = /^\/\S*$/.test(value) && !dismissed;
  const matches = slash ? commands.filter((command) => command.cmd.startsWith(value.toLowerCase())) : [];
  const smiles = useMemo(() => findSmiles(value), [value]);
  const { rdkit } = useRDKit(smiles.length > 0);
  const validity = useMemo(
    () => new Map(smiles.map((entry) => [entry, rdkit ? withMol(rdkit, entry, () => true) === true : null])),
    [smiles, rdkit],
  );
  const invalid = smiles.filter((entry) => validity.get(entry) === false);
  const uploading = attachments.some((item) => item.kind === "structure" && !item.path && !item.error);

  useEffect(() => {
    valueRef.current = value;
    const area = areaRef.current;
    if (!area) return;
    area.style.height = "auto";
    area.style.height = `${Math.min(area.scrollHeight, 220)}px`;
  }, [value]);

  useEffect(() => onSketchRequest(setSketch), []);

  const selectPlaceholder = (text: string, from = 0) => {
    const area = areaRef.current;
    if (!area) return false;
    const found = firstPlaceholder(text.slice(from));
    if (!found) return false;
    area.setSelectionRange(from + found[0], from + found[1]);
    return true;
  };

  const focusEnd = (text: string) =>
    requestAnimationFrame(() => {
      areaRef.current?.focus();
      areaRef.current?.setSelectionRange(text.length, text.length);
    });

  const fill = (text: string) => {
    onChange(text);
    setDismissed(true);
    requestAnimationFrame(() => {
      areaRef.current?.focus();
      if (!selectPlaceholder(text)) areaRef.current?.setSelectionRange(text.length, text.length);
    });
  };

  useImperativeHandle(ref, () => ({ fill }));

  /** Add *text* at the end of the message, on its own when it is a line. */
  const insert = (text: string, line = false) => {
    const current = valueRef.current;
    const separator = !current || /\s$/.test(current) ? "" : line ? "\n" : " ";
    const next = current + separator + text;
    valueRef.current = next;
    onChange(next);
    setDismissed(true);
    focusEnd(next);
  };

  async function addStructure(file: File, session: string) {
    const id = ++attachmentId;
    setAttachments((list) => [...list, { id, kind: "structure", name: file.name, path: null, error: null }]);
    try {
      const upload = await uploadFile(session, file);
      setAttachments((list) => list.map((item) => (item.id === id ? { ...item, name: upload.name, path: upload.path } : item)));
      const panel = uploadedStructure(upload);
      addUserStructure(session, panel);
      sendToViewer({ path: panel.path, source: "upload" });
    } catch (reason) {
      const error = reason instanceof Error ? reason.message : String(reason);
      setAttachments((list) => list.map((item) => (item.id === id ? { ...item, error } : item)));
    }
  }

  async function addMolecules(file: File) {
    try {
      const { smiles: found, failed } = await moleculesFromFile(file);
      if (!found.length) {
        setProblem(`No molecules found in ${file.name}.`);
        return;
      }
      if (found.length <= INLINE_MOLECULES) insert(found.join(" "));
      else setAttachments((list) => [...list, { id: ++attachmentId, kind: "molecules", name: file.name, smiles: found }]);
      if (failed) setProblem(`${failed} record${failed === 1 ? "" : "s"} of ${file.name} could not be read.`);
    } catch (reason) {
      setProblem(reason instanceof Error ? reason.message : String(reason));
    }
  }

  function takeFiles(files: File[]) {
    setProblem(null);
    const session = sessionId ?? "new";
    for (const file of files) {
      const kind = fileKind(file.name);
      if (kind === "structure") void addStructure(file, session);
      else if (kind) void addMolecules(file);
      else setProblem(`${file.name}: only .smi, .csv, .sdf, .mol, .pdb and .cif files can be added.`);
    }
  }

  function takeDrop(data: DataTransfer) {
    const files = Array.from(data.files);
    if (files.length) takeFiles(files);
    else {
      const text = data.getData("text/plain").trim();
      if (text) insert(text);
    }
  }

  useEffect(() => {
    dropRef.current = takeDrop;
  });

  // The whole chat column takes drops, not only the box.
  useEffect(() => {
    const zone = rootRef.current?.parentElement;
    if (!zone) return;
    const carries = (event: DragEvent) => {
      const types = Array.from(event.dataTransfer?.types ?? []);
      return types.includes("Files") || types.includes("text/plain");
    };
    const over = (event: DragEvent) => {
      if (!carries(event)) return;
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
      setDropArea((area) => area ?? zone.getBoundingClientRect());
    };
    const leave = (event: DragEvent) => {
      if (!(event.relatedTarget instanceof Node) || !zone.contains(event.relatedTarget)) setDropArea(null);
    };
    const drop = (event: DragEvent) => {
      setDropArea(null);
      if (!carries(event) || !event.dataTransfer) return;
      event.preventDefault();
      dropRef.current(event.dataTransfer);
    };
    zone.addEventListener("dragenter", over);
    zone.addEventListener("dragover", over);
    zone.addEventListener("dragleave", leave);
    zone.addEventListener("drop", drop);
    return () => {
      zone.removeEventListener("dragenter", over);
      zone.removeEventListener("dragover", over);
      zone.removeEventListener("dragleave", leave);
      zone.removeEventListener("drop", drop);
    };
  }, []);

  function chooseEffort(next: Effort) {
    setChosenEffort(next);
    setModelMenu(false);
    try {
      localStorage.setItem(EFFORT_KEY, next);
    } catch {
      // Private mode: the choice lasts for this page.
    }
  }

  function applySketch(drawn: string) {
    const target = sketch?.replace;
    setSketch(null);
    void loadRDKit()
      .then((module) => canonical(module, drawn), () => null)
      .then((found) => {
        const result = found || drawn;
        if (target && valueRef.current.includes(target)) {
          const next = replaceToken(valueRef.current, target, result);
          onChange(next);
          focusEnd(next);
        } else insert(result);
      });
  }

  function choose(command: SlashCommand) {
    fill(command.template);
  }

  function submit() {
    const extra = attachments.flatMap((item) =>
      item.kind === "structure"
        ? item.path
          ? [`Structure: ${item.path}`]
          : []
        : [`Molecules from ${item.name} (${item.smiles.length}):`, ...item.smiles],
    );
    const content = [value.trim(), ...extra].filter(Boolean).join("\n");
    if (!content) return;
    const local = content.toLowerCase().startsWith(VIEW_PREFIX);
    // While the other model loads the gateway has no endpoint to answer on.
    if (!local && (busy || !connected || loading)) return;
    if (uploading) {
      setProblem("Wait for the upload to finish.");
      return;
    }
    if (firstPlaceholder(content) && !window.confirm("The message still has a {PLACEHOLDER}. Send it anyway?")) return;
    if (
      invalid.length &&
      !window.confirm(`${invalid.length === 1 ? "One SMILES is" : `${invalid.length} SMILES are`} not valid (${invalid.join(", ")}). Send anyway?`)
    )
      return;
    if (onSend(content, effort)) {
      onChange("");
      setAttachments([]);
      setProblem(null);
    }
  }

  const placeholder = !connected
    ? "Waiting for the agent…"
    : busy
      ? "The agent is working. You can write your next message."
      : "Ask the agent, or type / for commands";
  const hasChips = smiles.length > 0 || attachments.length > 0 || problem;

  return (
    <div ref={rootRef} className="border-subtle bg-page border-t" style={{ paddingBottom: "env(safe-area-inset-bottom)" }}>
      <div className="relative mx-auto max-w-3xl px-3 py-2 sm:px-4 sm:py-3">
        {matches.length > 0 && (
          <ul
            role="listbox"
            aria-label="Commands"
            className="bg-raised border-subtle shadow-card absolute right-3 bottom-full left-3 mb-1 max-h-72 overflow-y-auto rounded-xl border py-1 sm:right-4 sm:left-4"
          >
            {matches.map((command, index) => (
              <li key={command.cmd} role="option" aria-selected={index === highlight}>
                <button
                  type="button"
                  onMouseDown={(event) => {
                    event.preventDefault();
                    choose(command);
                  }}
                  onMouseEnter={() => setHighlight(index)}
                  className={`flex w-full cursor-pointer items-baseline gap-3 px-3 py-1.5 text-left text-xs ${index === highlight ? "bg-sunken" : ""}`}
                >
                  <span className="w-24 shrink-0 font-mono font-semibold" style={{ color: "var(--color-accent)" }}>
                    {command.cmd}
                  </span>
                  <span className="text-muted">{command.desc}</span>
                </button>
              </li>
            ))}
          </ul>
        )}

        <div className="bg-raised border-subtle flex flex-col rounded-2xl border focus-within:border-strong">
          {hasChips && (
            <div className="flex max-h-36 flex-wrap items-center gap-1.5 overflow-y-auto px-2.5 pt-2.5">
              {smiles.map((entry) => (
                <SmilesChip key={entry} smiles={entry} valid={validity.get(entry) ?? null} theme={theme} />
              ))}
              {attachments.map((item) => (
                <FileChip
                  key={item.id}
                  name={item.name}
                  detail={
                    item.kind === "molecules"
                      ? `${item.smiles.length} molecules`
                      : item.error
                        ? `upload failed: ${item.error}`
                        : item.path
                          ? "in the viewer"
                          : "uploading…"
                  }
                  failed={item.kind === "structure" && item.error !== null}
                  onRemove={() => setAttachments((list) => list.filter((entry) => entry.id !== item.id))}
                />
              ))}
              {problem && (
                <span className="text-[11px]" style={{ color: "var(--color-warn)" }}>
                  {problem}
                </span>
              )}
            </div>
          )}

          <textarea
            ref={areaRef}
            value={value}
            rows={1}
            disabled={!connected}
            onChange={(event) => {
              onChange(event.target.value);
              setDismissed(false);
              setHighlight(0);
            }}
            onKeyDown={(event) => {
              if (matches.length > 0) {
                if (event.key === "ArrowDown" || event.key === "ArrowUp") {
                  event.preventDefault();
                  const step = event.key === "ArrowDown" ? 1 : -1;
                  setHighlight((current) => (current + step + matches.length) % matches.length);
                  return;
                }
                if (event.key === "Enter" || event.key === "Tab") {
                  event.preventDefault();
                  choose(matches[Math.min(highlight, matches.length - 1)]);
                  return;
                }
                if (event.key === "Escape") {
                  setDismissed(true);
                  return;
                }
              }
              if (event.key === "Tab" && !event.shiftKey) {
                const area = event.currentTarget;
                if (selectPlaceholder(value, area.selectionEnd) || selectPlaceholder(value)) event.preventDefault();
                return;
              }
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                submit();
              }
            }}
            placeholder={placeholder}
            aria-label="Message the agent"
            className="max-h-[220px] min-h-[24px] w-full resize-none bg-transparent px-3.5 pt-2.5 pb-1 text-base outline-none disabled:cursor-not-allowed sm:text-[13px]"
          />

          <div className="flex items-center gap-0.5 px-1.5 pb-1.5">
            <ToolButton label="Attach a file: SMILES, CSV, SDF, MOL, PDB or CIF" onClick={() => fileRef.current?.click()}>
              <path d="m20.5 11.5-8.3 8.3a5 5 0 0 1-7.1-7.1l8.6-8.6a3.4 3.4 0 0 1 4.8 4.8l-8.6 8.6a1.7 1.7 0 0 1-2.4-2.4l7.9-7.9" />
            </ToolButton>
            <ToolButton label="Draw a molecule" onClick={() => openSketcher()}>
              <path d="M12 3.5 19.4 7.75v8.5L12 20.5l-7.4-4.25v-8.5Z" />
              <path d="M12 7.2 16.2 9.6M7.8 14.4 12 16.8" />
            </ToolButton>
            <ToolButton label="New therapeutic target profile" onClick={() => setTtpOpen(true)}>
              <circle cx="12" cy="12" r="8" />
              <circle cx="12" cy="12" r="4" />
              <circle cx="12" cy="12" r="0.6" fill="currentColor" />
            </ToolButton>
            {onAuto && <AutoButton key={autoUntil ?? "off"} until={autoUntil ?? null} onSet={onAuto} />}
            <input
              ref={fileRef}
              type="file"
              multiple
              accept={ACCEPTED_FILES}
              hidden
              onChange={(event) => {
                takeFiles(Array.from(event.target.files ?? []));
                event.target.value = "";
              }}
            />

            <div className="relative ml-1">
              <button
                type="button"
                onClick={() => setModelMenu((open) => !open)}
                aria-haspopup="menu"
                aria-expanded={modelMenu}
                title={loading ? `${loading} is loading` : "Model and thinking level"}
                className="text-muted hover:text-body border-subtle flex h-7 cursor-pointer items-center gap-1.5 rounded-full border px-2.5 text-[11px] touch:h-9"
              >
                <span className="max-w-[14rem] truncate">{loading ? `Loading ${loading}…` : (current?.label ?? "Model")}</span>
                <svg viewBox="0 0 24 24" className="size-3" fill="none" stroke="currentColor" strokeWidth="2.2" aria-hidden="true">
                  <path d="m6 15 6-6 6 6" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </button>
              {modelMenu && (
                <div
                  role="menu"
                  aria-label="Model and thinking level"
                  className="bg-raised border-subtle shadow-card absolute bottom-full left-0 z-20 mb-1 w-60 overflow-hidden rounded-lg border py-1 text-xs"
                  onMouseLeave={() => setModelMenu(false)}
                >
                  {rows.map((row) => (
                    <Fragment key={row.key}>
                      <button
                        type="button"
                        role="menuitemradio"
                        aria-checked={row.key === currentKey}
                        disabled={!canSwitch}
                        onClick={() => models.choose(row.key)}
                        className="hover:bg-sunken flex w-full cursor-pointer items-baseline gap-2 px-3 py-1.5 text-left disabled:cursor-default disabled:hover:bg-transparent"
                      >
                        <span className="w-3 shrink-0" style={{ color: "var(--color-accent)" }}>
                          {row.key === currentKey ? "✓" : ""}
                        </span>
                        <span className="truncate font-medium">{row.label}</span>
                      </button>
                      {row.key === currentKey &&
                        row.efforts.map((value) => (
                          <button
                            key={value}
                            type="button"
                            role="menuitemradio"
                            aria-checked={value === effort}
                            onClick={() => chooseEffort(value)}
                            className="hover:bg-sunken flex w-full cursor-pointer items-baseline gap-2 py-1.5 pr-3 pl-8 text-left"
                          >
                            <span className="w-3 shrink-0" style={{ color: "var(--color-accent)" }}>
                              {value === effort ? "✓" : ""}
                            </span>
                            <span>{effortLabel(value)}</span>
                            <span className="text-faint truncate">{effortHint(value)}</span>
                          </button>
                        ))}
                    </Fragment>
                  ))}
                  {served && !served.switchable && <p className="text-faint px-3 pt-1 pb-0.5 leading-snug">The model is fixed for this session.</p>}
                  {models.note && (
                    <p className="px-3 pt-1 pb-0.5 leading-snug" style={{ color: "var(--color-warn)" }}>
                      {models.note}
                    </p>
                  )}
                </div>
              )}
            </div>

            <span className="flex-1" />
            {busy ? (
              <button
                type="button"
                onClick={onStop}
                aria-label="Stop the agent"
                title="Stop the agent"
                className="border-strong text-body hover:bg-sunken flex h-8 shrink-0 cursor-pointer items-center gap-1.5 rounded-xl border px-3 text-xs font-semibold touch:h-11"
              >
                <span aria-hidden="true" className="size-2.5 rounded-sm" style={{ background: "var(--color-accent)" }} />
                Stop
              </button>
            ) : (
              <button
                type="button"
                onClick={submit}
                disabled={!connected || loading !== null || (!value.trim() && attachments.length === 0)}
                aria-label="Send"
                title={loading ? `${loading} is still loading` : "Send"}
                className="bg-accent hover:bg-accent-strong disabled:bg-inset disabled:text-faint grid size-8 shrink-0 cursor-pointer place-items-center rounded-xl text-white transition-colors disabled:cursor-not-allowed touch:size-11"
              >
                <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth="2.2" aria-hidden="true">
                  <path d="M12 19V5M5 12l7-7 7 7" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </button>
            )}
          </div>
        </div>
      </div>

      {dropArea && (
        <div
          aria-hidden="true"
          className="pointer-events-none fixed z-40 grid place-items-center rounded-xl border-2 border-dashed p-6 text-center text-sm"
          style={{
            top: dropArea.top + 8,
            left: dropArea.left + 8,
            width: dropArea.width - 16,
            height: dropArea.height - 16,
            borderColor: "var(--color-accent)",
            background: "color-mix(in srgb, var(--surface-page) 88%, transparent)",
          }}
        >
          <span>
            <span className="block font-semibold">Drop to add to the message</span>
            <span className="text-muted mt-1 block text-xs">SMILES, molecule files (.smi .csv .sdf .mol) or structures (.pdb .cif)</span>
          </span>
        </div>
      )}

      {sketch && (
        <Suspense fallback={null}>
          <Sketcher smiles={sketch.smiles} theme={theme} onUse={applySketch} onClose={() => setSketch(null)} />
        </Suspense>
      )}

      {ttpOpen &&
        createPortal(
          <Suspense fallback={null}>
            <TtpBuilder
              molecules={molecules}
              onClose={() => setTtpOpen(false)}
              onCreated={(_, request) => {
                setTtpOpen(false);
                insert(request, true);
              }}
            />
          </Suspense>,
          document.body,
        )}
    </div>
  );
}));

function ToolButton({ label, onClick, children }: { label: string; onClick: () => void; children: ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      title={label}
      className="text-muted hover:text-body hover:bg-sunken grid size-7 shrink-0 cursor-pointer place-items-center rounded-lg touch:size-9"
    >
      <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        {children}
      </svg>
    </button>
  );
}

function SmilesChip({ smiles, valid, theme }: { smiles: string; valid: boolean | null; theme: Theme }) {
  const bad = valid === false;
  return (
    <button
      type="button"
      onClick={() => openSketcher(bad ? undefined : smiles, smiles)}
      title={bad ? `RDKit cannot read ${smiles}` : `${smiles}: edit in the sketcher`}
      className="bg-sunken flex h-10 max-w-[15rem] cursor-pointer items-center gap-1.5 rounded-lg border pr-2 pl-1 text-left hover:border-strong"
      style={bad ? { borderColor: "var(--color-accent)" } : undefined}
    >
      {bad ? (
        <span className="grid h-8 w-8 shrink-0 place-items-center text-sm font-bold" style={{ color: "var(--color-accent)" }} aria-hidden="true">
          !
        </span>
      ) : (
        <MolDepiction smiles={smiles} theme={theme} width={54} height={36} className="h-8 w-12 shrink-0" />
      )}
      <span className="min-w-0">
        <code className="text-muted block truncate font-mono text-[10.5px] leading-tight">{smiles}</code>
        {bad && (
          <span className="block text-[10px] leading-tight" style={{ color: "var(--color-accent)" }}>
            not a valid SMILES
          </span>
        )}
      </span>
    </button>
  );
}

function FileChip({ name, detail, failed, onRemove }: { name: string; detail: string; failed: boolean; onRemove: () => void }) {
  return (
    <span
      className="bg-sunken flex h-10 max-w-[18rem] items-center gap-2 rounded-lg border pr-1 pl-2.5 text-[11px]"
      style={failed ? { borderColor: "var(--color-accent)" } : undefined}
    >
      <svg viewBox="0 0 24 24" className="text-faint size-3.5 shrink-0" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
        <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8Z M14 3v5h5" strokeLinejoin="round" />
      </svg>
      <span className="min-w-0 leading-tight">
        <span className="block truncate font-medium">{name}</span>
        <span className="text-faint block truncate text-[10px]" style={failed ? { color: "var(--color-accent)" } : undefined} title={detail}>
          {detail}
        </span>
      </span>
      <button
        type="button"
        onClick={onRemove}
        aria-label={`Remove ${name}`}
        className="text-faint hover:text-body grid size-6 shrink-0 cursor-pointer place-items-center rounded"
      >
        ×
      </button>
    </span>
  );
}
