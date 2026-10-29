// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { memo, useEffect, useLayoutEffect, useMemo, useRef, useState, type AnchorHTMLAttributes, type HTMLAttributes } from "react";
import Markdown, { type Options } from "react-markdown";
import remarkGfm from "remark-gfm";
import type { AgentTurn, Panel, StructurePanel, TurnBlock } from "../../types/api";
import type { Theme } from "../../hooks/useTheme";
import { stepError, stepState, toolSteps, turnTrace, type StepState, type TraceEntry } from "../../lib/transcript";
import { clock, duration, summarise } from "../../lib/format";
import { isResultPanel, mergeTurnPanels } from "../../lib/artifacts";
import { remarkResidueLinks, residueFromHref, structureFromHref, turnStructures } from "../../lib/residueLinks";
import { sendToViewer } from "../../lib/viewerBus";
import { openWorkbench } from "../../lib/workbenchBus";
import { answerTitle, artifactLinks, remarkCallouts, withoutLeadingHeading, type ArtifactLink, type CalloutTone } from "../../lib/answer";
import { gate } from "../../lib/gate";
import { PanelGrid } from "../PanelGrid/PanelGrid";
import { CopyButton } from "../CopyButton/CopyButton";
import { Callout } from "../Callout/Callout";
import { AlertCard } from "../AlertCard/AlertCard";
import { ArtifactButton } from "./ArtifactButton";

type ToolStep = Extract<TurnBlock, { kind: "tool" }>;

/** When a turn ran, how long it took and what it cost. */
interface TurnMeta {
  at: string;
  took: string;
  tokens: string | null;
}

interface Props {
  turn: AgentTurn;
  theme: Theme;
  openStructurePath: string | null;
  onOpenMolecule: (smiles: string) => void;
  onShowProperties: () => void;
  /** Turn residue names in answers into viewer links; only useful once there is a structure. */
  linkResidues: boolean;
}

/** The time since *since*, counting up every second; only this text re-renders. */
function Elapsed({ since }: { since: number }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  return <>{duration(Math.max(0, now - since))}</>;
}

/**
 * One answer from the agent.
 *
 * A turn can run for an hour of tool calls, so its shape is: what the agent
 * is doing now, the steps it took (each with its duration and output), what
 * those steps produced, and finally its answer. Memoised: a finished turn
 * keeps its object, so only the running one re-renders on each event.
 */
export const AgentTurnView = memo(function AgentTurnView({ turn, theme, openStructurePath, onOpenMolecule, onShowProperties, linkResidues }: Props) {
  const running = !turn.done;
  const steps = toolSteps(turn);
  const panels = steps.flatMap((step) =>
    (step.result?.panels ?? []).filter((panel) => isResultPanel(step.name, panel)),
  );
  const lastThinking = [...turn.blocks].reverse().find((block) => block.kind === "thinking");
  const failedSteps = steps.filter((step) => stepState(step) === "failed").length;
  const trace = turnTrace(turn);
  // One state for the live dropdown and the finished fold, so an open trace stays open when the turn ends.
  const [traceOpen, setTraceOpen] = useState(false);
  const toggleTrace = () => setTraceOpen((value) => !value);
  const tools = steps.map((step) => step.name);
  const texts = turn.blocks.flatMap((block) => (block.kind === "text" ? [block.content] : []));
  const onlyAlerts = turn.blocks.length > 0 && turn.blocks.every((block) => block.kind === "alert");
  const stopped = texts.length === 0 && turn.blocks.some((block) => block.kind === "cancelled");
  const title = onlyAlerts ? "Monitor alert" : stopped ? "Stopped" : answerTitle(texts, tools);
  // Nothing has come back yet, so there is nothing to title: LiveStatus is the only progress line.
  const titled = !(running && texts.length === 0 && steps.length === 0);
  const firstText = turn.blocks.find((block) => block.kind === "text");
  const links = turn.done ? artifactLinks(panels, tools) : [];
  const failedGate = uniqueStructures(panels).filter((panel) => gate(panel.metrics).status === "fail");
  const thoughts = turn.blocks.some((block) => block.kind === "thinking");
  // One string, so the memoised answers keep their props while the turn runs.
  const structures = linkResidues ? turnStructures(steps, panels).join("\n") : "";

  return (
    <article className="flex flex-col gap-3" aria-busy={running}>
      {(titled || turn.auto) && (
        <div className="flex min-w-0 items-center gap-2">
          {turn.auto && <AutoTag />}
          {titled && (
            <h3 className="text-body truncate text-sm leading-snug font-semibold" title={title}>
              {title}
            </h3>
          )}
        </div>
      )}

      {running && (
        <LiveStatus
          steps={steps}
          thinking={lastThinking?.kind === "thinking" ? lastThinking.content : null}
          failed={failedSteps}
          trace={trace}
          open={traceOpen}
          onToggle={toggleTrace}
        />
      )}

      {running && !traceOpen && steps.length > 0 && <StepLog steps={steps} running={running} />}

      {panels.length > 0 && (
        <TurnResults
          panels={panels}
          theme={theme}
          openStructurePath={openStructurePath}
          onOpenMolecule={onOpenMolecule}
          onShowProperties={onShowProperties}
        />
      )}

      {failedGate.length > 0 && <GateCallout structures={failedGate} />}

      {turn.blocks.map((block) => {
        switch (block.kind) {
          case "text":
            return (
              <AnswerBlock
                key={block.id}
                content={block === firstText ? withoutLeadingHeading(block.content, title) : block.content}
                linkResidues={linkResidues}
                structures={structures}
              />
            );
          case "alert":
            return <AlertCard key={block.id} alert={block.alert} />;
          case "error":
            return <ErrorBlock key={block.id} message={block.message} />;
          case "cancelled":
            return (
              <p key={block.id} className="text-faint text-xs">
                Jobs it already launched keep running; stop them from the Jobs tab.
              </p>
            );
          default:
            return null;
        }
      })}

      {links.length > 0 && <ArtifactLinks links={links} />}

      {!running && (steps.length > 0 || thoughts) && (
        <TurnTrace
          steps={steps.length}
          trace={trace}
          failed={failedSteps}
          open={traceOpen}
          onToggle={toggleTrace}
          meta={{
            at: clock(turn.at),
            took: turn.endedAt !== undefined ? duration(turn.endedAt - turn.at) : "",
            tokens: turn.tokens ? `${turn.tokens.input.toLocaleString()} in · ${turn.tokens.output.toLocaleString()} out` : null,
          }}
        />
      )}
    </article>
  );
});

/** A turn the agent took on its own, under its autonomy budget. */
function AutoTag() {
  return (
    <span
      className="text-faint shrink-0 rounded-full border px-1.5 text-[10px] leading-[1.5] font-medium"
      style={{ borderColor: "var(--color-accent)" }}
      title="The agent started this turn on its own"
    >
      auto
    </span>
  );
}

function uniqueStructures(panels: Panel[]): StructurePanel[] {
  return mergeTurnPanels(panels).filter((panel): panel is StructurePanel => panel.kind === "structure");
}

/** Poses whose affinity should not be quoted. */
function GateCallout({ structures }: { structures: StructurePanel[] }) {
  const one = structures.length === 1 ? structures[0] : null;
  return (
    <Callout tone="warn" title={one ? `${one.label || one.name} fails the confidence gate` : `${structures.length} poses fail the confidence gate`}>
      <p>
        {one
          ? `${gate(one.metrics).reasons.join("; ")}. Read the affinity with caution.`
          : "Their affinities are not reliable; check the pose quality before comparing them."}
      </p>
    </Callout>
  );
}

function ArtifactLinks({ links }: { links: ArtifactLink[] }) {
  const open = (link: ArtifactLink) => {
    if (link.target.kind === "viewer") sendToViewer(link.target.request);
    else openWorkbench(link.target.tab);
  };
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
      {links.map((link) => (
        <ArtifactButton key={link.label} label={link.label} onClick={() => open(link)} />
      ))}
    </div>
  );
}

/** A finished turn's steps, timing and reasoning, folded into one line under the answer. */
function TurnTrace({
  steps,
  trace,
  failed,
  meta,
  open,
  onToggle,
}: {
  steps: number;
  trace: TraceEntry[];
  failed: number;
  meta: TurnMeta;
  open: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="text-[11px]">
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={open}
        className="text-faint hover:text-muted flex max-w-full cursor-pointer items-center gap-x-1.5 text-left touch:min-h-11"
      >
        <span aria-hidden="true" className={`inline-block transition-transform ${open ? "rotate-90" : ""}`}>
          ›
        </span>
        <span className="font-medium">{steps > 0 ? `${steps} step${steps === 1 ? "" : "s"}` : "Details"}</span>
        {failed > 0 && <span style={{ color: "var(--color-accent)" }}>· {failed} failed</span>}
      </button>
      {open && (
        <div className="mt-2 flex flex-col gap-2">
          <TurnMetaList meta={meta} />
          <TraceList entries={trace} follow={false} />
        </div>
      )}
    </div>
  );
}

type TraceGroup = Exclude<TraceEntry, { kind: "tool" }> | { kind: "steps"; id: string; steps: { step: ToolStep; index: number }[] };

/** Consecutive tool calls share one box; reasoning and errors sit between them, in order. */
function groupTrace(entries: TraceEntry[]): TraceGroup[] {
  const groups: TraceGroup[] = [];
  let index = 0;
  for (const entry of entries) {
    if (entry.kind !== "tool") {
      groups.push(entry);
      continue;
    }
    index += 1;
    const last = groups[groups.length - 1];
    if (last?.kind === "steps") last.steps.push({ step: entry, index });
    else groups.push({ kind: "steps", id: `s-${entry.id}`, steps: [{ step: entry, index }] });
  }
  return groups;
}

/** How close to the bottom, in pixels, still counts as reading the end. */
const STICK_PX = 16;

/**
 * The reasoning and tool calls of a turn, in order, in a box of its own
 * height. *follow*: keep the newest entry in view while the reader is at
 * the bottom; scrolling up to read stops that until they come back down.
 */
export function TraceList({ entries, follow }: { entries: TraceEntry[]; follow: boolean }) {
  const box = useRef<HTMLDivElement>(null);
  const atBottom = useRef(true);

  useLayoutEffect(() => {
    const element = box.current;
    if (element && follow && atBottom.current) element.scrollTop = element.scrollHeight;
  }, [entries, follow]);

  if (entries.length === 0) return <p className="text-faint text-[11px] italic">Nothing yet.</p>;

  return (
    <div
      ref={box}
      onScroll={(event) => {
        const element = event.currentTarget;
        atBottom.current = element.scrollHeight - element.scrollTop - element.clientHeight <= STICK_PX;
      }}
      className="flex max-h-72 flex-col gap-2 overflow-y-auto overscroll-contain pr-1"
    >
      {groupTrace(entries).map((group) => {
        switch (group.kind) {
          case "thinking":
            return <TraceThought key={group.id} content={group.content} />;
          case "error":
            return <TraceError key={group.id} message={group.message} />;
          case "steps":
            return (
              <ol key={group.id} className="bg-sunken border-subtle shrink-0 overflow-hidden rounded-lg border">
                {group.steps.map(({ step, index }) => (
                  <StepRow key={step.id} step={step} index={index} />
                ))}
              </ol>
            );
        }
      })}
    </div>
  );
}

const TraceThought = memo(function TraceThought({ content }: { content: string }) {
  return <p className="text-muted border-subtle shrink-0 border-l-2 py-0.5 pl-3 text-[11px] whitespace-pre-wrap">{content.trim()}</p>;
});

function TraceError({ message }: { message: string }) {
  const line = message.split("\n").find((part) => part.trim())?.trim() ?? message;
  return (
    <p className="flex shrink-0 items-start gap-1.5 pl-3 text-[11px]" style={{ color: "var(--color-accent)" }}>
      <span aria-hidden="true" className="font-semibold">
        !
      </span>
      <span className="min-w-0">{line}</span>
    </p>
  );
}

function TurnMetaList({ meta }: { meta: TurnMeta }) {
  return (
    <div className="text-faint flex flex-col gap-0.5">
      <p>{meta.at}</p>
      <p>
        <span className="text-muted">Duration</span> {meta.took}
      </p>
      {meta.tokens && (
        <p>
          <span className="text-muted">Token</span> {meta.tokens}
        </p>
      )}
    </div>
  );
}

/** The tool calls of the turn, one line each, expandable to the full output. */
function StepLog({ steps, running }: { steps: ToolStep[]; running: boolean }) {
  // Open while the agent works, so progress is visible; a finished turn
  // with many steps folds away behind its answer.
  const [open, setOpen] = useState<boolean | null>(null);
  const shown = open ?? (running || steps.length <= 3);

  return (
    <div className="bg-sunken border-subtle overflow-hidden rounded-lg border">
      <button
        type="button"
        onClick={() => setOpen(!shown)}
        aria-expanded={shown}
        className="text-muted hover:text-body flex w-full cursor-pointer items-center gap-2 px-3 py-1.5 text-left text-[11px] font-medium"
      >
        <span aria-hidden="true" className={`inline-block transition-transform ${shown ? "rotate-90" : ""}`}>
          ›
        </span>
        Steps
        <span className="text-faint ml-auto font-normal">
          {steps.map((step) => step.name).filter((name, index, all) => all.indexOf(name) === index).slice(0, 4).join(" · ")}
        </span>
      </button>
      {shown && (
        <ol className="border-subtle border-t">
          {steps.map((step, index) => (
            <StepRow key={step.id} step={step} index={index + 1} />
          ))}
        </ol>
      )}
    </div>
  );
}

const StepRow = memo(function StepRow({ step, index }: { step: ToolStep; index: number }) {
  const [open, setOpen] = useState(false);
  const result = step.result;
  const state = stepState(step);
  const failed = state === "failed";
  const error = failed ? stepError(step) : null;
  const took = result ? (result.elapsedMs ?? Date.parse(result.ts) - step.at) : null;
  const args = summarise(step.arguments);

  return (
    <li
      className="border-subtle border-b last:border-b-0"
      style={failed ? { boxShadow: "inset 2px 0 0 var(--color-accent)" } : undefined}
    >
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="hover:bg-inset flex w-full cursor-pointer items-center gap-2 px-3 py-1.5 text-left text-xs"
      >
        <span className="text-faint w-5 shrink-0 text-right font-mono text-[10px]">{index}</span>
        <StepIcon state={state} />
        <span className="shrink-0 font-mono font-semibold" style={{ color: failed ? "var(--color-accent)" : undefined }}>
          {step.name}
        </span>
        <span className="text-faint min-w-0 flex-1 truncate font-mono text-[11px]" title={args}>
          {args}
        </span>
        <span className={`shrink-0 font-mono text-[10px] ${result ? "text-faint" : "text-muted"}`}>
          {took === null ? <Elapsed since={step.at} /> : Number.isFinite(took) ? duration(Math.max(0, took)) : ""}
        </span>
      </button>
      {error && !open && (
        <p className="-mt-1 truncate pr-3 pb-1.5 pl-10 text-[11px]" style={{ color: "var(--color-accent)" }} title={error}>
          {error}
        </p>
      )}
      {open && (
        <div className="space-y-2 px-3 pb-3">
          <details className="text-[11px]">
            <summary className="text-muted cursor-pointer">Arguments</summary>
            <pre className="bg-raised border-subtle mt-1 max-h-60 overflow-auto rounded-md border p-2 font-mono text-[11px]">
              {JSON.stringify(step.arguments, null, 2)}
            </pre>
          </details>
          {result ? (
            <div className="relative">
              <div className="absolute top-1 right-1">
                <CopyButton value={result.result} label="Copy output" />
              </div>
              <pre className="bg-raised border-subtle max-h-80 overflow-auto rounded-md border p-2 pr-10 font-mono text-[11px] leading-relaxed whitespace-pre-wrap">
                {result.result}
                {result.truncated && <span className="text-faint">{"\n\n… output cut for display."}</span>}
              </pre>
            </div>
          ) : (
            <p className="text-muted text-[11px]">Running since {clock(step.at)}…</p>
          )}
        </div>
      )}
    </li>
  );
});

function StepIcon({ state }: { state: StepState }) {
  if (state === "running") {
    return (
      <span
        aria-label="Running"
        className="size-3 shrink-0 animate-spin rounded-full border-2 border-t-transparent"
        style={{ borderColor: "var(--color-accent)", borderTopColor: "transparent" }}
      />
    );
  }
  return (
    <span
      aria-label={state === "ok" ? "Done" : "Failed"}
      className="grid size-3 shrink-0 place-items-center rounded-full text-[8px] font-bold text-white"
      style={{ background: state === "ok" ? "var(--color-ok)" : "var(--color-accent)" }}
    >
      {state === "ok" ? "✓" : "✕"}
    </span>
  );
}

/**
 * What the agent is doing right now. The line opens onto the whole trace so
 * far, which follows along as events arrive.
 */
function LiveStatus({
  steps,
  thinking,
  failed,
  trace,
  open,
  onToggle,
}: {
  steps: ToolStep[];
  thinking: string | null;
  failed: number;
  trace: TraceEntry[];
  open: boolean;
  onToggle: () => void;
}) {
  const runningStep = steps.find((step) => !step.result);
  const label = runningStep ? (
    <>
      Running {runningStep.name} · <Elapsed since={runningStep.at} />
    </>
  ) : (
    "Thinking"
  );
  const preview = thinking?.trim().split("\n").filter(Boolean).slice(-2).join(" ");

  return (
    <div className="flex flex-col gap-1.5">
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={open}
        title={open ? "Hide the reasoning and steps" : "Show the reasoning and steps so far"}
        className="text-muted hover:text-body flex max-w-full cursor-pointer items-center gap-2 text-left text-xs touch:min-h-11"
      >
        <span
          aria-hidden="true"
          className={`${runningStep ? "size-1.5" : "size-[7px]"} shrink-0 animate-pulse rounded-full`}
          style={{ background: runningStep ? "var(--color-molecule)" : "var(--color-structure)" }}
        />
        <span role="status" className={`min-w-0 truncate ${runningStep ? "" : "italic"}`}>
          {label}…
        </span>
        {failed > 0 && (
          <span className="shrink-0 text-[11px]" style={{ color: "var(--color-accent)" }}>
            · {failed} failed
          </span>
        )}
        <span aria-hidden="true" className={`text-faint inline-block shrink-0 transition-transform ${open ? "rotate-90" : ""}`}>
          ›
        </span>
      </button>
      {open ? (
        <TraceList entries={trace} follow />
      ) : (
        preview &&
        !runningStep && <p className="text-faint border-subtle line-clamp-2 border-l-2 pl-3 text-[11px] italic">{preview}</p>
      )}
    </div>
  );
}

/** Everything the steps produced, deduplicated, newest structure first. */
function TurnResults({
  panels,
  theme,
  openStructurePath,
  onOpenMolecule,
  onShowProperties,
}: {
  panels: Panel[];
  theme: Theme;
  openStructurePath: string | null;
  onOpenMolecule: (smiles: string) => void;
  onShowProperties: () => void;
}) {
  return (
    <PanelGrid
      panels={panels}
      theme={theme}
      openStructurePath={openStructurePath}
      onOpenMolecule={onOpenMolecule}
      onShowProperties={onShowProperties}
    />
  );
}

/** Links in answers; a residue link shows the residue in the 3D viewer instead of navigating. */
function AnswerLink({ href, title, children }: AnchorHTMLAttributes<HTMLAnchorElement>) {
  const residue = residueFromHref(href);
  if (!residue) {
    return (
      <a href={href} title={title} target="_blank" rel="noreferrer">
        {children}
      </a>
    );
  }
  const { path, ref, near } = structureFromHref(href);
  return (
    <button
      type="button"
      className="residue-link"
      title="Show in the 3D viewer"
      onClick={() => sendToViewer({ path, ref, near, action: { kind: "residues", residues: [residue], mode: "focus" }, source: "answer" })}
    >
      {children}
    </button>
  );
}

/** Blockquotes, which remarkCallouts has typed, as callouts. */
function AnswerQuote({ children, ...props }: HTMLAttributes<HTMLQuoteElement> & { "data-tone"?: CalloutTone; "data-title"?: string }) {
  return (
    <Callout tone={props["data-tone"] ?? "info"} title={props["data-title"]}>
      {children}
    </Callout>
  );
}

const PLAIN_PLUGINS = [remarkGfm, remarkCallouts];
const COMPONENTS = { a: AnswerLink, blockquote: AnswerQuote };

/**
 * Memoised: the running turn re-renders on every event, and parsing Markdown is the costly part.
 * *structures*: the turn's structure files, one per line, for the residue links.
 */
const AnswerBlock = memo(function AnswerBlock({ content, linkResidues, structures }: { content: string; linkResidues: boolean; structures: string }) {
  const plugins = useMemo<Options["remarkPlugins"]>(
    () => (linkResidues ? [remarkGfm, remarkCallouts, [remarkResidueLinks, { structures: structures ? structures.split("\n") : [] }]] : PLAIN_PLUGINS),
    [linkResidues, structures],
  );
  return (
    <div className="group relative">
      <div className="markdown">
        <Markdown remarkPlugins={plugins} components={COMPONENTS}>
          {content}
        </Markdown>
      </div>
      <div className="mt-1 opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100 touch:opacity-100">
        <CopyButton value={content} label="Copy answer" />
      </div>
    </div>
  );
});

/**
 * A failure. Backend errors are multi-line reports, so the first line is
 * the summary and the rest stays one click away, with its whitespace.
 */
export function ErrorBlock({ message }: { message: string }) {
  const [open, setOpen] = useState(false);
  const lines = message.split("\n");
  const hasDetail = lines.length > 1;

  return (
    <div role="alert" className="border-subtle overflow-hidden rounded-lg border text-xs" style={{ background: "var(--color-accent-soft)" }}>
      <div className="flex items-start gap-2 px-3 py-2">
        <span aria-hidden="true" className="font-semibold" style={{ color: "var(--color-accent)" }}>
          !
        </span>
        <p className="min-w-0 flex-1 font-medium">{lines[0]}</p>
        {hasDetail && (
          <button
            type="button"
            onClick={() => setOpen((value) => !value)}
            aria-expanded={open}
            className="text-muted hover:text-body shrink-0 cursor-pointer font-medium touch:min-h-11"
          >
            {open ? "Less" : "Details"}
          </button>
        )}
      </div>
      {open && hasDetail && (
        <pre className="border-subtle max-h-72 overflow-auto border-t px-3 py-2 font-mono text-[11px] leading-relaxed whitespace-pre-wrap">
          {lines.slice(1).join("\n").trim()}
        </pre>
      )}
    </div>
  );
}