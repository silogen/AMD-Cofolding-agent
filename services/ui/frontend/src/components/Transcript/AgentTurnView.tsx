// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useState, type AnchorHTMLAttributes, type HTMLAttributes } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { AgentTurn, Panel, StructurePanel, TurnBlock } from "../../types/api";
import type { Theme } from "../../hooks/useTheme";
import { toolSteps } from "../../lib/transcript";
import { clock, duration, summarise } from "../../lib/format";
import { isResultPanel, panelKey } from "../../lib/artifacts";
import { remarkResidueLinks, residueFromHref } from "../../lib/residueLinks";
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

/** Re-render every second while *running*, for live timers. */
function useNow(running: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [running]);
  return now;
}

/**
 * One answer from the agent.
 *
 * A turn can run for an hour of tool calls, so its shape is: what the agent
 * is doing now, the steps it took (each with its duration and output), what
 * those steps produced, and finally its answer.
 */
export function AgentTurnView({ turn, theme, openStructurePath, onOpenMolecule, onShowProperties, linkResidues }: Props) {
  const running = !turn.done;
  const now = useNow(running);
  const steps = toolSteps(turn);
  const elapsed = (turn.endedAt ?? now) - turn.at;
  const panels = steps.flatMap((step) =>
    (step.result?.panels ?? []).filter((panel) => isResultPanel(step.name, panel)),
  );
  const lastThinking = [...turn.blocks].reverse().find((block) => block.kind === "thinking");
  const failedSteps = steps.filter((step) => step.result?.isError).length;
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
  const thoughts = turn.blocks.flatMap((block) => (block.kind === "thinking" ? [block] : []));

  return (
    <article className="flex flex-col gap-3" aria-busy={running}>
      {titled && (
        <h3 className="text-body truncate text-sm leading-snug font-semibold" title={title}>
          {title}
        </h3>
      )}

      {running && <LiveStatus steps={steps} thinking={lastThinking?.kind === "thinking" ? lastThinking.content : null} now={now} />}

      {running && steps.length > 0 && <StepLog steps={steps} running={running} now={now} />}

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

      {!running && (steps.length > 0 || thoughts.length > 0) && (
        <TurnTrace
          steps={steps}
          thoughts={thoughts.map((block) => block.content)}
          failed={failedSteps}
          meta={{
            at: clock(turn.at),
            took: duration(elapsed),
            tokens: turn.tokens ? `${turn.tokens.input.toLocaleString()} in · ${turn.tokens.output.toLocaleString()} out` : null,
          }}
          now={now}
        />
      )}
    </article>
  );
}

function uniqueStructures(panels: Panel[]): StructurePanel[] {
  const seen = new Set<string>();
  return panels.filter((panel): panel is StructurePanel => {
    if (panel.kind !== "structure" || seen.has(panel.path)) return false;
    seen.add(panel.path);
    return true;
  });
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
  thoughts,
  failed,
  meta,
  now,
}: {
  steps: ToolStep[];
  thoughts: string[];
  failed: number;
  meta: TurnMeta;
  now: number;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className="text-[11px]">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="text-faint hover:text-muted flex max-w-full cursor-pointer items-center gap-x-1.5 text-left touch:min-h-11"
      >
        <span aria-hidden="true" className={`inline-block transition-transform ${open ? "rotate-90" : ""}`}>
          ›
        </span>
        <span className="font-medium">{steps.length > 0 ? `${steps.length} step${steps.length === 1 ? "" : "s"}` : "Details"}</span>
        {failed > 0 && <span style={{ color: "var(--color-accent)" }}>· {failed} failed</span>}
      </button>
      {open && (
        <div className="mt-2 flex flex-col gap-2">
          <TurnMetaList meta={meta} />
          {steps.length > 0 && <StepList steps={steps} now={now} />}
          {thoughts.map((content, index) => (
            <p key={index} className="text-muted border-subtle border-l-2 py-0.5 pl-3 text-xs whitespace-pre-wrap">
              {content}
            </p>
          ))}
        </div>
      )}
    </div>
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

function StepList({ steps, now }: { steps: ToolStep[]; now: number }) {
  return (
    <ol className="bg-sunken border-subtle overflow-hidden rounded-lg border">
      {steps.map((step, index) => (
        <StepRow key={step.id} step={step} index={index + 1} now={now} />
      ))}
    </ol>
  );
}

/** The tool calls of the turn, one line each, expandable to the full output. */
function StepLog({ steps, running, now }: { steps: ToolStep[]; running: boolean; now: number }) {
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
            <StepRow key={step.id} step={step} index={index + 1} now={now} />
          ))}
        </ol>
      )}
    </div>
  );
}

function StepRow({ step, index, now }: { step: ToolStep; index: number; now: number }) {
  const [open, setOpen] = useState(false);
  const result = step.result;
  const failed = result?.isError ?? false;
  const took = result ? (result.elapsedMs ?? Date.parse(result.ts) - step.at) : now - step.at;
  const args = summarise(step.arguments);

  return (
    <li className="border-subtle border-b last:border-b-0">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="hover:bg-inset flex w-full cursor-pointer items-center gap-2 px-3 py-1.5 text-left text-xs"
      >
        <span className="text-faint w-5 shrink-0 text-right font-mono text-[10px]">{index}</span>
        <StepIcon state={!result ? "running" : failed ? "failed" : "ok"} />
        <span className="shrink-0 font-mono font-semibold" style={{ color: failed ? "var(--color-accent)" : undefined }}>
          {step.name}
        </span>
        <span className="text-faint min-w-0 flex-1 truncate font-mono text-[11px]" title={args}>
          {args}
        </span>
        <span className={`shrink-0 font-mono text-[10px] ${result ? "text-faint" : "text-muted"}`}>
          {Number.isFinite(took) ? duration(Math.max(0, took)) : ""}
        </span>
      </button>
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
}

function StepIcon({ state }: { state: "running" | "ok" | "failed" }) {
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

/** What the agent is doing right now. */
function LiveStatus({ steps, thinking, now }: { steps: ToolStep[]; thinking: string | null; now: number }) {
  const runningStep = steps.find((step) => !step.result);
  const label = runningStep ? `Running ${runningStep.name} · ${duration(now - runningStep.at)}` : "Thinking";
  const preview = thinking?.trim().split("\n").filter(Boolean).slice(-2).join(" ");

  return (
    <div className="flex flex-col gap-1" role="status">
      <p className={`text-muted flex items-center gap-2 text-xs ${runningStep ? "" : "italic"}`}>
        <span
          aria-hidden="true"
          className={`${runningStep ? "size-1.5" : "size-[7px]"} animate-pulse rounded-full`}
          style={{ background: runningStep ? "var(--color-molecule)" : "var(--color-structure)" }}
        />
        {label}…
      </p>
      {preview && !runningStep && (
        <p className="text-faint border-subtle line-clamp-2 border-l-2 pl-3 text-[11px] italic">{preview}</p>
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
  const seen = new Set<string>();
  const unique = panels.filter((panel) => {
    const key = panelKey(panel);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  return (
    <PanelGrid
      panels={unique}
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
  return (
    <button
      type="button"
      className="residue-link"
      title={`Show ${residue.resnum}${residue.chain ? `.${residue.chain}` : ""} in the 3D viewer`}
      onClick={() => sendToViewer({ action: { kind: "residues", residues: [residue], mode: "focus" }, source: "answer" })}
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
const LINKED_PLUGINS = [remarkGfm, remarkCallouts, remarkResidueLinks];
const COMPONENTS = { a: AnswerLink, blockquote: AnswerQuote };

function AnswerBlock({ content, linkResidues }: { content: string; linkResidues: boolean }) {
  return (
    <div className="group relative">
      <div className="markdown">
        <Markdown remarkPlugins={linkResidues ? LINKED_PLUGINS : PLAIN_PLUGINS} components={COMPONENTS}>
          {content}
        </Markdown>
      </div>
      <div className="mt-1 opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100 touch:opacity-100">
        <CopyButton value={content} label="Copy answer" />
      </div>
    </div>
  );
}

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