// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useState, type AnchorHTMLAttributes } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { AgentTurn, Panel, TurnBlock } from "../../types/api";
import type { Theme } from "../../hooks/useTheme";
import { toolSteps } from "../../lib/transcript";
import { clock, duration, summarise } from "../../lib/format";
import { panelKey } from "../../lib/artifacts";
import { suggestFollowups } from "../../lib/followups";
import { remarkResidueLinks, residueFromHref } from "../../lib/residueLinks";
import { sendToViewer } from "../../lib/viewerBus";
import { PanelGrid } from "../PanelGrid/PanelGrid";
import { CopyButton } from "../CopyButton/CopyButton";

type ToolStep = Extract<TurnBlock, { kind: "tool" }>;

interface Props {
  turn: AgentTurn;
  theme: Theme;
  openStructurePath: string | null;
  onOpenMolecule: (smiles: string) => void;
  onShowProperties: () => void;
  /** Turn residue names in answers into viewer links; only useful once there is a structure. */
  linkResidues: boolean;
  /** Set on the latest answer only: offers next questions and puts the chosen one in the composer. */
  onFollowup?: (text: string) => void;
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
export function AgentTurnView({ turn, theme, openStructurePath, onOpenMolecule, onShowProperties, linkResidues, onFollowup }: Props) {
  const running = !turn.done;
  const now = useNow(running);
  const steps = toolSteps(turn);
  const elapsed = (turn.endedAt ?? now) - turn.at;
  const panels = steps.flatMap((step) =>
    (step.result?.panels ?? []).filter((panel) => !(panel.kind === "molecule" && step.name === "profile_interactions")),
  );
  const lastThinking = [...turn.blocks].reverse().find((block) => block.kind === "thinking");
  const failedSteps = steps.filter((step) => step.result?.isError).length;
  const answered = turn.done && turn.blocks.some((block) => block.kind === "text") && !turn.blocks.some((block) => block.kind === "cancelled");
  const followups = onFollowup && answered ? suggestFollowups(panels, steps.map((step) => step.name)) : [];

  return (
    <article className="flex flex-col gap-3" aria-busy={running}>
      <div className="text-faint flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px]">
        <span className="text-body flex items-center gap-1.5 font-semibold">
          <span aria-hidden="true" className="size-2 rounded-sm" style={{ background: "var(--color-accent)" }} />
          Agent
        </span>
        <span>{clock(turn.at)}</span>
        <span title="Elapsed">{duration(elapsed)}</span>
        {steps.length > 0 && (
          <span>
            {steps.length} step{steps.length === 1 ? "" : "s"}
            {failedSteps > 0 && <span style={{ color: "var(--color-accent)" }}> · {failedSteps} failed</span>}
          </span>
        )}
        {turn.tokens && (
          <span title="Tokens used by the conversation so far">
            {turn.tokens.input.toLocaleString()} in · {turn.tokens.output.toLocaleString()} out
          </span>
        )}
      </div>

      {steps.length > 0 && <StepLog steps={steps} running={running} now={now} />}

      {running && <LiveStatus steps={steps} thinking={lastThinking?.kind === "thinking" ? lastThinking.content : null} now={now} />}

      {panels.length > 0 && (
        <TurnResults
          panels={panels}
          theme={theme}
          openStructurePath={openStructurePath}
          onOpenMolecule={onOpenMolecule}
          onShowProperties={onShowProperties}
        />
      )}

      {turn.blocks.map((block) => {
        switch (block.kind) {
          case "text":
            return <AnswerBlock key={block.id} content={block.content} linkResidues={linkResidues} />;
          case "error":
            return <ErrorBlock key={block.id} message={block.message} />;
          case "cancelled":
            return (
              <p key={block.id} className="text-muted border-subtle rounded-lg border px-3 py-2 text-xs">
                Stopped. Jobs the agent already launched keep running; stop them from the Jobs tab.
              </p>
            );
          case "thinking":
            return running ? null : <ThinkingBlock key={block.id} content={block.content} />;
          default:
            return null;
        }
      })}

      {followups.length > 0 && onFollowup && (
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <span className="text-faint text-[10px] font-semibold tracking-wider uppercase">Next</span>
          {followups.map((text) => (
            <button
              key={text}
              type="button"
              onClick={() => onFollowup(text)}
              className="text-muted cursor-pointer text-xs hover:text-[var(--color-structure)] hover:underline"
            >
              {text}
            </button>
          ))}
        </div>
      )}
    </article>
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

const PLAIN_PLUGINS = [remarkGfm];
const LINKED_PLUGINS = [remarkGfm, remarkResidueLinks];
const COMPONENTS = { a: AnswerLink };

function AnswerBlock({ content, linkResidues }: { content: string; linkResidues: boolean }) {
  return (
    <div className="group relative">
      <div className="markdown text-[0.9375rem]">
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

function ThinkingBlock({ content }: { content: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="text-xs">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="text-faint hover:text-muted cursor-pointer font-medium touch:-my-1 touch:min-h-11"
      >
        {open ? "Hide reasoning" : "Show reasoning"}
      </button>
      {open && <p className="text-muted border-subtle mt-1.5 border-l-2 py-0.5 pl-3 whitespace-pre-wrap">{content}</p>}
    </div>
  );
}
