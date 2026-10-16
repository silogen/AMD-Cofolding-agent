// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useRef, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { StructurePanel, Turn } from "../../types/api";
import type { Theme } from "../../hooks/useTheme";
import { ToolBlock } from "../ToolBlock/ToolBlock";

interface Props {
  turns: Turn[];
  busy: boolean;
  theme: Theme;
  onOpenStructure: (panel: StructurePanel) => void;
  openStructurePath: string | null;
}

/**
 * The conversation.
 *
 * Everything the agent did stays in place next to the question that caused
 * it, which is what lets a user answer "what was computed, when, and why".
 * A separate generated report cannot do that.
 */
export function Transcript({ turns, busy, theme, onOpenStructure, openStructurePath }: Props) {
  const endRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef(true);

  // Follow new output, but stop following as soon as the user scrolls up to
  // read something: yanking them back to the bottom loses their place.
  useEffect(() => {
    const container = scrollRef.current;
    if (!container) return;
    const onScroll = () => {
      const distance = container.scrollHeight - container.scrollTop - container.clientHeight;
      pinnedRef.current = distance < 120;
    };
    container.addEventListener("scroll", onScroll, { passive: true });
    return () => container.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    if (pinnedRef.current) {
      endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [turns, busy]);

  return (
    <div ref={scrollRef} className="flex-1 overflow-y-auto">
      {/* Extra bottom padding on small screens keeps the last message clear of
          the floating "View structure" button. */}
      <div className="mx-auto flex max-w-3xl flex-col gap-5 px-3 py-4 pb-20 sm:gap-6 sm:px-4 sm:py-6 lg:pb-6">
        {turns.length === 0 && <EmptyState />}

        {turns.map((turn) =>
          turn.role === "user" ? (
            <UserBubble key={turn.id} content={turn.content} />
          ) : (
            <div key={turn.id} className="flex flex-col gap-3">
              {turn.blocks.map((block) => {
                switch (block.kind) {
                  case "thinking":
                    return <ThinkingBlock key={block.id} content={block.content} />;
                  case "tool":
                    return (
                      <ToolBlock
                        key={block.id}
                        name={block.name}
                        args={block.arguments}
                        result={block.result}
                        theme={theme}
                        onOpenStructure={onOpenStructure}
                        openStructurePath={openStructurePath}
                      />
                    );
                  case "text":
                    return (
                      <div key={block.id} className="markdown text-[0.9375rem]">
                        <Markdown remarkPlugins={[remarkGfm]}>{block.content}</Markdown>
                      </div>
                    );
                  case "error":
                    return <ErrorBlock key={block.id} message={block.message} />;
                }
              })}

              {turn.tokens && (
                <p className="text-faint text-[11px]">
                  {turn.tokens.input.toLocaleString()} in · {turn.tokens.output.toLocaleString()} out
                </p>
              )}
            </div>
          ),
        )}

        {busy && <Working />}
        <div ref={endRef} />
      </div>
    </div>
  );
}

/**
 * A failed turn.
 *
 * Backend errors arrive as formatted, multi-line text: the request, the
 * endpoint, the response, and a list of things to check. Rendering that as
 * ordinary prose collapses every newline and turns a readable report into one
 * run-on paragraph, so the whitespace is preserved.
 *
 * The first line alone says what went wrong, and the rest is for whoever is
 * debugging the deployment, so only the first line is shown until asked.
 */
function ErrorBlock({ message }: { message: string }) {
  const [open, setOpen] = useState(false);
  const lines = message.split("\n");
  const summary = lines[0];
  const hasDetail = lines.length > 1;

  return (
    <div
      role="alert"
      className="border-subtle overflow-hidden rounded-lg border text-xs"
      style={{ background: "var(--color-accent-soft)" }}
    >
      <div className="flex items-start gap-2 px-3 py-2">
        <span aria-hidden="true" className="font-semibold" style={{ color: "var(--color-accent)" }}>
          !
        </span>
        <p className="min-w-0 flex-1 font-medium">{summary}</p>
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

function UserBubble({ content }: { content: string }) {
  return (
    <div className="flex justify-end">
      <p className="bg-inset max-w-[85%] rounded-2xl rounded-br-sm px-4 py-2.5 text-[0.9375rem] whitespace-pre-wrap">
        {content}
      </p>
    </div>
  );
}

/**
 * The agent's reasoning, collapsed by default.
 *
 * It is useful when a result is surprising and noise the rest of the time,
 * so it is present but never in the way.
 */
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
      {open && (
        <p className="text-muted border-subtle mt-1.5 border-l-2 py-0.5 pl-3 whitespace-pre-wrap">
          {content}
        </p>
      )}
    </div>
  );
}

function Working() {
  return (
    <p className="text-muted flex items-center gap-2 text-xs" role="status">
      <span className="flex gap-1" aria-hidden="true">
        {[0, 150, 300].map((delay) => (
          <span
            key={delay}
            className="size-1.5 animate-bounce rounded-full"
            style={{ background: "var(--color-accent)", animationDelay: `${delay}ms` }}
          />
        ))}
      </span>
      Working…
    </p>
  );
}

function EmptyState() {
  return (
    <div className="mt-8 text-center">
      <h2 className="text-lg font-semibold">Design a molecule</h2>
      <p className="text-muted mx-auto mt-2 max-w-md text-sm">
        Ask for a design campaign in plain English. The agent scores candidates with
        Boltz, REINVENT and the ADMET models, and everything it computes appears here,
        next to the question that caused it.
      </p>
    </div>
  );
}
