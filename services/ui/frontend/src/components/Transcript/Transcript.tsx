// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useRef } from "react";
import type { Turn } from "../../types/api";
import type { Theme } from "../../hooks/useTheme";
import { clock } from "../../lib/format";
import { AgentTurnView } from "./AgentTurnView";

interface Props {
  turns: Turn[];
  theme: Theme;
  openStructurePath: string | null;
  onOpenMolecule: (smiles: string) => void;
  onShowProperties: () => void;
  onPickPrompt: (text: string) => void;
  /** Whether the conversation has a structure to show residues on. */
  hasStructure: boolean;
}

/**
 * The conversation.
 *
 * Everything the agent did stays in place next to the question that caused
 * it, which is what lets a user answer "what was computed, when, and why".
 */
export function Transcript({ turns, theme, openStructurePath, onOpenMolecule, onShowProperties, onPickPrompt, hasStructure }: Props) {
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
    if (pinnedRef.current) endRef.current?.scrollIntoView({ block: "end" });
  }, [turns]);

  return (
    <div ref={scrollRef} className="flex-1 overflow-y-auto">
      <div className="mx-auto flex max-w-3xl flex-col gap-5 px-3 py-4 pb-20 sm:gap-6 sm:px-4 sm:py-6 lg:pb-6">
        {turns.length === 0 && <EmptyState onPick={onPickPrompt} />}

        {turns.map((turn, index) =>
          turn.role === "user" ? (
            <UserBubble key={turn.id} content={turn.content} at={turn.at} />
          ) : (
            <AgentTurnView
              key={turn.id}
              turn={turn}
              theme={theme}
              openStructurePath={openStructurePath}
              onOpenMolecule={onOpenMolecule}
              onShowProperties={onShowProperties}
              linkResidues={hasStructure}
              onFollowup={index === turns.length - 1 ? onPickPrompt : undefined}
            />
          ),
        )}
        <div ref={endRef} />
      </div>
    </div>
  );
}

function UserBubble({ content, at }: { content: string; at: number }) {
  return (
    <div className="flex flex-col items-end gap-1">
      <p className="bg-inset max-w-[85%] rounded-2xl rounded-br-sm px-4 py-2.5 text-[0.9375rem] whitespace-pre-wrap">
        {content}
      </p>
      <span className="text-faint text-[10px]">{clock(at)}</span>
    </div>
  );
}

/** Starting points that exercise the agent's real tools. */
const STARTERS: { title: string; body: string; prompt: string }[] = [
  {
    title: "Design against a target",
    body: "Define a TTP, run REINVENT and score with Boltz-2 affinity.",
    prompt:
      "Design ligands for BRD4 BD1: define a TTP with Boltz affinity and QED, run a short REINVENT de novo campaign, and show me the top candidates.",
  },
  {
    title: "Analogues of a known active",
    body: "Mol2Mol around a seed, scored on the same objectives.",
    prompt: "Generate analogues of JQ1 with Mol2Mol and score them against BRD4 with Boltz-2.",
  },
  {
    title: "Cofold and check a ligand",
    body: "Boltz-2 pose, confidence, PoseBusters and contacts.",
    prompt: "Cofold JQ1 into BRD4 BD1 with Boltz-2, check the pose with PoseBusters and profile its interactions.",
  },
  {
    title: "Check ADMET",
    body: "Solubility, permeability, efflux and lipophilicity for a set of molecules.",
    prompt: "Predict ADMET properties for the top 10 candidates of the latest job.",
  },
];

function EmptyState({ onPick }: { onPick: (text: string) => void }) {
  return (
    <div className="mt-6 sm:mt-10">
      <div className="text-center">
        <h2 className="text-lg font-semibold">Design a molecule</h2>
        <p className="text-muted mx-auto mt-2 max-w-md text-sm">
          Ask in plain English. The agent scores candidates with Boltz-2, REINVENT and the ADMET models, and
          everything it computes appears here and in the workbench.
        </p>
      </div>
      <div className="mt-6 grid gap-2 sm:grid-cols-2">
        {STARTERS.map((starter) => (
          <button
            key={starter.title}
            type="button"
            onClick={() => onPick(starter.prompt)}
            className="bg-raised border-subtle shadow-card hover:border-strong cursor-pointer rounded-xl border p-3 text-left"
          >
            <span className="block text-xs font-semibold">{starter.title}</span>
            <span className="text-muted mt-1 block text-[11px]">{starter.body}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
