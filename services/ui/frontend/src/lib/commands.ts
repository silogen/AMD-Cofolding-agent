// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * Prompt templates for the composer's slash commands. Each one maps to tools
 * the agent has today; {PLACEHOLDERS} are selected after insertion so the
 * user types over them.
 */
export interface SlashCommand {
  cmd: string;
  desc: string;
  template: string;
  /** Tools the template relies on; the command is hidden if one is missing. */
  tools: string[];
}

/** Runs in the browser on the 3D viewer; never sent to the agent. */
export const VIEW_PREFIX = "/view";

export const SLASH_COMMANDS: SlashCommand[] = [
  {
    cmd: VIEW_PREFIX,
    desc: "Drive the 3D viewer: pocket, focus ASN140, surface, show hbonds…",
    template: `${VIEW_PREFIX} {COMMAND}`,
    tools: [],
  },
  {
    cmd: "/design",
    desc: "De novo REINVENT campaign against a target",
    template: "Design ligands for {TARGET}: define a TTP with Boltz affinity and QED, run REINVENT de novo, and show the top candidates.",
    tools: ["create_ttp", "run_optimization"],
  },
  {
    cmd: "/mol2mol",
    desc: "Analogues of a known active",
    template: "Generate analogues of {SMILES} with Mol2Mol and score them against {TARGET} with Boltz-2.",
    tools: ["run_optimization"],
  },
  {
    cmd: "/cofold",
    desc: "Score ligands with Boltz-2 and PoseBusters",
    template: "Cofold {SMILES} into {TARGET} with Boltz-2 and check the pose with PoseBusters.",
    tools: ["score_molecules"],
  },
  {
    cmd: "/finetune",
    desc: "Transfer learning on a set of actives",
    template: "Fine-tune the REINVENT prior on these actives with transfer learning: {SMILES_LIST}",
    tools: ["run_transfer_learning"],
  },
  {
    cmd: "/admet",
    desc: "Predict ADMET endpoints",
    template: "Predict ADMET properties for {SMILES}.",
    tools: ["predict_admet"],
  },
  {
    cmd: "/dmta",
    desc: "RetroDMTA property predictions",
    template: "Run RetroDMTA predictions for {SMILES}.",
    tools: ["predict_retrodmta"],
  },
  {
    cmd: "/interactions",
    desc: "Profile protein-ligand contacts",
    template: "Profile the interactions in the latest cofolded complex.",
    tools: ["profile_interactions"],
  },
  {
    cmd: "/ttp",
    desc: "Show or edit the therapeutic target profile",
    template: "Show me the current TTP and {CHANGE}.",
    tools: ["get_ttp", "update_ttp"],
  },
  {
    cmd: "/target",
    desc: "Look up a target in UniProt",
    template: "What is {TARGET}? Give me its UniProt entry, sequence length and known structures.",
    tools: ["fetch_target_info"],
  },
  {
    cmd: "/compound",
    desc: "Look up a compound in PubChem",
    template: "Look up {NAME} in PubChem and give me its SMILES and properties.",
    tools: ["fetch_compound"],
  },
  {
    cmd: "/jobs",
    desc: "Running jobs and how they converge",
    template: "What jobs are running and how are they converging?",
    tools: ["list_jobs", "get_optimization_status"],
  },
  {
    cmd: "/top",
    desc: "Best candidates of a job",
    template: "Show the top 10 candidates of the latest job with their scores.",
    tools: ["get_top_candidates"],
  },
];

/** Commands whose tools the agent has. With no tool list yet, all are shown. */
export function availableCommands(tools: string[]): SlashCommand[] {
  if (tools.length === 0) return SLASH_COMMANDS;
  const have = new Set(tools);
  return SLASH_COMMANDS.filter((command) => command.tools.every((tool) => have.has(tool)));
}

/** The first {PLACEHOLDER} in *text*, as [start, end), or null. */
export function firstPlaceholder(text: string): [number, number] | null {
  const match = /\{[A-Z_]+\}/.exec(text);
  return match ? [match.index, match.index + match[0].length] : null;
}
