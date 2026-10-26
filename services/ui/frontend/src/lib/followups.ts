// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { Panel } from "../types/api";
import { gate } from "./gate";

/**
 * Up to two natural next questions for an answer, from what its steps
 * produced: a pose invites a contact profile, a failed gate a "why", a
 * contact profile a question about its strongest residue.
 */
export function suggestFollowups(panels: Panel[], tools: string[]): string[] {
  const out: string[] = [];
  const add = (text: string) => {
    if (!out.includes(text)) out.push(text);
  };

  const structures = panels.filter((panel) => panel.kind === "structure");
  const profiles = panels.filter((panel) => panel.kind === "interactions");
  const properties = panels.filter((panel) => panel.kind === "properties");
  const molecules = panels.filter((panel) => panel.kind === "molecule");

  if (structures.some((panel) => gate(panel.metrics).status === "fail")) add("Why did this pose fail the confidence gate?");
  if (structures.length > 0 && profiles.length === 0) add("Profile the interactions in this pose");

  for (const profile of profiles) {
    const byResidue = new Map<string, number>();
    for (const contact of profile.contacts) {
      if (contact.resnum === null) continue;
      byResidue.set(contact.label, (byResidue.get(contact.label) ?? 0) + 1);
    }
    const top = [...byResidue.entries()].sort((a, b) => b[1] - a[1])[0]?.[0];
    if (top) add(`Which ligand fragment makes the ${top} contact?`);
    if (!profile.counts.hbond) add("Suggest a change that adds a hydrogen bond in this pocket");
  }

  if (properties.length > 0) {
    const many = properties.some((panel) => panel.molecules.length > 1);
    add(many ? "Which of these molecules has the best ADMET profile?" : "Are any of these predictions outside the model's domain?");
  }

  if (tools.some((name) => /^(start|launch|run)_/.test(name) && /reinvent|run|generation/.test(name))) add("How is the run going?");
  if (molecules.length > 1 && structures.length === 0) add("Cofold the top molecule with Boltz-2");

  return out.slice(0, 2);
}
