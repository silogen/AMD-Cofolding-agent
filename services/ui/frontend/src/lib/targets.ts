// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * The protein targets one conversation touched. A chat can look up a UniProt
 * entry, cofold against the construct of another and keep a third in a
 * profile, so the Target tab lists them and shows one at a time.
 *
 * Facts are read from the same tool events as extractTargetFacts, only kept
 * apart per protein; a lookup made in the panel joins the list too.
 */

import {
  asTtp,
  emptyTargetFacts,
  findSequence,
  hasTargetFacts,
  isAccession,
  mergeContacts,
  mergeSequence,
  mergeTtpTarget,
  mergeUniprot,
  ttpIn,
  type TargetFacts,
  type ToolEvent,
} from "./ttp";

export interface TargetEntry {
  /** Stable while the conversation grows: the accession, else an entry, sequence or profile. */
  key: string;
  facts: TargetFacts;
  /** From the conversation's own tool events, not from a lookup made in the panel. */
  inSession: boolean;
  /** Position of the last event that touched it, so the latest can lead. */
  seen: number;
}

/** The shortest name that still identifies a target. */
export function targetName(facts: TargetFacts): string {
  return (
    facts.gene ??
    facts.name ??
    facts.uniprot ??
    facts.pdbIds[0] ??
    facts.ttpId ??
    (facts.sequenceLength ? `${facts.sequenceLength} aa` : "Target")
  );
}

/**
 * Whether two fact sets are the same protein: the same accession, a shared
 * PDB entry, a sequence one contains (a construct of the other), or the same
 * profile. Two known but different accessions are never the same target.
 */
export function sameTarget(a: TargetFacts, b: TargetFacts): boolean {
  if (a.uniprot && b.uniprot) return a.uniprot === b.uniprot;
  if (a.pdbIds.some((id) => b.pdbIds.includes(id))) return true;
  if (a.sequence && b.sequence) return a.sequence.includes(b.sequence) || b.sequence.includes(a.sequence);
  return Boolean(a.ttpId && a.ttpId === b.ttpId);
}

export function targetKey(facts: TargetFacts): string {
  if (facts.uniprot) return facts.uniprot;
  if (facts.pdbIds.length) return facts.pdbIds[0];
  if (facts.sequence) return `seq:${facts.sequence.length}:${facts.sequence.slice(0, 8)}`;
  return facts.ttpId ? `ttp:${facts.ttpId}` : "target";
}

const entry = (facts: TargetFacts, seen: number, inSession: boolean): TargetEntry => ({ key: targetKey(facts), facts, inSession, seen });

/** Every protein the conversation's tool calls are about, in the order first seen. */
export function extractTargets(events: ToolEvent[]): TargetEntry[] {
  const found: TargetEntry[] = [];
  let current = -1;

  /** Fold what an event says into the target it is about, and make that one current. */
  const touch = (probe: TargetFacts, merge: (facts: TargetFacts) => TargetFacts, at: number) => {
    const index = found.findIndex((e) => sameTarget(e.facts, probe));
    const facts = merge(index < 0 ? emptyTargetFacts() : found[index].facts);
    const next = entry(facts, at, true);
    if (index < 0) found.push(next);
    else found[index] = next;
    current = index < 0 ? found.length - 1 : index;
  };

  events.forEach((event, at) => {
    if (event.type === "tool_call") {
      const ttp = asTtp(event.arguments.ttp);
      const query = typeof event.arguments.query === "string" ? event.arguments.query.trim().toUpperCase() : "";
      const sequence = ttp ? null : findSequence(event.arguments);
      if (ttp) touch(mergeTtpTarget(emptyTargetFacts(), ttp, event.name), (f) => mergeTtpTarget(f, ttp, event.name), at);
      else if (event.name === "fetch_target_info" && isAccession(query)) touch({ ...emptyTargetFacts(), uniprot: query }, (f) => ({ ...f, uniprot: f.uniprot ?? query }), at);
      else if (sequence) touch(mergeSequence(emptyTargetFacts(), sequence, event.name), (f) => mergeSequence(f, sequence, event.name), at);
      return;
    }

    if (event.isError) return;
    if (event.name === "fetch_target_info") {
      const probe = mergeUniprot(emptyTargetFacts(), event.result, event.name);
      if (hasTargetFacts(probe)) touch(probe, (f) => mergeUniprot(f, event.result, event.name), at);
    } else {
      const ttp = ttpIn(event);
      if (ttp) touch(mergeTtpTarget(emptyTargetFacts(), ttp, event.name), (f) => mergeTtpTarget(f, ttp, event.name), at);
    }
    // Contacts name no protein; they belong to the complex the session was last working on.
    for (const panel of event.panels ?? []) {
      if (panel.kind !== "interactions" || panel.contacts.length === 0) continue;
      if (current < 0) current = found.push(entry(emptyTargetFacts(), at, true)) - 1;
      found[current] = entry(mergeContacts(found[current].facts, panel.contacts, event.name), at, found[current].inSession);
    }
  });

  // A lookup names its accession before the answer arrives; that placeholder
  // goes once the answer landed on a protein the session already knew.
  return found.filter((e) => e.facts.sources.length > 0 || !found.some((other) => other !== e && other.facts.uniprot === e.facts.uniprot));
}

/** The conversation's own targets first, the latest of each group leading. */
export function orderTargets(entries: readonly TargetEntry[]): TargetEntry[] {
  return [...entries].sort((a, b) => Number(b.inSession) - Number(a.inSession) || b.seen - a.seen);
}

/** A target looked up in the panel, merged into the conversation's when it is one of them. */
export function withLookup(entries: readonly TargetEntry[], payload: unknown): TargetEntry[] {
  const probe = mergeUniprot(emptyTargetFacts(), payload, "lookup");
  if (!hasTargetFacts(probe)) return [...entries];
  const at = entries.reduce((max, e) => Math.max(max, e.seen), -1) + 1;
  const index = entries.findIndex((e) => sameTarget(e.facts, probe));
  if (index < 0) return [...entries, entry(probe, at, false)];
  const known = entries[index];
  return entries.map((e, i) => (i === index ? entry(mergeUniprot(known.facts, payload, "lookup"), at, known.inSession) : e));
}

/** The target to show: the one picked, else the one the current run or profile uses, else the latest. */
export function scopedTarget(entries: readonly TargetEntry[], picked: string | null, ttpId: string | null): TargetEntry | null {
  return (
    (picked ? entries.find((e) => e.key === picked) : undefined) ??
    (ttpId ? entries.find((e) => e.facts.ttpId === ttpId) : undefined) ??
    orderTargets(entries)[0] ??
    null
  );
}
