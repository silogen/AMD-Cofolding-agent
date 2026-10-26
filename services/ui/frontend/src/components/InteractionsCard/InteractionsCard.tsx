// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { InteractionsPanel } from "../../types/api";
import { contactColor, contactLabel, sortedTypes } from "../../lib/contacts";
import { sendToViewer } from "../../lib/viewerBus";

/**
 * The protein-ligand contacts of one complex: counts by type and the
 * residues involved. Every type and residue shows itself on the structure.
 */
export function InteractionsCard({ panel }: { panel: InteractionsPanel }) {
  const path = panel.structurePath;
  const types = sortedTypes(panel.counts);
  const residues = new Map<string, { label: string; chain: string | null; resnum: number; types: Set<string> }>();
  for (const contact of panel.contacts) {
    if (contact.resnum === null) continue;
    const key = `${contact.chain}:${contact.resnum}`;
    const entry = residues.get(key) ?? { label: contact.label, chain: contact.chain, resnum: contact.resnum, types: new Set<string>() };
    entry.types.add(contact.type);
    residues.set(key, entry);
  }
  // Specific interactions first: a residue with an H-bond matters more than one with only vdW contacts.
  const rank = (set: Set<string>) => Math.min(...[...set].map((type) => types.indexOf(type)));
  const top = [...residues.values()].sort((a, b) => rank(a.types) - rank(b.types)).slice(0, 12);

  return (
    <section className="bg-raised border-subtle shadow-card overflow-hidden rounded-xl border">
      <header className="border-subtle flex items-center gap-2 border-b px-3 py-2">
        <span aria-hidden="true" className="size-2 shrink-0 rounded-full" style={{ background: "#00c2de" }} />
        <span className="min-w-0 truncate text-xs font-semibold">Contacts{panel.name ? ` · ${panel.name}` : ""}</span>
        <span className="text-faint ml-auto shrink-0 text-[11px]">
          {panel.totalContacts} contact{panel.totalContacts === 1 ? "" : "s"} · {residues.size} residues
        </span>
      </header>
      {panel.error ? (
        <p className="px-3 py-2 text-xs" style={{ color: "var(--color-accent)" }}>
          {panel.error}
        </p>
      ) : (
        <div className="space-y-2 px-3 py-2">
          <div className="flex flex-wrap gap-1">
            {types.map((type) => (
              <button
                key={type}
                type="button"
                onClick={() => sendToViewer({ path, action: { kind: "contacts", type }, source: "chat" })}
                title={`Show the ${contactLabel(type).toLowerCase()} contacts on the structure`}
                className="border-subtle hover:border-strong flex cursor-pointer items-center gap-1 rounded-full border px-2 py-0.5 text-[11px]"
              >
                <span className="size-2 rounded-full" style={{ background: contactColor(type) }} />
                {contactLabel(type)}
                <b className="font-semibold">{panel.counts[type]}</b>
              </button>
            ))}
          </div>
          {top.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {top.map((residue) => (
                <button
                  key={`${residue.chain}:${residue.resnum}`}
                  type="button"
                  onClick={() =>
                    sendToViewer({
                      path,
                      action: { kind: "residues", residues: [{ chain: residue.chain, resnum: residue.resnum }], mode: "focus" },
                      source: "chat",
                    })
                  }
                  title={[...residue.types].map(contactLabel).join(", ")}
                  className="bg-sunken hover:bg-inset flex cursor-pointer items-center gap-1 rounded px-1.5 py-0.5 font-mono text-[11px]"
                >
                  {[...residue.types].slice(0, 3).map((type) => (
                    <span key={type} className="size-1.5 rounded-full" style={{ background: contactColor(type) }} />
                  ))}
                  {residue.label}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
      <footer className="border-subtle flex gap-1 border-t px-3 py-1.5">
        <button
          type="button"
          onClick={() => sendToViewer({ path, action: { kind: "contacts", type: null }, source: "chat" })}
          className="cursor-pointer rounded-md px-2 py-1 text-[11px] font-semibold text-white hover:opacity-90"
          style={{ background: "var(--color-accent)" }}
        >
          Show on structure
        </button>
        <button
          type="button"
          onClick={() => sendToViewer({ path, action: { kind: "ligand" }, source: "chat" })}
          className="text-muted hover:text-body hover:bg-inset cursor-pointer rounded-md px-2 py-1 text-[11px] font-medium"
        >
          Ligand view
        </button>
      </footer>
    </section>
  );
}
