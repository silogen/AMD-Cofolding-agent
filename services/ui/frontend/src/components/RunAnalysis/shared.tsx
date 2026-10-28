// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useRef, useState, type ReactNode } from "react";

export function Card({ title, actions, children, className = "" }: { title: string; actions?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`bg-raised border-subtle min-w-0 rounded-xl border px-3 py-2.5 ${className}`}>
      <div className="mb-1.5 flex min-h-6 flex-wrap items-center gap-x-2 gap-y-1">
        <h3 className="text-faint text-[10px] font-semibold tracking-wider uppercase">{title}</h3>
        {actions && <div className="ml-auto flex flex-wrap items-center gap-1">{actions}</div>}
      </div>
      {children}
    </section>
  );
}

export function Chip({ label, colour, on, title, onClick, onHover }: { label: string; colour: string; on: boolean; title?: string; onClick: () => void; onHover?: (on: boolean) => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      onPointerEnter={() => onHover?.(true)}
      onPointerLeave={() => onHover?.(false)}
      aria-pressed={on}
      title={title ?? label}
      className={`inline-flex max-w-40 cursor-pointer items-center gap-1 rounded-full px-1.5 py-px text-[11px] transition-colors ${on ? "text-body" : "text-faint hover:text-muted"}`}
    >
      <span className="inline-block size-2 shrink-0 rounded-full transition-opacity" style={{ background: colour, opacity: on ? 1 : 0.25 }} />
      <span className="truncate">{label}</span>
    </button>
  );
}

/** A small menu of checkboxes, e.g. which endpoints to show. */
export function PickMenu({ label, items, checked, onToggle }: { label: ReactNode; items: { key: string; label: string; colour?: string; group?: string }[]; checked: ReadonlySet<string>; onToggle: (key: string) => void }) {
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (event: PointerEvent) => {
      if (!box.current?.contains(event.target as Node)) setOpen(false);
    };
    const esc = (event: KeyboardEvent) => event.key === "Escape" && setOpen(false);
    document.addEventListener("pointerdown", close);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("pointerdown", close);
      document.removeEventListener("keydown", esc);
    };
  }, [open]);
  return (
    <span ref={box} className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="text-muted hover:text-body hover:bg-inset inline-flex cursor-pointer items-center gap-0.5 rounded px-1.5 py-0.5 text-[11px]"
      >
        {label}
        <svg width={8} height={8} viewBox="0 0 10 10" aria-hidden="true">
          <path d="M2 3.5 5 6.5 8 3.5" fill="none" stroke="currentColor" strokeWidth={1.4} />
        </svg>
      </button>
      {open && (
        <span className="bg-raised border-subtle shadow-card absolute top-full right-0 z-30 mt-1 flex max-h-72 w-max max-w-64 flex-col overflow-y-auto rounded-lg border p-1 text-[11px]">
          {items.map((item, i) => {
            const heading = item.group && item.group !== items[i - 1]?.group ? item.group : null;
            return (
              <span key={item.key} className="contents">
                {heading && <span className="text-faint px-1.5 pt-1.5 pb-0.5 text-[10px]">{heading}</span>}
                <label className="hover:bg-inset flex cursor-pointer items-center gap-1.5 rounded px-1.5 py-0.5" title={item.key}>
                  <input type="checkbox" checked={checked.has(item.key)} onChange={() => onToggle(item.key)} className="size-3 accent-[var(--color-accent)]" />
                  {item.colour && <span className="inline-block size-2 shrink-0 rounded-full" style={{ background: item.colour }} />}
                  <span className="truncate">{item.label}</span>
                </label>
              </span>
            );
          })}
        </span>
      )}
    </span>
  );
}

export function Placeholder({ children, height = 160 }: { children: ReactNode; height?: number }) {
  return (
    <div className="text-faint grid place-items-center text-center text-[11px]" style={{ height }}>
      {children}
    </div>
  );
}
