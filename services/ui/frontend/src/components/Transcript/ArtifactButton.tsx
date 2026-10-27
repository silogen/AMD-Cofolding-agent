// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/** A quiet "Open … ↗" that takes the user to what an answer produced. */
export function ArtifactButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="text-muted cursor-pointer text-xs font-medium transition-colors hover:text-[var(--color-accent)] touch:min-h-11"
    >
      {label} <span aria-hidden="true">↗</span>
    </button>
  );
}
