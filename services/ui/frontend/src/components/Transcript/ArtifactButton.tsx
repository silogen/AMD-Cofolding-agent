// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/** A link-coloured "Open … ↗" that takes the user to what an answer produced. */
export function ArtifactButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="cursor-pointer text-[12px] font-semibold hover:underline"
      style={{ color: "var(--color-structure)" }}
    >
      {label} <span aria-hidden="true">↗</span>
    </button>
  );
}
