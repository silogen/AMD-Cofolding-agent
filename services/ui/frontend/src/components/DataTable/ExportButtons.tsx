// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useState } from "react";
import { download } from "../../lib/api";
import { fileName, toCsv, toTsv, type Sheet } from "../../lib/csv";

interface Props {
  /** Built on click, so large tables cost nothing until exported. */
  sheet: () => Sheet;
  /** File name without extension. */
  name: string;
  /** Said in the tooltips, e.g. "the first 200 rows". */
  scope?: string;
}

/** "CSV" downloads the table, "Copy" puts it on the clipboard as tab-separated cells. */
export function ExportButtons({ sheet, name, scope }: Props) {
  const [copied, setCopied] = useState<"ok" | "failed" | null>(null);

  const copy = async () => {
    const { columns, rows } = sheet();
    try {
      await navigator.clipboard.writeText(toTsv(columns, rows));
      setCopied("ok");
    } catch {
      setCopied("failed");
    }
    window.setTimeout(() => setCopied(null), 1500);
  };

  return (
    <span className="flex shrink-0 items-center gap-0.5">
      <TextButton
        title={`Download ${scope ?? "the table"} as CSV`}
        onClick={() => {
          const { columns, rows } = sheet();
          download(fileName(name, "csv"), toCsv(columns, rows), "text/csv");
        }}
      >
        <DownloadIcon /> CSV
      </TextButton>
      <TextButton title={`Copy ${scope ?? "the table"} (pastes into a spreadsheet)`} onClick={() => void copy()}>
        {copied === "ok" ? "Copied" : copied === "failed" ? "Copy failed" : "Copy"}
      </TextButton>
    </span>
  );
}

export function TextButton({ children, onClick, title, disabled }: { children: React.ReactNode; onClick: () => void; title?: string; disabled?: boolean }) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      disabled={disabled}
      className="text-faint hover:text-body hover:bg-inset inline-flex cursor-pointer items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium disabled:cursor-default disabled:opacity-40 touch:min-h-11"
    >
      {children}
    </button>
  );
}

export function DownloadIcon() {
  return (
    <svg width={10} height={10} viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth={1.5} aria-hidden="true">
      <path d="M6 1.5v6M3.5 5.2 6 7.7l2.5-2.5M2 10.5h8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
