// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { ConversationEvent, SessionSummary } from "../types/api";
import { applyEvents, EMPTY_TRANSCRIPT } from "./transcript";

/**
 * A conversation as Markdown: questions, tool calls with their arguments,
 * and answers, in order. Meant for a lab notebook or a report, so the raw
 * tool output is left out; the JSON export has everything.
 */
export function sessionToMarkdown(session: SessionSummary, events: ConversationEvent[]): string {
  const { turns } = applyEvents(EMPTY_TRANSCRIPT, events);
  const lines = [`# ${session.title || "Untitled conversation"}`, "", `_${session.id} · started ${session.created}_`, ""];
  for (const turn of turns) {
    if (turn.role === "user") {
      lines.push("## Question", "", turn.content, "");
      continue;
    }
    for (const block of turn.blocks) {
      switch (block.kind) {
        case "tool": {
          const status = !block.result ? "did not finish" : block.result.isError ? "failed" : "ok";
          const took = block.result?.elapsedMs != null ? `, ${(block.result.elapsedMs / 1000).toFixed(1)} s` : "";
          lines.push(`- \`${block.name}\` (${status}${took})`, "", "  ```json", indent(JSON.stringify(block.arguments, null, 2)), "  ```", "");
          break;
        }
        case "text":
          lines.push("## Answer", "", block.content, "");
          break;
        case "error":
          lines.push(`> **Error:** ${block.message.split("\n")[0]}`, "");
          break;
        case "cancelled":
          lines.push("> Stopped by the user.", "");
          break;
        case "thinking":
          break;
      }
    }
  }
  return lines.join("\n");
}

function indent(text: string): string {
  return text
    .split("\n")
    .map((line) => `  ${line}`)
    .join("\n");
}

/** A file name from a title: lower case, dashes, no punctuation. */
export function slug(title: string, fallback: string): string {
  const cleaned = title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 60);
  return cleaned || fallback;
}
