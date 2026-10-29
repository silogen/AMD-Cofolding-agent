// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import type { AutoEvent, SessionSummary, StatusEvent } from "../types/api";
import { applyEvent, EMPTY_TRANSCRIPT } from "../lib/transcript";
import { autoDeadline, autoMessage } from "./useSession";

const UNTIL = "2026-10-29T12:00:00+00:00";

const summary = (auto: string | null): SessionSummary => ({
  id: "s-20261029-080000-abcdef",
  title: "BRD4",
  created: UNTIL,
  updated: UNTIL,
  turns: 1,
  busy: false,
  auto_until: auto,
});

describe("the autonomy budget over the socket", () => {
  it("asks for whole minutes, and 0 to stop", () => {
    expect(autoMessage(30)).toEqual({ type: "auto", minutes: 30 });
    expect(autoMessage(0)).toEqual({ type: "auto", minutes: 0 });
    expect(autoMessage(90.4)).toEqual({ type: "auto", minutes: 90 });
    expect(autoMessage(-5)).toEqual({ type: "auto", minutes: 0 });
  });

  it("reads the deadline back from the event the gateway answers with", () => {
    expect(autoDeadline({ type: "auto", until: UNTIL } as AutoEvent)).toBe(UNTIL);
    expect(autoDeadline({ type: "auto", until: null } as AutoEvent)).toBeNull();
  });

  it("reads it from the session too, so a reconnect keeps the time left", () => {
    expect(autoDeadline(summary(UNTIL))).toBe(UNTIL);
    expect(autoDeadline(summary(null))).toBeNull();
    // Recorded before the budget existed.
    expect(autoDeadline(summary(undefined as unknown as null))).toBeNull();
  });
});

describe("automatic turns", () => {
  const busy = (auto?: boolean): StatusEvent => ({
    type: "status",
    state: "busy",
    seq: 1,
    ts: UNTIL,
    ...(auto === undefined ? {} : { auto }),
  });

  it("are marked, and turns the user asked for are not", () => {
    const [turn] = applyEvent(EMPTY_TRANSCRIPT, busy(true)).turns;
    expect(turn.role === "agent" && turn.auto).toBe(true);
    const [asked] = applyEvent(EMPTY_TRANSCRIPT, busy()).turns;
    expect(asked.role === "agent" && asked.auto).toBeUndefined();
  });
});
