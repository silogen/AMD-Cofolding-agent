// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useCallback, useEffect, useState } from "react";
import type { Effort, ModelOption, ModelsState } from "../types/api";
import { ApiError, request } from "./api";

/**
 * The served models, and switching between them.
 *
 * Only one model runs on the cluster at a time, so choosing the other one
 * restarts the server: the POST answers 202 with state "loading" and the
 * gateway is unreachable for three to six minutes, which is why this polls
 * instead of waiting on the response. Deployments without a cluster behind
 * them have no endpoint at all; then nothing here is available and the UI
 * keeps showing the one model the ready event named.
 *
 * The hook lives next to the contract rather than in src/hooks because the
 * polling loop is the contract: it is how a 202 is read to its end.
 */

/** How often to ask whether the new model is up, in ms. */
const POLL_MS = 4000;

export const loadModels = () => request<ModelsState>("/api/models");

export const switchTo = (key: string) => request<ModelsState>("/api/models", { method: "POST", body: JSON.stringify({ key }) });

/** The entry for *key*, or undefined when the gateway does not serve it. */
export function modelOf(state: ModelsState | null, key: string | null): ModelOption | undefined {
  return key ? state?.models.find((model) => model.key === key) : undefined;
}

/** What to call *key* on screen; the key itself when it is not in the list. */
export function modelLabel(state: ModelsState | null, key: string | null): string {
  return modelOf(state, key)?.label ?? key ?? "";
}

/**
 * The thinking level to send: the remembered one when this model offers it,
 * else the model's own default. Null for a model without levels, so a
 * remembered level never reaches one.
 */
export function pickEffort(efforts: Effort[], chosen: Effort | null, fallback: Effort | null): Effort | null {
  if (!efforts.length) return null;
  if (chosen && efforts.includes(chosen)) return chosen;
  if (fallback && efforts.includes(fallback)) return fallback;
  return efforts[0];
}

/** One line for the menu explaining why a switch did not happen. */
export function switchNote(reason: unknown): string {
  if (reason instanceof ApiError) {
    if (reason.status === 409) return "A turn is running. Wait for it to finish, then switch.";
    if (reason.status === 503) return "Switching models is not available here.";
    if (reason.status === 400) return "That model is not served.";
  }
  return reason instanceof Error ? reason.message : String(reason);
}

export interface ModelsView {
  /** What the gateway last said, null while unknown or when it has no endpoint. */
  state: ModelsState | null;
  /** The model being loaded, by key, null when none is. */
  pending: string | null;
  /** A note for the menu: a refused switch, or one that failed. */
  note: string | null;
  /** Ask for *key*; ignored for the current model or when switching is off. */
  choose: (key: string) => void;
}

export function useModels(): ModelsView {
  const [state, setState] = useState<ModelsState | null>(null);
  const [pending, setPending] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    loadModels().then(
      (next) => {
        if (!alive) return;
        setState(next);
        // A switch another tab started, or one begun before this page loaded.
        if (next.state === "loading") setPending(next.current);
      },
      () => undefined,
    );
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (!pending) return;
    let alive = true;
    const timer = window.setInterval(() => {
      loadModels().then(
        (next) => {
          if (!alive) return;
          setState(next);
          if (next.state === "ready") {
            setPending(null);
            setNote(null);
          } else if (next.state === "failed") {
            setPending(null);
            setNote(`${modelLabel(next, pending)} could not be loaded.`);
          }
        },
        // The server is restarting: unreachable is the expected answer here.
        () => undefined,
      );
    }, POLL_MS);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, [pending]);

  const choose = useCallback(
    (key: string) => {
      if (!state || pending || !state.switchable || key === state.current) return;
      setNote(null);
      void switchTo(key).then(
        (next) => {
          setState(next);
          setPending(key);
        },
        (reason) => setNote(switchNote(reason)),
      );
    },
    [state, pending],
  );

  return { state, pending, note, choose };
}
