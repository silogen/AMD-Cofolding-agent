// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useMemo, useRef, useState } from "react";
import { fetchCofoldStatus } from "../../lib/api";
import { cofoldPollDelay, EMPTY_MEMORY, learn, ownStatus, type CofoldCall, type CofoldMemory, type CofoldStatus } from "../../lib/cofold";

const UNAVAILABLE: CofoldStatus = { available: false };

function usePageVisible(): boolean {
  const [visible, setVisible] = useState(() => !document.hidden);
  useEffect(() => {
    const onChange = () => setVisible(!document.hidden);
    document.addEventListener("visibilitychange", onChange);
    return () => document.removeEventListener("visibilitychange", onChange);
  }, []);
  return visible;
}

/**
 * The last answer of /api/services/cofold as far as this conversation is
 * concerned (see ownStatus), asked for only while a cofold is believed to run
 * (see cofoldPollDelay) and never while the page is hidden. *running*: a run
 * of this conversation is going on.
 */
export function useCofoldStatus(
  calls: readonly CofoldCall[],
  boltzRun: boolean | null,
  running: boolean,
): { status: CofoldStatus | null; at: number; memory: CofoldMemory } {
  const visible = usePageVisible();
  const [read, setRead] = useState<{ status: CofoldStatus; at: number; memory: CofoldMemory } | null>(null);
  const lastAt = useRef(0);
  const runScoring = running && boltzRun !== false;
  const own = useMemo(() => (read ? ownStatus(read.status, calls, runScoring) : null), [read, calls, runScoring]);
  const lastActive = own !== null && own.available && own.active;
  const delay = cofoldPollDelay({ visible, calls, boltzRun, running, lastActive });

  useEffect(() => {
    if (delay === null) return;
    let cancelled = false;
    let timer = 0;
    const poll = async () => {
      lastAt.current = Date.now();
      const status = await fetchCofoldStatus().catch(() => UNAVAILABLE);
      if (cancelled) return;
      setRead((previous) => ({ status, at: Date.now(), memory: learn(previous?.memory ?? EMPTY_MEMORY, status) }));
      timer = window.setTimeout(() => void poll(), delay);
    };
    timer = window.setTimeout(() => void poll(), Math.max(0, lastAt.current + delay - Date.now()));
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [delay]);

  return { status: own, at: read?.at ?? 0, memory: read?.memory ?? EMPTY_MEMORY };
}
