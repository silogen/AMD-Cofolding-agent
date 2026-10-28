// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useEffect, useRef, useState } from "react";
import { fetchCofoldStatus } from "../../lib/api";
import { cofoldPollDelay, type CofoldCall, type CofoldStatus } from "../../lib/cofold";

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
 * The last answer of /api/services/cofold, asked for only while a cofold is
 * believed to run (see cofoldPollDelay) and never while the page is hidden.
 */
export function useCofoldStatus(calls: readonly CofoldCall[], boltzRun: boolean | null, running: boolean): { status: CofoldStatus | null; at: number } {
  const visible = usePageVisible();
  const [read, setRead] = useState<{ status: CofoldStatus; at: number } | null>(null);
  const lastAt = useRef(0);
  const lastActive = read !== null && read.status.available && read.status.active;
  const delay = cofoldPollDelay({ visible, calls, boltzRun, running, lastActive });

  useEffect(() => {
    if (delay === null) return;
    let cancelled = false;
    let timer = 0;
    const poll = async () => {
      lastAt.current = Date.now();
      const status = await fetchCofoldStatus().catch(() => UNAVAILABLE);
      if (cancelled) return;
      setRead({ status, at: Date.now() });
      timer = window.setTimeout(() => void poll(), delay);
    };
    timer = window.setTimeout(() => void poll(), Math.max(0, lastAt.current + delay - Date.now()));
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [delay]);

  return { status: read?.status ?? null, at: read?.at ?? 0 };
}
