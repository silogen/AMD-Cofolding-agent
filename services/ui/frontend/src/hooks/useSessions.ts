// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useCallback, useEffect, useState } from "react";
import type { SessionSummary } from "../types/api";
import { ApiError, deleteSession, listSessions, renameSession } from "../lib/api";

/** How often the list is refreshed, so turns started in other tabs show up. */
const POLL_MS = 15000;

interface SessionsState {
  sessions: SessionSummary[];
  error: string | null;
  refresh: () => void;
  rename: (id: string, title: string) => Promise<void>;
  remove: (id: string) => Promise<void>;
}

/** The saved conversations, newest first. */
export function useSessions(): SessionsState {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    listSessions()
      .then((list) => {
        setSessions(list);
        setError(null);
      })
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : String(reason)));
  }, []);

  useEffect(() => {
    refresh();
    const timer = window.setInterval(() => {
      if (document.visibilityState === "visible") refresh();
    }, POLL_MS);
    const onVisible = () => {
      if (document.visibilityState === "visible") refresh();
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [refresh]);

  const rename = useCallback(
    async (id: string, title: string) => {
      const updated = await renameSession(id, title);
      setSessions((previous) => previous.map((s) => (s.id === id ? { ...s, ...updated } : s)));
    },
    [],
  );

  const remove = useCallback(async (id: string) => {
    // A purge from the data view, or another tab, may have deleted it already.
    await deleteSession(id).catch((reason: unknown) => {
      if (!(reason instanceof ApiError && reason.status === 404)) throw reason;
    });
    setSessions((previous) => previous.filter((s) => s.id !== id));
  }, []);

  return { sessions, error, refresh, rename, remove };
}
