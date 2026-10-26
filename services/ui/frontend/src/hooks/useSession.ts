// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import type { AgentEvent, ClientMessage, ConnectionState, ConversationEvent, SessionSummary } from "../types/api";
import { applyEvent, EMPTY_TRANSCRIPT, type Transcript } from "../lib/transcript";

/** Reconnect delays in ms. The last value repeats for later attempts. */
const RETRY_DELAYS = [1000, 2000, 5000, 10000];

/** Which conversation to attach to. A new nonce forces a fresh attach. */
export interface SessionRequest {
  /** null starts a new conversation. */
  id: string | null;
  nonce: number;
}

export interface AgentInfo {
  model: string;
  backend: string;
  reasoningEffort: string | null;
  tools: string[];
}

export interface Notice {
  id: number;
  message: string;
}

interface SessionState {
  transcript: Transcript;
  session: SessionSummary | null;
  connection: ConnectionState;
  agent: AgentInfo | null;
  /** Set when the gateway refuses connections, e.g. while it starts. */
  gatewayError: string | null;
  notices: Notice[];
  dismissNotice: (id: number) => void;
  send: (text: string) => boolean;
  cancel: () => void;
  reconnect: () => void;
}

type Action = { type: "reset" } | { type: "event"; event: ConversationEvent };

function reducer(state: Transcript, action: Action): Transcript {
  return action.type === "reset" ? EMPTY_TRANSCRIPT : applyEvent(state, action.event);
}

function chatUrl(sessionId: string | null, after: number): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const query = new URLSearchParams();
  if (sessionId) query.set("session", sessionId);
  if (after > 0) query.set("after", String(after));
  const text = query.toString();
  const suffix = text ? `?${text}` : "";
  return `${protocol}//${window.location.host}/api/chat${suffix}`;
}

/**
 * Close a socket without the "closed before the connection is established"
 * error, which React StrictMode would otherwise trigger on every mount in
 * development.
 */
function closeSocket(socket: WebSocket | null): void {
  if (!socket) return;
  if (socket.readyState === WebSocket.CONNECTING) {
    socket.addEventListener("open", () => socket.close(), { once: true });
    return;
  }
  if (socket.readyState === WebSocket.OPEN) socket.close();
}

let noticeId = 0;

/**
 * One saved conversation and the WebSocket attached to it.
 *
 * The conversation lives in the gateway, not here: closing the tab does not
 * stop a turn, and a reconnect asks for the events after the last one seen,
 * so a dropped connection loses nothing. *onAttached* reports the session
 * id the gateway chose, which is how a new conversation gets its address.
 */
export function useSession(request: SessionRequest, onAttached: (session: SessionSummary) => void): SessionState {
  const [transcript, dispatch] = useReducer(reducer, EMPTY_TRANSCRIPT);
  const [session, setSession] = useState<SessionSummary | null>(null);
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [agent, setAgent] = useState<AgentInfo | null>(null);
  const [gatewayError, setGatewayError] = useState<string | null>(null);
  const [notices, setNotices] = useState<Notice[]>([]);

  const socketRef = useRef<WebSocket | null>(null);
  const attemptRef = useRef(0);
  const retryRef = useRef<number | null>(null);
  const closedByUs = useRef(false);
  /** The session the gateway attached us to; reconnects go back to it. */
  const attachedRef = useRef<string | null>(null);
  /** The session whose events are on screen. */
  const shownRef = useRef<string | null>(null);
  const lastSeqRef = useRef(0);
  const onAttachedRef = useRef(onAttached);

  useEffect(() => {
    onAttachedRef.current = onAttached;
  }, [onAttached]);

  useEffect(() => {
    lastSeqRef.current = transcript.lastSeq;
  }, [transcript.lastSeq]);

  const pushNotice = useCallback((message: string) => {
    setNotices((previous) => [...previous.slice(-3), { id: ++noticeId, message }]);
  }, []);

  const handleEvent = useCallback(
    (event: AgentEvent) => {
      switch (event.type) {
        case "ready":
          setGatewayError(null);
          setAgent({
            model: event.model,
            backend: event.backend,
            reasoningEffort: event.reasoningEffort ?? null,
            tools: event.tools,
          });
          return;
        case "session": {
          const summary: SessionSummary = {
            id: event.id,
            title: event.title,
            created: event.created,
            updated: event.updated,
            turns: event.turns,
            busy: event.busy,
            inputTokens: event.inputTokens,
            outputTokens: event.outputTokens,
          };
          // What is on screen belongs to another conversation: start over.
          // The replay that follows fills it in.
          if (shownRef.current !== summary.id) {
            dispatch({ type: "reset" });
            lastSeqRef.current = 0;
            shownRef.current = summary.id;
          }
          attachedRef.current = summary.id;
          setSession(summary);
          onAttachedRef.current(summary);
          return;
        }
        case "notice":
          pushNotice(event.message);
          return;
        default:
          if (event.type === "error" && typeof event.seq !== "number") {
            // Sent before the gateway closes a connection it cannot serve.
            setGatewayError(event.message);
            return;
          }
          dispatch({ type: "event", event });
          if (event.type === "user_message" || event.type === "status") {
            setSession((previous) =>
              previous
                ? {
                    ...previous,
                    busy: event.type === "status" ? event.state === "busy" : previous.busy,
                    title: previous.title || (event.type === "user_message" ? event.content.split("\n")[0] : ""),
                  }
                : previous,
            );
          }
      }
    },
    [pushNotice],
  );

  const connectRef = useRef<() => void>(() => {});

  const connect = useCallback(() => {
    if (retryRef.current !== null) {
      window.clearTimeout(retryRef.current);
      retryRef.current = null;
    }

    const socket = new WebSocket(chatUrl(attachedRef.current, lastSeqRef.current));
    socketRef.current = socket;
    const isCurrent = () => socketRef.current === socket;

    socket.onopen = () => {
      if (!isCurrent()) return;
      attemptRef.current = 0;
      setConnection("open");
    };

    socket.onmessage = (message) => {
      if (!isCurrent()) return;
      let event: AgentEvent;
      try {
        event = JSON.parse(message.data as string) as AgentEvent;
      } catch {
        console.error("Ignoring unparseable event from the gateway");
        return;
      }
      handleEvent(event);
    };

    socket.onerror = () => {
      if (!isCurrent()) return;
      setConnection("error");
    };

    socket.onclose = () => {
      if (!isCurrent()) return;
      if (closedByUs.current) return;
      setConnection("closed");
      const delay = RETRY_DELAYS[Math.min(attemptRef.current, RETRY_DELAYS.length - 1)];
      attemptRef.current += 1;
      retryRef.current = window.setTimeout(() => connectRef.current(), delay);
    };
  }, [handleEvent]);

  useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

  // Attach whenever a different conversation is asked for. The transcript
  // is reset when the gateway confirms the session, not here, so the
  // screen never shows an empty conversation that is not one.
  useEffect(() => {
    closedByUs.current = false;
    if (attachedRef.current !== request.id) {
      attachedRef.current = request.id;
      lastSeqRef.current = 0;
      shownRef.current = null;
    }
    attemptRef.current = 0;
    connect();
    return () => {
      closedByUs.current = true;
      if (retryRef.current !== null) window.clearTimeout(retryRef.current);
      closeSocket(socketRef.current);
      socketRef.current = null;
    };
    // request.id is read through the nonce: the same id with a new nonce
    // (e.g. "new conversation" twice) must still re-attach.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [request.nonce, connect]);

  const sendMessage = useCallback((message: ClientMessage): boolean => {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) return false;
    socket.send(JSON.stringify(message));
    return true;
  }, []);

  // No local echo: the gateway sends the user's message back as an event,
  // so every open tab shows it, in the order the gateway recorded it.
  const send = useCallback(
    (text: string) => {
      const content = text.trim();
      return content ? sendMessage({ type: "user_message", content }) : false;
    },
    [sendMessage],
  );

  const cancel = useCallback(() => {
    sendMessage({ type: "cancel" });
  }, [sendMessage]);

  const reconnect = useCallback(() => {
    attemptRef.current = 0;
    setConnection("connecting");
    closeSocket(socketRef.current);
    connect();
  }, [connect]);

  const dismissNotice = useCallback((id: number) => {
    setNotices((previous) => previous.filter((notice) => notice.id !== id));
  }, []);

  return {
    transcript,
    session,
    connection,
    agent,
    gatewayError,
    notices,
    dismissNotice,
    send,
    cancel,
    reconnect,
  };
}
