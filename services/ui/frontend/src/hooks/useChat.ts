// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  AgentEvent,
  AgentTurn,
  ConnectionState,
  Panel,
  Turn,
  TurnBlock,
} from "../types/api";

/** Reconnect delays in ms. The last value repeats for later attempts. */
const RETRY_DELAYS = [1000, 2000, 5000, 10000];

function chatUrl(): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/api/chat`;
}

let nextId = 0;
const makeId = (prefix: string) => `${prefix}-${++nextId}`;

/**
 * Close a socket without the "closed before the connection is established"
 * error.
 *
 * Calling close() on a socket that is still CONNECTING makes the browser raise
 * that error. React's StrictMode mounts, unmounts and remounts every component
 * in development, so the first socket is always torn down mid-handshake and the
 * error appears on every page load. Waiting for the handshake and closing then
 * is the only way to end it cleanly.
 */
function closeSocket(socket: WebSocket | null): void {
  if (!socket) return;
  if (socket.readyState === WebSocket.CONNECTING) {
    socket.addEventListener("open", () => socket.close(), { once: true });
    return;
  }
  if (socket.readyState === WebSocket.OPEN) socket.close();
}

interface ChatState {
  turns: Turn[];
  connection: ConnectionState;
  busy: boolean;
  model: string | null;
  tools: string[];
  /** Every structure seen this session, newest first, for the side panel. */
  structures: Panel[];
  send: (text: string) => void;
  reconnect: () => void;
}

/**
 * Holds the conversation and the WebSocket to the agent gateway.
 *
 * The transcript is built by appending blocks to the turn that is open, so
 * thinking, tool calls, tool results and the final answer stay in the order
 * the agent produced them. That order is what lets a user see what was
 * computed when, instead of only reading a finished report.
 */
export function useChat(): ChatState {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [busy, setBusy] = useState(false);
  const [model, setModel] = useState<string | null>(null);
  const [tools, setTools] = useState<string[]>([]);

  const socketRef = useRef<WebSocket | null>(null);
  const attemptRef = useRef(0);
  const retryRef = useRef<number | null>(null);
  const closedByUs = useRef(false);

  /** Append a block to the open agent turn, starting one if needed. */
  const appendBlock = useCallback((block: TurnBlock) => {
    setTurns((previous) => {
      const last = previous[previous.length - 1];
      if (last && last.role === "agent" && !last.done) {
        return [...previous.slice(0, -1), { ...last, blocks: [...last.blocks, block] }];
      }
      const turn: AgentTurn = {
        role: "agent",
        id: makeId("turn"),
        at: Date.now(),
        blocks: [block],
        done: false,
      };
      return [...previous, turn];
    });
  }, []);

  const handleEvent = useCallback(
    (event: AgentEvent) => {
      switch (event.type) {
        case "ready":
          setModel(event.model);
          setTools(event.tools);
          break;

        case "status":
          setBusy(event.state === "busy");
          if (event.state === "idle") {
            // Close the open turn so the next prompt starts a new one.
            setTurns((previous) => {
              const last = previous[previous.length - 1];
              if (last && last.role === "agent" && !last.done) {
                return [...previous.slice(0, -1), { ...last, done: true }];
              }
              return previous;
            });
          }
          break;

        case "thinking":
          appendBlock({ kind: "thinking", id: makeId("think"), content: event.content });
          break;

        case "tool_call":
          appendBlock({
            kind: "tool",
            id: event.id,
            name: event.name,
            arguments: event.arguments,
          });
          break;

        case "tool_result":
          // Attach the result to its call, so the pair renders as one block.
          setTurns((previous) => {
            const last = previous[previous.length - 1];
            if (!last || last.role !== "agent") return previous;
            const index = findOpenToolCall(last.blocks, event.id, event.name);
            if (index < 0) return previous;
            const blocks = [...last.blocks];
            blocks[index] = { ...(blocks[index] as Extract<TurnBlock, { kind: "tool" }>), result: event };
            return [...previous.slice(0, -1), { ...last, blocks }];
          });
          break;

        case "assistant_message":
          appendBlock({ kind: "text", id: makeId("text"), content: event.content });
          setTurns((previous) => {
            const last = previous[previous.length - 1];
            if (last && last.role === "agent") {
              return [
                ...previous.slice(0, -1),
                { ...last, tokens: { input: event.inputTokens, output: event.outputTokens } },
              ];
            }
            return previous;
          });
          break;

        case "monitor_response":
          appendBlock({ kind: "text", id: makeId("monitor"), content: event.content });
          break;

        case "monitor_alert":
          appendBlock({
            kind: "text",
            id: makeId("alert"),
            content: `**Monitor alert**\n\n\`\`\`json\n${JSON.stringify(event.alert, null, 2)}\n\`\`\``,
          });
          break;

        case "error":
          appendBlock({ kind: "error", id: makeId("err"), message: event.message });
          setBusy(false);
          break;
      }
    },
    [appendBlock],
  );

  // connect() schedules its own retry, so it has to refer to itself. The ref
  // breaks that cycle: the callback reads the current connect through the ref
  // instead of closing over a value that does not exist yet.
  const connectRef = useRef<() => void>(() => {});

  /**
   * Open the socket. This deliberately does not set the connection state:
   * "connecting" is the initial state, and every later transition is reported
   * by the socket's own handlers, so opening the socket stays a pure effect
   * on an external system.
   */
  const connect = useCallback(() => {
    if (retryRef.current !== null) {
      window.clearTimeout(retryRef.current);
      retryRef.current = null;
    }

    const socket = new WebSocket(chatUrl());
    socketRef.current = socket;

    // A socket that is no longer the current one must not touch state. React
    // StrictMode opens a socket, tears it down and opens another on every
    // mount in development, and the first one can still finish its handshake
    // and fire onclose afterwards. Without this check that stale close reads
    // as a dropped connection and schedules a reconnect, so the session ends
    // up with three sockets and a header stuck on "Reconnecting".
    const isCurrent = () => socketRef.current === socket;

    socket.onopen = () => {
      if (!isCurrent()) return;
      attemptRef.current = 0;
      setConnection("open");
    };

    socket.onmessage = (message) => {
      if (!isCurrent()) return;
      try {
        handleEvent(JSON.parse(message.data as string) as AgentEvent);
      } catch {
        // A frame we cannot parse is a gateway bug, not a user error. Keep
        // the session alive rather than tearing down the conversation.
        console.error("Ignoring unparseable event from the gateway");
      }
    };

    socket.onerror = () => {
      if (!isCurrent()) return;
      setConnection("error");
    };

    socket.onclose = () => {
      if (!isCurrent()) return;
      setBusy(false);
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

  useEffect(() => {
    closedByUs.current = false;
    connect();
    return () => {
      closedByUs.current = true;
      if (retryRef.current !== null) window.clearTimeout(retryRef.current);
      closeSocket(socketRef.current);
      socketRef.current = null;
    };
  }, [connect]);

  const send = useCallback((text: string) => {
    const content = text.trim();
    const socket = socketRef.current;
    if (!content || !socket || socket.readyState !== WebSocket.OPEN) return;

    setTurns((previous) => [
      ...previous,
      { role: "user", id: makeId("user"), content, at: Date.now() },
    ]);
    setBusy(true);
    socket.send(JSON.stringify({ type: "user_message", content }));
  }, []);

  const reconnect = useCallback(() => {
    attemptRef.current = 0;
    setConnection("connecting");
    closeSocket(socketRef.current);
    connect();
  }, [connect]);

  const structures = collectStructures(turns);

  return { turns, connection, busy, model, tools, structures, send, reconnect };
}

/**
 * Find the tool block a result belongs to.
 *
 * The gateway sends an id, so that is the reliable match. It can be null when
 * a result arrives for a call the handler never saw, in which case the last
 * unanswered call of the same name is the best available guess.
 */
function findOpenToolCall(blocks: TurnBlock[], id: string | null, name: string): number {
  if (id) {
    const byId = blocks.findIndex((block) => block.kind === "tool" && block.id === id);
    if (byId >= 0) return byId;
  }
  for (let index = blocks.length - 1; index >= 0; index -= 1) {
    const block = blocks[index];
    if (block.kind === "tool" && block.name === name && !block.result) return index;
  }
  return -1;
}

/** Every structure produced this session, newest first, without duplicates. */
function collectStructures(turns: Turn[]): Panel[] {
  const seen = new Set<string>();
  const out: Panel[] = [];
  for (let index = turns.length - 1; index >= 0; index -= 1) {
    const turn = turns[index];
    if (turn.role !== "agent") continue;
    for (const block of turn.blocks) {
      if (block.kind !== "tool" || !block.result) continue;
      for (const panel of block.result.panels) {
        if (panel.kind !== "structure" || seen.has(panel.path)) continue;
        seen.add(panel.path);
        out.push(panel);
      }
    }
  }
  return out;
}
