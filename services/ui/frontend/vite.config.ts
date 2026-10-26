// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

/**
 * Dev proxy
 * ---------
 * The app calls /api/... as RELATIVE paths, and opens the chat WebSocket at
 * /api/chat. In production nginx proxies both to the agent gateway (see
 * services/ui/nginx.conf). In dev there is no nginx, so Vite's proxy fills the
 * same role and forwards them to VITE_PROXY_TARGET.
 *
 * The default target is the mock gateway (services/ui/mock/mock_gateway.py) on
 * :8000 — no GPU, no LLM, instant and deterministic. To drive the real agent:
 *
 *     VITE_PROXY_TARGET=http://localhost:8000 npm run dev    # real, local
 *     VITE_PROXY_TARGET=http://some-host:8000 npm run dev    # real, remote
 *
 * The mock listens on 8000 too, so the default just works once it is running.
 */
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const target = env.VITE_PROXY_TARGET || "http://localhost:8000";
  // Sent on every proxied request, the chat socket included, when dd-web
  // runs with REINVENT4_WEB_TOKEN. The browser never sees it.
  const headers = env.VITE_PROXY_TOKEN ? { "X-DD-Token": env.VITE_PROXY_TOKEN } : undefined;

  return {
    plugins: [react(), tailwindcss()],
    server: {
      proxy: {
        // ws: true also upgrades /api/chat, which is the chat WebSocket.
        "/api": { target, changeOrigin: true, ws: true, headers },
      },
    },
    build: {
      // 3Dmol is large, and StructureViewer imports it dynamically, so Rollup
      // already gives it its own chunk: the app shell paints without it and
      // the browser caches the viewer separately. Raise the warning limit so
      // that expected chunk does not look like a regression on every build.
      chunkSizeWarningLimit: 1500,
    },
  };
});
