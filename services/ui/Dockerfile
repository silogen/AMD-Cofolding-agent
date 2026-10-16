# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ─────────────────────────────────────────────
# AI Cofolding Drug Design Agent — web UI
# Stage 1 – build the React/Vite single-page app
# Stage 2 – serve it with nginx and proxy the agent gateway
# ─────────────────────────────────────────────

# ── build stage ──────────────────────────────
FROM node:22-alpine AS builder

WORKDIR /app

# Dependencies first, so a source-only change reuses the cached install.
COPY services/ui/frontend/package.json services/ui/frontend/package-lock.json ./
# --include=dev is explicit: the build needs TypeScript, Vite and Tailwind,
# and a builder with NODE_ENV=production would otherwise omit them.
RUN npm ci --prefer-offline --include=dev

COPY services/ui/frontend/ ./
RUN npm run build

# ── runtime stage ────────────────────────────
FROM nginx:1.27-alpine

RUN rm /etc/nginx/conf.d/default.conf

# Templates in this directory are rendered with envsubst by the nginx
# entrypoint, which is how AGENT_UPSTREAM reaches the config at start-up.
COPY services/ui/nginx.conf.template /etc/nginx/templates/default.conf.template
COPY services/ui/security-headers.conf /etc/nginx/snippets/security-headers.conf

COPY --from=builder /app/dist /usr/share/nginx/html

# Where the agent gateway listens. docker-compose.yml overrides this with
# the dd-web service; the default matches dd-web's own default port.
ENV AGENT_UPSTREAM="localhost:8000"

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -qO- http://localhost:8080/nginx-health || exit 1

CMD ["nginx", "-g", "daemon off;"]
