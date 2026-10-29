<!--
Copyright Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Web user interface

The browser app for the cofolding agent: saved conversations, a 3D structure
viewer (Mol*), molecule drawings (RDKit.js), a sketcher (Ketcher), and the
REINVENT, Boltz and ADMET results each prompt produced.

```
browser ── Vite (laptop) ──SSH tunnel──▶ dd-web (compute node) ─┬─ dd-mcp: REINVENT, jobs
                 adds the access token                          └─ Boltz, ADMET, … over HTTP
```

`dd-web` (`reinvent4_agent/web/`) is the gateway; it runs the same agent as
the terminal client. The app talks to it only through `/api`.

## Run it

- **On LUMI:** see [services/lumi/README.md](../lumi/README.md).
- **With Docker Compose:** `docker compose --profile ui up -d dd-web ui` on the
  compute node, then `ssh -L 8080:localhost:8080 <compute-node>` and open
  <http://localhost:8080>. This path has not been tested with the current UI.

## Develop

Node 20.19+ or 22.12+. With a session running on LUMI, the dev server
reloads on every save:

```powershell
.\services\lumi\start-ui.ps1 -Dev          # or: DEV=1 bash services/lumi/start-ui.sh
```

Against any other gateway, set `VITE_PROXY_TARGET` (and `VITE_PROXY_TOKEN`
if it was started with `REINVENT4_WEB_TOKEN`) before `npm run dev`.
`mock/mock_gateway.py` answers the chat only; the side panels need a real
gateway.

The event and API shapes are in `frontend/src/types/api.ts` and
`reinvent4_agent/web/events.py`; change them together.

Checks, from `services/ui/frontend`:

```bash
npx tsc -b && npx eslint src && npx vitest run && npx vite build
```

The gateway tests are `tests/test_web_*.py` at the repository root.

## Conventions

- Colours come from the semantic tokens in `src/index.css` (`bg-raised`,
  `text-muted`, …), never raw values.
- The layout is mobile first; `lg:` adds the side-by-side view. Touch targets
  use the `touch:` variant (`pointer: coarse`), not a width breakpoint.
- `package-lock.json` is committed; the image build runs `npm ci`.
