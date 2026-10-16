<!--
Copyright Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Web user interface

A chat beside a 3D structure viewer, for the AI Cofolding Drug Design Agent.

The agent designs molecules. Reading about a molecule is not the same as seeing
it, so every result the agent produces is shown where it was produced — next to
the prompt that caused it — rather than collected into a report at the end. That
is the one design rule this interface follows: **you can always tell what was
computed, when, and in answer to which question.**

## What one prompt can produce

| Kind | Where it comes from | How it is shown |
|------|--------------------|-----------------|
| Text | The agent's own answer | Markdown in the conversation |
| Structure | A protein-ligand complex cofolded by Boltz (a CIF or PDB file on `/shared`) | A 3D viewer, inline and in the side panel |
| Molecule | A ligand designed by REINVENT, as SMILES | A 2D structure drawing |
| Tabular | Predicted affinity, ADMET endpoints, REINVENT scores | A sortable table |

The agent does not label its output this way. Every MCP tool returns a JSON
string, and `reinvent4_agent/web/artifacts.py` classifies it. That code
is deliberately tolerant: an unrecognised payload produces no panel rather than
an error, and the raw JSON is always one click away under **Show output**.

## How it fits together

```
browser ──SSH tunnel──▶ ui container (nginx) ──proxy──▶ dd-web container
                              │                             │
                              │                             ├─ dd-mcp (stdio subprocess)
                              └─ static app                 └─ Boltz, ADMET, … over HTTP
```

Both containers run on the compute node, next to the scoring services. They
are in the `ui` profile of `docker-compose.yml`, so a plain
`docker compose up` does not start them.

- **`dd-web`** (`reinvent4_agent/web/`) is the gateway. It does not
  reimplement the agent: it plugs a WebSocket into the existing
  `AgentEventHandler` protocol, the same seam the terminal client uses for its
  Rich output. Agent behaviour is identical in both.
- **nginx** serves the built app and proxies `/api/`. The WebSocket route has a
  one-hour read timeout, because a design campaign runs for minutes with no
  traffic between events.
- **Artifacts** stay on the shared directory (`data/shared`, mounted at
  `/shared`) and are fetched over
  `/api/artifacts/<path>`. Paths are resolved and confined to that directory,
  so a crafted path cannot read anything else. Set `REINVENT4_SHARED_DIR` to
  use a different directory.

## Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /api/health` | 200 when the MCP server is up, 503 while it is not |
| `GET /api/session` | Backend, model and the list of available tools |
| `GET /api/artifacts/{path}` | One file from the shared directory |
| `WS /api/chat` | The conversation |

The event shapes are in `src/types/api.ts` on this side and in
`reinvent4_agent/web/events.py` on the other. Change them together.

## Run it on a compute node

Use the same allocation and `env.local.sh` as for the terminal agent (see
"Running with Docker Compose" in the top-level README). Then, on the compute
node:

```bash
docker compose --profile ui build dd-web ui
docker compose --profile ui up -d dd-web ui
docker compose ps                     # dd-web is healthy when MCP is up
```

This also starts the five scoring services, because `dd-web` depends on
them. It does not start the terminal agent (`reinvent4-mcp`).

Both ports listen on `127.0.0.1` of the compute node only. To open the UI
from your own machine, forward the UI port over SSH. Use the login node as a
jump host if the compute node is not reachable directly:

```bash
ssh -J <login-node> -L 8080:localhost:8080 <compute-node>
```

Then open <http://localhost:8080>. To use other ports on the compute node,
set `UI_PORT` (default 8080) and `DD_WEB_PORT` (default 8000) before
`docker compose up`.

`dd-web` uses one agent per browser tab. It does not run the Reviewer, so
`REINVENT4_AGENT_MULTI_AGENT` has no effect on the UI. If the connection
drops, the running turn stops and the next connection starts a new
conversation; the UI says so.

The `ui` container looks up `dd-web` only when it starts. Compose restarts
`ui` when it restarts `dd-web`; if you restart `dd-web` with plain `docker`,
restart `ui` too.

### Security

`dd-web` has no login. Anybody who can connect to the two ports on the
compute node can use the agent with your LLM key and read `data/shared`.
The ports listen on `127.0.0.1` only, but on a node that is shared with
other users, they can connect too. Do not use the UI on a shared node with
data that other users must not see.

The chat socket accepts browser pages only from its own host and from
`localhost`, so other web pages open in your browser cannot use the agent
through the tunnel. To allow another origin (for example, a reverse proxy),
set `REINVENT4_WEB_ALLOWED_ORIGINS` on `dd-web` to a comma-separated list
such as `http://proxy.example:8080`.

To stop:

```bash
docker compose --profile ui stop dd-web ui
```

Not all of this was tested on a compute node yet. See
[BACKEND_CHECKS.md](BACKEND_CHECKS.md) for the checks that are still open.

## Local development with live reload

You need **Node 20.19 or later, or 22.12 or later** (Vite 8 refuses to start on
anything older), and Python 3.10 or later for the mock.

`mock/mock_gateway.py` speaks the same contract as `dd-web` with canned answers
and generated PDB files, so the whole interface — viewer, tables, drawings —
runs on a laptop with no GPU, no LLM and no Slurm allocation.

Use two terminals. The mock
deliberately listens on the real gateway's port, so the default proxy target
works for either one.

```bash
# terminal 1 — the fake agent, on the same port dd-web uses
pip install fastapi uvicorn websockets
python services/ui/mock/mock_gateway.py

# terminal 2 — the app, with live reload
cd services/ui/frontend
npm install
npm run dev
```

Open the URL Vite prints (`http://localhost:5173` unless that port is taken).

### What live reload gives you

Vite's hot module replacement is on by default; there is nothing to configure
and no watch flag to add. Save a file and the browser updates in well under a
second **without reloading the page**, so React state survives: a half-typed
prompt stays in the composer, an open conversation stays on screen, and the
structure you were looking at stays loaded in the viewer. That last point
matters, because a full reload costs a new WebGL context and a refetch of the
structure file.

This applies to `.tsx`, `.ts` and `.css` alike — a Tailwind class change is
applied the same way.

Two cases still force a full reload, which is normal and not a fault:

* editing `vite.config.ts`, which restarts the dev server;
* an edit that breaks a module's exports badly enough that React Fast Refresh
  cannot patch it. The Vite overlay shows the error; fix it and the page
  recovers on the next save.

### Testing the phone layout

The layout is mobile first, so it is worth looking at on a real phone. Bind the
dev server to your network and open the address it prints from the phone:

```bash
npm run dev -- --host
```

Browser device emulation is fine for layout, but it does not reproduce a real
touch pointer, the on-screen keyboard, or a notch, and the interface adapts to
all three.

### Driving the real agent instead of the mock

Start `dd-web` on the compute node (see "Run it on a compute node" above),
forward its port to your machine, and point the proxy at it. No code change
is needed; only the environment variable differs.

```bash
ssh -L 8000:localhost:8000 <compute-node>
VITE_PROXY_TARGET=http://localhost:8000 npm run dev
```

If you open the dev server from another device (`npm run dev -- --host`),
`dd-web` refuses the chat socket from that origin. Add the origin, for
example `http://192.168.1.20:5173`, to `REINVENT4_WEB_ALLOWED_ORIGINS`.

The mock imports the real classifier from `reinvent4_agent.web.artifacts` when
it can, so even against the mock the panels are built by the same code that runs
in the real gateway.

### If edits are not picked up

File watching is native and needs no polling when the repository sits on a Linux
filesystem. Under WSL2 that is only true for a path such as `/home/you/code`.
A repository on a Windows drive, reached through `/mnt/c`, gets no inotify
events and live reload appears dead. Either move the clone onto the Linux
filesystem, which is also much faster, or turn on polling:

```ts
// vite.config.ts — only for a repository on /mnt/c
server: { watch: { usePolling: true, interval: 300 } },
```

## Checks

```bash
cd services/ui/frontend
npm run lint     # eslint
npm run build    # tsc -b && vite build

cd ../../..
python -m pytest tests/test_web_artifacts.py   # the classifier and path safety
python -m pytest tests/test_web_server.py      # the chat socket (needs the [web] extra)
```

## Layout

```
services/ui/
  BACKEND_CHECKS.md       checks still to do on a compute node
  Dockerfile              two stages: node build → nginx runtime
  nginx.conf.template     rendered at start-up; AGENT_UPSTREAM comes from docker-compose.yml
  mock/mock_gateway.py    GPU-free stand-in for dd-web
  frontend/
    src/
      App.tsx             chat on the left, viewer on the right
      types/api.ts        the gateway contract
      hooks/useChat.ts    the WebSocket and the transcript
      hooks/useTheme.ts   light/dark, remembered per browser
      components/
        Transcript/       the conversation
        ToolBlock/        one tool call and the data it returned
        StructureCard/    a complex, inline
        StructureViewer/  3Dmol
        MoleculeCard/     a ligand drawn from SMILES
        DataTable/        sortable predictions
        WorkspacePanel/   the viewer beside the chat
        Composer/         the prompt box
        Header/           title, connection state, theme
```

## Notes

- **Theming** is Tailwind v4 with semantic tokens in `src/index.css`. Components
  use `bg-raised`, `text-muted` and so on, never a raw colour, so a theme change
  never becomes a component change.
- **The layout is mobile first.** One column is the base case: the conversation
  fills the screen and the viewer opens over it as a full-screen sheet. The
  side-by-side arrangement is an enhancement that starts at `lg`, where a viewer
  is finally wide enough to rotate a protein in. Write the phone case first and
  add `sm:`, `lg:` and `xl:` on top; do not write a desktop layout and hide
  parts of it on small screens.
- **Tap targets are sized by input type, not by screen width.** Use the `touch:`
  variant, defined in `src/index.css` as `@media (pointer: coarse)`, to reach
  44px. A width breakpoint gets one of the two cases wrong: a 768px tablet is a
  touch device and needs large targets, while a 768px desktop window does not.
- **3Dmol needs WebGL.** Where WebGL is unavailable the viewer says so and the
  structure file is still downloadable from the card.
- **The bundle is large** (~250 kB gzipped, plus ~165 kB for 3Dmol). 3Dmol is
  loaded on demand, so the app paints before the viewer arrives. smiles-drawer
  is in the main bundle because almost every answer contains a molecule.
- **`package-lock.json` is committed.** The image build runs `npm ci`, which
  requires it.
