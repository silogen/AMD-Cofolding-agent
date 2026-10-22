# Web UI: checks that need a real backend

The web UI and the `dd-web` gateway came from the Kubernetes version of this
agent. In this repository they were checked only with static checks and with
the mock gateway, because no GPU allocation was available. Do the checks
below on a compute node before you use the UI for real work.

## What is already checked

| Check | Command | Result |
|-------|---------|--------|
| Frontend lint | `npm run lint` in `services/ui/frontend` | Pass |
| Frontend type check and build | `npm run build` in `services/ui/frontend` | Pass. 3Dmol gives an `eval` warning from its own code. |
| Classifier and path safety | `python -m pytest tests/test_web_artifacts.py` | 17 pass |
| Chat socket with a fake agent | `python -m pytest tests/test_web_server.py` | 14 pass: Origin check, frames that are not JSON objects, a disconnect cancels the running turn |
| Gateway without MCP | `create_app()` with a missing `dd-mcp` | `/api/health` gives 503, `/api/chat` sends an error, `../` paths give 404 |
| nginx config | `nginx:1.27-alpine` with the template, the header file and the built app | Starts with no warning. Security headers on `/`, assets and `/api/`. A missing gateway gives 503 with the JSON message. |
| Mock gateway | `python services/ui/mock/mock_gateway.py` | Loads the real classifier from the repository root. ADMET answer gives molecule and table panels. |
| Compose file | Parsed with PyYAML | `dd-web` has the same environment, volumes and dependencies as `reinvent4-mcp`. `docker compose config` was not run: Compose is not installed here. |

The `ui` image was not built here: `npm ci` inside the build container
stopped with the npm error "Exit handler never called", which is a network
or npm problem of this machine. The same `npm ci` works outside Docker.

## What must be checked on a compute node

### 1. The images build

- [ ] `docker compose --profile ui build dd-web ui` completes.
- [ ] The `dd-agent` image installs the `[web]` extra (`fastapi`, `uvicorn`
      and `websockets`) without a version conflict with the REINVENT4
      dependencies. The
      `pip install` step pins `numpy` and `scipy` last, so make sure the
      final import check in `services/dd-agent/Dockerfile` still passes.
- [ ] `docker compose config` accepts the file (the YAML anchors
      `agent-volumes`, `agent-environment` and `agent-depends-on`, the `ui`
      profile, and `restart: true` under `depends_on`, which needs Compose
      2.17 or later).
- [ ] After `docker compose --profile ui up -d --force-recreate dd-web`,
      the `ui` container is restarted too and `/api/health` through port
      8080 works.

### 2. The stack starts

- [ ] `docker compose --profile ui up -d dd-web ui` starts the five scoring
      services, `dd-web` and `ui`, and does not start `reinvent4-mcp`.
- [ ] `dd-web` becomes healthy. Its health check calls `/api/health`, which
      gives 503 until `dd-mcp` is up. If `start_period: 60s` is too short
      for `dd-mcp` to start, make it longer.
- [ ] `curl localhost:8000/api/session` shows the backend, the model and
      the MCP tools (24 tools at the time of the migration).
- [ ] `curl localhost:8080/nginx-health` gives `ok`.
- [ ] `tty: true` from `rocm-common` does not cause a problem for `dd-web`.
      If the logs have no timestamps, set `REINVENT4_AGENT_LOG_TIMESTAMPS=1`.

### 3. The UI can be opened from a laptop

- [ ] `ssh -J <login-node> -L 8080:localhost:8080 <compute-node>` works on
      the target cluster. Some clusters do not let users connect to compute
      nodes with SSH. If so, find the supported method (for example,
      `srun --pty` with port forwarding, or the cluster's own web proxy) and
      update `services/ui/README.md` and the top-level README.
- [ ] The page loads at `http://localhost:8080`, and the header shows the
      model name and the tool count.

### 4. The chat works end to end

- [ ] A simple question (for example, "What scoring components are
      available?") gets an answer. Tool calls and results show in the chat.
- [ ] The WebSocket stays open during a long Boltz run. nginx keeps the
      socket for one hour without traffic. Check that no SSH or cluster
      timeout closes it first.
- [ ] A second message while the agent works gives the message "The agent
      is still working on the previous request." The composer stays
      disabled until the turn ends.
- [ ] Close the tab while a tool runs. The `dd-web` log shows "Cancelling
      the turn of a disconnected client". The MCP connection still works
      for the next tab (a cancelled MCP call must not break the shared
      `dd-mcp` session). A job that was already started keeps running;
      that is expected.
- [ ] After a reconnect, the UI shows that a new conversation started.
- [ ] A slow LLM answer does not drop the connection. `Agent.chat` calls
      the LLM with a blocking call on the gateway's event loop, so during
      an answer `dd-web` cannot answer WebSocket pings (uvicorn closes the
      socket after about 40 s), `/api/health`, or other tabs. If this
      happens, the fix is in `reinvent4_agent/agent.py` (run
      `backend.complete` with `asyncio.to_thread`). That is a change to the
      agent itself, so it was not part of the UI migration.
- [ ] The LLM endpoint and key in `env.local.sh` reach `dd-web` the same way
      they reach `reinvent4-mcp` (the same environment anchor).

### 5. The panels show real results

- [ ] After a cofolding run (Boltz), a structure panel shows and the 3D
      viewer loads the file over `/api/artifacts/...`. The classifier keeps
      only absolute paths below `/shared`. Check which tool results contain
      the structure paths, and in which form.
- [ ] `list_shared_assets` gives paths relative to `/shared`. The
      classifier now reads relative paths from `/shared` (tested), so these
      results must show structure panels.
- [ ] `score_molecules` and `predict_admet` results show a table and
      molecule drawings.
- [ ] `get_top_candidates` and `read_results` show a table.
- [ ] Files that the containers write to `data/shared` can be read by the
      `dd-web` process (owner and permissions).

### 6. Effects on the rest of the stack

- [ ] `dd-web` and `reinvent4-mcp` use the same GPU (`REINVENT_GPU`) and the
      same job, TTP and log directories. Find out if the two can run at the
      same time. If not, document that only one of them must run.
- [ ] Each browser tab gets its own agent, but all tabs share one `dd-mcp`
      process and one job store. Check that two tabs do not interfere.

## Known gaps (not regressions)

These features of the terminal agent are not in the web UI. They were not in
the Kubernetes version either.

- The Reviewer agent and `/review`. `REINVENT4_AGENT_MULTI_AGENT` has no
  effect on `dd-web`.
- Background job monitoring (`start_monitoring`). The UI can show
  `monitor_alert` and `monitor_response` events, but `dd-web` does not start
  the monitor.
- `/reset` and `/clear`. Reload the page to start a new conversation.
- `dd-coscientist` autonomous runs.
- An Apptainer/Singularity path. `services/singularity/` has no definition
  for the UI, and `dd-agent.def` does not install the `[web]` extra.
- A login. `dd-web` and the `ui` container have no authentication. See
  "Security" in `services/ui/README.md`. Decide whether a token is needed
  before the UI is used on nodes that are shared with other users.

## Agent fixes that only the Kubernetes version has

The Kubernetes version of the agent has four fixes that this repository does
not have. Without them, the UI can show long loops of failed tool calls and
answers that fail because the LLM context is full.

- Stop a turn after 3 failed tool calls with the same arguments
  (`MAX_IDENTICAL_FAILURES` in `reinvent4_agent/agent.py`).
- Cut one tool result at 12000 characters before it is added to the
  conversation (`MAX_TOOL_RESULT_CHARS` in `reinvent4_agent/agent.py`).
- List only the 40 newest paths per category in `list_shared_assets`
  (`_MAX_LISTED` in `reinvent4_mcp/tools/external.py`).
- Repair tool arguments that the LLM sends as encoded JSON text
  (`JsonArgument` in `reinvent4_mcp/server.py`).

The source is `solution-blueprints/cofolding-agent/packages/` in
silogen/blueprints. Port the tests with the fixes
(`test_repeated_tool_failure.py`, `test_context_overflow.py`,
`test_json_arguments.py`).

## Product decisions to confirm

- The page title is now "AI Cofolding Drug Design Agent". The Kubernetes
  version used "AMD Solution Blueprint — AI Cofolding Drug Design Agent".
  The top-level README calls this project "Drug Discovery Agentic
  Platform". Choose one name.
