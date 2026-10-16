# Drug Discovery Agentic Platform

An LLM-powered drug design assistant built on [REINVENT4](https://github.com/MolecularAI/REINVENT4). A CLI agent (`dd-agent`) connects to a Drug Discovery MCP server (`dd-mcp`) that wraps REINVENT4's scoring engine, exposing molecular scoring, RL optimization, and Therapeutic Target Profile (TTP) management as LLM-callable tools, plus five containerized scoring services for structure- and property-based evaluation.

By default the agent runs as **two collaborating agents** — a Driver that does the work and an independent read-only Reviewer that vets it. This can be toggled off to run a single agent.

The stack runs either under **Docker Compose** or, on clusters with no Docker daemon, under **Apptainer/Singularity** (AMD ROCm and NVIDIA CUDA image variants).

## Architecture

```
┌───────────────────────── dd-agent container ──────────────────────────┐
│                                                                       │
│    ┌───────────────┐   JSON handoff    ┌───────────────┐              │
│    │    Driver     │──────────────────►│   Reviewer    │  multi-agent │
│    │ full toolset  │◄──────────────────│   read-only   │  (toggleable)│
│    └───────┬───────┘  verdict/critique └───────┬───────┘              │
│            │                                   │                      │
│            └──────────────┬────────────────────┘                      │
│                           ▼  stdio / MCP                              │
│                    ┌─────────────┐                                    │
│                    │   dd-mcp    │  REINVENT4 scoring, TTPs, RL jobs  │
│                    └──────┬──────┘                                    │
└───────────────────────────┼───────────────────────────────────────────┘
                            │  POST /score  (HTTP)
      ┌──────────┬──────────┼───────────┬──────────────┐
      ▼          ▼          ▼           ▼              ▼
 ┌─────────┐ ┌─────────┐ ┌───────┐ ┌───────────┐ ┌─────────────┐
 │ Boltz2  │ │  Pose-  │ │ ADMET │ │ RetroDMTA │ │ Interaction │
 │  :8091  │ │ busters │ │ :8084 │ │   :8085   │ │  Profiler   │
 │   GPU   │ │  :8082  │ │  CPU  │ │    CPU    │ │    :8083    │
 └────┬────┘ └────┬────┘ └───────┘ └───────────┘ └──────┬──────┘
      │           │                                     │
      └───────────┴──────────────┬──────────────────────┘
                                 ▼
                       ┌───────────────────┐
                       │  /shared volume   │  structures, MSAs, poses
                       └───────────────────┘
```

The agent spawns the MCP server as a subprocess over stdio. The server imports REINVENT4 for native scoring (QED, molecular weight, LogP, TPSA, SA score, Tanimoto similarity, custom alerts, etc.) and manages TTPs — JSON documents that fully specify multi-property optimization objectives.

External scoring services run as separate containers, each exposing a uniform `POST /score` REST API. The MCP server calls them through the `ExternalRESTScorer` REINVENT4 plugin when a TTP includes external objectives. A `PipelineAwareScorer` handles artifact passing: Boltz2 writes predicted structures to the shared volume, and PoseBusters and the Interaction Profiler read them downstream.

## Current Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | MCP server skeleton + discovery + native scoring | Done |
| 2 | TTP schema, persistence, mapper, TTP-based scoring | Done |
| 3 | Boltz2 / PoseBusters external REST wrappers + pipeline | Done |
| 4 | RL optimization workflows (run/monitor/stop/resume) + transfer learning | Done |
| 5a | Interactive CLI agent | Done |
| 5b | System prompt with medchem context | Done |
| 6 | ADMET, RetroDMTA, and Interaction Profiler services | Done |
| 7 | Multi-agent Driver + Reviewer (handoff protocol, `/review`, campaign monitor) | Done |
| 8 | Apptainer/Singularity stack (AMD + NVIDIA) | Done |
| 9 | Per-molecule evaluation detail tools for the Reviewer (`get_evaluation_detail`, `read_structure_file`) | Planned |
| 10 | OpenTelemetry tracing of agent turns + tool calls | Planned |

## Prerequisites

- A Slurm cluster with AMD GPUs (ROCm) or NVIDIA GPUs (CUDA)
- **Either** Docker with Compose v2 on the compute nodes, **or** Apptainer/Singularity with `--fakeroot`
- An LLM API key, or any Anthropic / OpenAI-compatible endpoint (see [Configure credentials](#3-configure-credentials))

Everything runs inside containers. No local Python environment, venv, or pip install is needed.

## Repository Layout

```
dd_agentic/
├── reinvent4_mcp/                  # MCP server package (dd-mcp)
│   ├── server.py                   # FastMCP entry point (stdio transport), 24 tools
│   ├── config.py                   # Process config loading (reinvent4_mcp_config.yaml)
│   ├── utils.py                    # REINVENT4 path discovery, device detection
│   ├── scoring_pipeline.py         # PipelineAwareScorer (artifact passing between services)
│   ├── tools/
│   │   ├── discovery.py            # list_scoring_components / _available_models / _transforms
│   │   ├── scoring.py              # score_molecules (direct + TTP modes)
│   │   ├── optimization.py         # run_optimization, stop_optimization, run_transfer_learning
│   │   ├── results.py              # get_optimization_status, get_top_candidates, read_results
│   │   ├── external.py             # service health, list_shared_assets, predict_admet,
│   │   │                           #   predict_retrodmta, profile_interactions
│   │   ├── ttp.py                  # create/update/get/list/delete TTP
│   │   ├── uniprot.py              # fetch_target_info (UniProt sequence lookup)
│   │   └── pubchem.py              # fetch_compound (compound name -> real SMILES)
│   ├── _plugins/reinvent_plugins/components/
│   │   └── comp_external_rest.py   # ExternalRESTScorer plugin (HTTP -> ComponentResults)
│   ├── jobs/
│   │   ├── manager.py              # JobManager: subprocess tracking, CSV parsing, diagnostics
│   │   ├── toml_generator.py       # TOML generation (staged_learning + transfer_learning)
│   │   └── _run_worker.py          # Subprocess entry point (registers plugins, runs reinvent)
│   ├── ttp/
│   │   ├── schema.py               # Pydantic models + validation
│   │   ├── store.py                # JSON file persistence + version history
│   │   └── mapper.py               # TTP -> REINVENT4 scorer config translation
│   └── pyproject.toml
├── reinvent4_agent/                # CLI agent package (dd-agent, dd-coscientist, dd-web)
│   ├── cli.py                      # Entry points, arg parsing, interactive loop, /review
│   ├── agent.py                    # Agentic loop, tool scoping, background job monitoring
│   ├── coordinator.py              # Driver <-> Reviewer handoff protocol + campaign monitor
│   ├── system_prompt.py            # MEDCHEM_KNOWLEDGE + Driver / Reviewer system prompts
│   ├── mcp_client.py               # MCP stdio client wrapper
│   ├── display.py                  # Rich formatting with per-agent attribution
│   ├── web/                        # Web gateway (dd-web): HTTP + WebSocket for the UI
│   ├── backends/
│   │   ├── anthropic_backend.py
│   │   └── openai_backend.py
│   └── pyproject.toml
├── admet-predictor/                # ADMET consensus regression package (+ README, SERVICE.md)
├── retrodmta-predictor/            # RetroDMTA per-target potency/ADME package
├── interaction-profiler/           # Fragment taxonomy + protein-ligand interaction package
├── services/
│   ├── env.sh                      # Environment template -- copy to env.local.sh
│   ├── dd-agent/
│   │   ├── Dockerfile              # ROCm + REINVENT4 + dd-mcp + dd-agent
│   │   └── entrypoint.sh           # Timestamps captured (non-TTY) output
│   ├── boltz2/                     # Cofolding + affinity   (Dockerfile, app.py, requirements)
│   ├── posebusters/                # Pose validation        (Dockerfile, app.py, requirements)
│   ├── admet-predictor/            # ADMET REST wrapper     (Dockerfile, app.py)
│   ├── retrodmta-predictor/        # RetroDMTA REST wrapper (Dockerfile, app.py)
│   ├── interaction-profiler/       # Interaction wrapper    (Dockerfile, app.py)
│   ├── ui/                         # Web UI: React app + nginx (see services/ui/README.md)
│   ├── singularity/                # Apptainer stack -- no Docker daemon needed
│   │   ├── README.md               # Full Apptainer guide
│   │   ├── dd-agent.def            # GPU images, vendor-templated via build args
│   │   ├── boltz.def
│   │   ├── admet-predictor.def
│   │   ├── posebusters.def         # CPU images, vendor-neutral
│   │   ├── retrodmta-predictor.def
│   │   ├── interaction-profiler.def
│   │   ├── vendors.sh              # AMD/NVIDIA base image + torch index matrix
│   │   ├── build.sh                # apptainer build --fakeroot driver
│   │   ├── run.sh                  # start instances, health-gate, run agent, stop on exit
│   │   └── reinvent4_mcp_config.singularity.yaml   # localhost service URLs
│   └── slurm/
│       ├── run_interactive.sh      # Agent-only interactive Docker run
│       ├── agent.sbatch            # Batch job submission
│       └── setup.sh                # Bare-metal venv install (no containers)
├── docker-compose.yml              # 6-service compose stack (+ optional web UI)
├── reinvent4_mcp_config.yaml       # Process config (service URLs, RL defaults, storage paths)
├── TESTING.md                      # Manual and scripted test scenarios
├── tests/                          # ~316 unit tests (REINVENT4 imports mocked, no GPU needed)
│   ├── test_scoring.py             # Native + TTP scoring paths
│   ├── test_ttp.py                 # Schema, store, mapper
│   ├── test_optimization.py        # Job manager, TOML gen, diagnostics, transfer learning
│   ├── test_boltz_wrapper.py       # Boltz2 FastAPI app
│   ├── test_posebusters_wrapper.py # PoseBusters FastAPI app
│   ├── test_external_scoring.py    # ExternalRESTScorer + pipeline integration
│   ├── test_agent_tool_repair.py   # Orphaned tool-call repair in the agent loop
│   ├── test_web_artifacts.py       # Web UI tool-result classifier + artifact path safety
│   └── conftest.py
│
│  # --- Gitignored: restore manually before building ---
├── REINVENT4/                      # REINVENT4 source tree
├── boltz2/                         # Boltz2 source
├── posebusters/                    # PoseBusters source
├── models/                         # ADMET + RetroDMTA weights (from the model weights archive)
├── data/                           # Runtime state bind-mounted into the containers
│                                   #   shared/ ttps/ jobs/ logs/ tb_logs/ agent_logs/
└── services/env.local.sh           # Your keys and cluster-specific overrides
```

## Setup

### 1. Restore the gitignored dependencies

`REINVENT4/`, `boltz2/`, `posebusters/`, and `models/` are gitignored because they are large external artifacts. From the project root:

```bash
cd dd_agentic

git clone https://github.com/MolecularAI/REINVENT4.git   # native scoring + RL
git clone https://github.com/jwohlwend/boltz.git boltz2   # cofolding + affinity
git clone https://github.com/maabuu/posebusters.git       # pose validation
```

The ADMET and RetroDMTA services need trained model weights, distributed as a separate archive rather than in git. Unzip it so that `models/admet/` and `models/retrodmta/` exist — the Dockerfiles bake them into the images at build time:

```bash
unzip dd_agentic_models.zip          # provides ./models/
```

### 2. Sync the project to the cluster

`rsync` is delta-based, so repeated syncs only transfer changed bytes. The trailing `/` matters: `src/` copies contents, `src` copies the directory itself.

```bash
rsync -avz --exclude '.git' --exclude '__pycache__' \
    /local/path/to/dd_agentic/ cluster:~/dd_agentic/
```

For incremental syncs during development, target individual packages:

```bash
# MCP server only (after editing scoring/TTP/job code)
rsync -avz --exclude '__pycache__' \
    /local/path/to/dd_agentic/reinvent4_mcp/ cluster:~/dd_agentic/reinvent4_mcp/

# Agent only (after editing CLI/coordinator/prompt/display code)
rsync -avz --exclude '__pycache__' \
    /local/path/to/dd_agentic/reinvent4_agent/ cluster:~/dd_agentic/reinvent4_agent/
```

### 3. Configure credentials

All configuration lives in `services/env.local.sh`, a gitignored copy of the `services/env.sh` template. Every launch path — Docker Compose, Apptainer, and the Slurm scripts — reads it.

```bash
ssh cluster
cd ~/dd_agentic
cp services/env.sh services/env.local.sh
```

You must provide **exactly one** of two credentials, depending on which endpoint you use.

**An API-management LLM gateway** — set `LLM_GATEWAY_KEY`. It is sent as the `Ocp-Apim-Subscription-Key` header, together with the `user` and `anthropic-version` headers such gateways typically require:

```bash
export LLM_GATEWAY_KEY="your-gateway-subscription-key"
export REINVENT4_AGENT_BACKEND="anthropic"
export REINVENT4_AGENT_MODEL="claude-sonnet-4.6"
export REINVENT4_AGENT_BASE_URL="https://<your-gateway-host>/Anthropic"
```

When the gateway publishes OpenAI models as **Azure deployments**, point the base URL at the gateway root and use the deployment id as the model name. The backend appends `/openai/deployments/<model>` and sends `api-version=2024-12-01-preview` by default. Authentication uses the same `Ocp-Apim-Subscription-Key` and `user` headers as the Anthropic route:

```bash
export LLM_GATEWAY_KEY="your-gateway-subscription-key"
export REINVENT4_AGENT_BACKEND="openai"
export REINVENT4_AGENT_MODEL="GPT-5.4"            # Azure deployment id
export REINVENT4_AGENT_BASE_URL="https://<your-gateway-host>"
```

Override `REINVENT4_AGENT_API_VERSION` if a deployment needs a newer version. If your gateway also exposes the plain OpenAI-compatible route, name it in the base URL (`https://<your-gateway-host>/openai/v1`) and the backend uses the standard client instead.

**Any endpoint without gateway headers (Azure OpenAI, Anthropic direct, or another OpenAI-compatible gateway)** — leave `LLM_GATEWAY_KEY` empty and set `REINVENT4_AGENT_API_KEY` instead. This is the plain bearer/API key for the endpoint and is passed straight to the SDK; no gateway headers are added:

```bash
# Anthropic direct
export REINVENT4_AGENT_API_KEY="sk-ant-..."
export REINVENT4_AGENT_BACKEND="anthropic"
export REINVENT4_AGENT_MODEL="claude-sonnet-4.6"
unset REINVENT4_AGENT_BASE_URL                  # use the SDK default endpoint

# Azure OpenAI — the *.openai.azure.com host selects the deployments
# endpoint and the api-key auth header; set REINVENT4_AGENT_API_VERSION
# only to override the 2024-12-01-preview default.
export REINVENT4_AGENT_API_KEY="your-azure-api-key"
export REINVENT4_AGENT_BACKEND="openai"
export REINVENT4_AGENT_MODEL="gpt-4o"             # Azure deployment id
export REINVENT4_AGENT_BASE_URL="https://<resource>.openai.azure.com"
export REINVENT4_AGENT_NO_VERIFY_SSL="false"

# Plain OpenAI-compatible endpoint (OpenAI, vLLM, Ollama, LiteLLM, ...)
export REINVENT4_AGENT_API_KEY="sk-..."
export REINVENT4_AGENT_BACKEND="openai"
export REINVENT4_AGENT_MODEL="gpt-4o"
export REINVENT4_AGENT_BASE_URL="https://api.openai.com/v1"
unset REINVENT4_AGENT_API_VERSION                 # no api-version -> plain client
```

A private LLM gateway that speaks the Azure dialect but is not hosted on `*.openai.azure.com` is the last case above: leave `REINVENT4_AGENT_API_VERSION` unset so the base URL is used verbatim, and authenticate with `REINVENT4_AGENT_API_KEY`. Set `REINVENT4_AGENT_API_VERSION` only if that gateway addresses models as `<base-url>/openai/deployments/<model>`.

On **reasoning models** (`gpt-5`, `o1`, `o3`, `o4`) the response cap is sent as `max_completion_tokens` and bounds the model's hidden reasoning tokens *as well as* the reply. A cap sized for a chat model can therefore be spent entirely on reasoning, which truncates the turn before any visible token is produced — the symptom is a reply that never arrives while the output-token count keeps climbing, and it gets more likely the longer the conversation grows.

`env.sh` therefore defaults `REINVENT4_AGENT_MAX_TOKENS` to the model's full output limit when the model name looks like a reasoning model — 128000 for `gpt-5`, 100000 for the o-series — and 4096 otherwise, since other models reject a cap above their own limit. The cap only bounds a single response; it is not reserved or billed, so there is no cost to leaving it at the ceiling. The backend independently floors the cap at 32768 on these models, so a stale 4096 in an existing `env.local.sh` cannot silence the agent either.

```bash
export REINVENT4_AGENT_MODEL="gpt-5"
export REINVENT4_AGENT_MAX_TOKENS="128000"      # env.sh already does this
export REINVENT4_AGENT_REASONING_EFFORT="low"   # optional: less reasoning per turn
```

If neither key is set, `env.local.sh` aborts with an error rather than starting a session that cannot authenticate. As a last resort the agent also falls back to the SDK-standard `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` variables, but those are not forwarded into the containers — inside Docker or Apptainer, use `REINVENT4_AGENT_API_KEY`.

Remaining options worth setting in `env.local.sh`:

```bash
export REINVENT4_PROJECT_DIR="$HOME/dd_agentic"   # default
export REINVENT4_AGENT_MULTI_AGENT="true"         # Driver + Reviewer; "false" for Driver only
export REINVENT4_AGENT_THINKING_BUDGET="1024"     # extended thinking (Anthropic)
export SLURM_PARTITION="batch"                    # your cluster partition
```

> `env.local.sh` is gitignored, so secrets stay out of version control.

The compute node must be able to reach the LLM endpoint directly.

---

## Running with Docker Compose

Docker is typically only available on compute nodes, so building images, starting services, and running the agent all happen inside an allocation.

### Get an allocation and shell onto the compute node

```bash
salloc --partition=batch --gres=gpu:2 --cpus-per-task=8 --mem=64G --time=4:00:00
srun --jobid=$SLURM_JOB_ID --pty bash
cd ~/dd_agentic
source services/env.local.sh
```

`salloc` reserves resources but usually keeps you on the login node; `srun --pty bash` gives you a shell on the compute node where Docker and the GPUs actually are.

Two GPUs are requested because REINVENT and Boltz2 run on separate devices. `env.local.sh` splits `SLURM_STEP_GPUS` into `REINVENT_GPU` (first) and `BOLTZ_GPU` (second). Override explicitly if needed:

```bash
export REINVENT_GPU=2  BOLTZ_GPU=3
```

### Build the images

```bash
docker compose build
```

| Image | Base | Contents |
|-------|------|----------|
| `reinvent4-mcp` | `rocm/pytorch:rocm7.0_ubuntu24.04_py3.12` | REINVENT4 + priors + dd-mcp + dd-agent |
| `boltz` | `rocm/pytorch:rocm7.0_ubuntu24.04_py3.12` | Boltz2 from source + FastAPI wrapper |
| `posebusters` | `python:3.11-slim` | PoseBusters + FastAPI wrapper |
| `admet-predictor` | `python:3.12-slim` | admet-predictor + baked `models/admet` |
| `retrodmta-predictor` | `python:3.12-slim` | retrodmta-predictor + baked `models/retrodmta` |
| `interaction-profiler` | `mambaorg/micromamba:1.5.8` | RDKit, OpenMM, PDBFixer, MDAnalysis, AccFG |

Each Dockerfile is self-contained. The first build pulls base images and compiles everything, which takes a while; later builds reuse Docker's layer cache.

> **Multi-node note**: images are local to a node. If you `salloc` onto a different node, rebuild there, or push to a registry / move them with `docker save` and `docker load` over shared storage.

### Start the full stack

```bash
docker compose run --rm reinvent4-mcp
```

This starts the five scoring services, waits for all of their health checks to pass, and then drops you into the agent chat with an interactive terminal. When you exit the chat the scoring services keep running.

The container reaches the LLM endpoint directly by DNS on the Compose bridge network.

```bash
docker compose down          # stop all containers
docker compose down -v       # also remove the boltz_cache volume
```

### Step-by-step alternative

Useful when debugging startup:

```bash
docker compose up -d boltz posebusters admet-predictor \
                     retrodmta-predictor interaction-profiler
docker compose ps                              # watch health status
curl http://localhost:8091/health              # Boltz2
curl http://localhost:8082/health              # PoseBusters
curl http://localhost:8083/health              # Interaction Profiler
curl http://localhost:8084/health              # ADMET
curl http://localhost:8085/health              # RetroDMTA
docker compose run --rm reinvent4-mcp          # then start the agent
```

Boltz2 can take up to 60 seconds to become healthy (model loading); the others are usually ready within 10 seconds. From inside the agent you can confirm the same thing by asking it to check `list_external_services`.

### Agent-only mode

If you only need native REINVENT4 scoring (QED, SA score, etc.), skip Compose and run the agent container alone on a single GPU:

```bash
salloc --partition=batch --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=4:00:00
srun --jobid=$SLURM_JOB_ID --pty bash
cd ~/dd_agentic && source services/env.local.sh
bash services/slurm/run_interactive.sh
```

TTP scoring automatically falls back to native-only components when external services are unreachable.

### Volumes and persistence

Everything the stack writes lands under `${DATA_DIR:-./data}` on the host, so state survives container restarts:

| Host path | Container path | Contents |
|-----------|----------------|----------|
| `data/shared` | `/shared` (all services) | Predicted structures, MSAs, poses |
| `data/ttps` | `/root/.reinvent4_mcp/ttps` | TTP definitions + version history |
| `data/jobs` | `/data/reinvent4_mcp/jobs` | RL job working dirs, configs, result CSVs |
| `data/logs` | `/data/reinvent4_mcp/logs` | MCP server logs |
| `data/tb_logs` | `/data/reinvent4_mcp/tb_logs` | TensorBoard logs |
| `data/agent_logs` | `/logs` | Agent session `.log` + `.jsonl` transcripts |
| `boltz_cache` (named volume) | `/cache` | Boltz2 model weights |

### Web UI

A browser UI is available as an option. It shows the chat next to a 3D
viewer for cofolded structures, 2D drawings of molecules and sortable tables
of predictions. It runs in two more containers, `dd-web` (the agent behind
HTTP and WebSocket) and `ui` (nginx), in the `ui` Compose profile:

```bash
docker compose --profile ui up -d dd-web ui
ssh -J <login-node> -L 8080:localhost:8080 <compute-node>   # on your machine
```

Then open <http://localhost:8080>. See [services/ui/README.md](services/ui/README.md)
for the details and for UI development without a GPU.

---

## Running with Apptainer/Singularity

On clusters with no Docker daemon, the same six services run as Apptainer instances. The Compose setup is untouched — the Apptainer stack is entirely additive, with separate image variants for AMD (ROCm) and NVIDIA (CUDA).

Key differences from Compose:

| Compose | Apptainer equivalent |
|---------|----------------------|
| `build:` from Dockerfiles | `.def` files, `apptainer build --fakeroot` |
| bridge DNS `http://boltz:8080` | host networking + `http://localhost:<port>` |
| `depends_on` + health checks | `apptainer instance start` + `/health` polling in `run.sh` |
| named volumes / mounts | `--bind` (all services share one `/shared`) |
| `runtime: amd` | `--rocm` (AMD) or `--nv` (NVIDIA), selected by `--vendor` |

### Build

Requires `apptainer` (or legacy `singularity`) with `--fakeroot`; on HPC this is usually `module load apptainer`. Both scripts auto-detect either name, or set `APPTAINER_BIN=/path/to/apptainer`.

```bash
bash services/singularity/build.sh nvidia            # all NVIDIA images
bash services/singularity/build.sh amd               # all AMD images
bash services/singularity/build.sh amd boltz         # a single service
bash services/singularity/build.sh nvidia --dry-run  # print commands only
```

Base images and torch wheel indexes live in `services/singularity/vendors.sh`. GPU builds must run on (or target) matching hardware; use `--dry-run` to preview the resolved commands anywhere. ADMET and RetroDMTA build CPU-only for both vendors, matching their Compose behaviour.

### Run

Source `env.local.sh` first — `run.sh` forwards `LLM_GATEWAY_KEY`, `REINVENT4_AGENT_API_KEY`, and the `REINVENT4_AGENT_*` variables into the container, so the same file configures both launch paths.

```bash
source services/env.local.sh
bash services/singularity/run.sh --vendor amd        # or --vendor nvidia
```

`run.sh` starts the five service instances (with binds, per-instance GPU visibility, and an env-driven `PORT` so the three services that internally listen on `:8080` do not collide under host networking), polls each `/health`, then runs the agent. All instances are stopped on exit via an EXIT/INT/TERM trap.

| Flag | Purpose |
|------|---------|
| `--vendor <amd\|nvidia>` | Required. Selects `--rocm`/`--nv` and the `*-<vendor>.sif` images |
| `--data-dir <path>` | Host dir to bind (default `./data`) |
| `--reinvent-gpu` / `--boltz-gpu` | GPU indices (default `0` and `1`) |
| `--skip-agent` | Start and health-gate the services only |
| `--dry-run` | Print the `apptainer` commands without running them |

Other modes work the same as under Compose:

```bash
# batch mode: pipe a prompt
echo "Score aspirin CC(=O)OC1=CC=CC=C1C(=O)O for drug-likeness" \
  | bash services/singularity/run.sh --vendor amd

# services only, for debugging
bash services/singularity/run.sh --vendor amd --skip-agent
```

The agent uses `services/singularity/reinvent4_mcp_config.singularity.yaml`, a copy of the root config with every `service_url` pointed at `http://localhost:<port>/score`. `run.sh` binds it over the baked copy, so config edits take effect without rebuilding the image. The root `reinvent4_mcp_config.yaml` is never modified.

Since SIF images are read-only, `run.sh` binds writable host directories under `--data-dir` for `/shared`, the job/log/tb dirs, TTP storage, and `/logs`, creating them if missing.

For the full Apptainer guide and validation steps, see [services/singularity/README.md](services/singularity/README.md).

---

## Usage

Once the agent is running you interact with it in a chat interface:

```
You> What scoring components are available?

Agent> [calls list_scoring_components]
There are 45 scoring components available. The most commonly used are:
- QED (drug-likeness)
- MolecularWeight, SlogP, TPSA (physicochemical)
- HBondDonors, HBondAcceptors, NumRotBond (Lipinski)
- SAScore (synthetic accessibility)
- TanimotoSimilarity (structural similarity)
- CustomAlerts (SMARTS-based filters)
...

You> Score these for drug-likeness: CCO, c1ccccc1, CC(=O)OC1=CC=CC=C1C(=O)O

Agent> [calls score_molecules with QED]
┌──────────────┬───────┐
│ SMILES       │ QED   │
├──────────────┼───────┤
│ CCO          │ 0.41  │
│ c1ccccc1     │ 0.34  │
│ CC(=O)OC1... │ 0.55  │
└──────────────┴───────┘
```

### Agent Commands

| Command | Description |
|---------|-------------|
| `/help`, `/?` | Show available commands |
| `/review [job_id]` | Summon the read-only Reviewer to critique the latest (or given) candidates — multi-agent mode only |
| `/reset`, `/clear` | Clear conversation history (both agents in multi-agent mode) |
| `/exit`, `/quit` | Exit the agent |

### CLI Arguments

Every flag has an environment-variable equivalent; CLI flags take precedence.

```bash
dd-agent [OPTIONS]
```

**LLM backend**

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | `anthropic` | `anthropic` or `openai` |
| `--model` | `claude-sonnet-4` | Model name as the endpoint knows it |
| `--base-url` | — | Base URL of the LLM endpoint (omit for the SDK default) |
| `--api-key` | — | API key for a non-gateway endpoint |
| `--gateway-key` | — | LLM gateway subscription key (`Ocp-Apim-Subscription-Key`) |
| `--api-version` | `2024-12-01-preview` on Azure routes | Azure `api-version` (OpenAI only). A gateway key or an `*.openai.azure.com` host selects the `<base-url>/openai/deployments/<model>` route; override only if a deployment needs a newer version |
| `--no-verify-ssl` | `false` | Disable TLS certificate verification |
| `--thinking-budget` | — | Extended-thinking token budget (Anthropic only; omit to disable) |
| `--reasoning-effort` | — | `low`, `medium`, or `high` for o-series deployments (OpenAI only) |
| `--max-tokens` | `4096` | Maximum tokens per LLM response. On reasoning models (`gpt-5`, o-series) the cap also covers hidden reasoning tokens and is raised to a floor of 32768 |

**Multi-agent**

| Flag | Default | Description |
|------|---------|-------------|
| `--multi-agent` | on | Run the read-only Reviewer alongside the Driver |
| `--no-multi-agent` | — | Run the Driver only |
| `--goal` | — | Design goal for a `dd-coscientist` run (prompted if omitted) |
| `--max-turns` | `8` | Cap on combined Driver + Reviewer turns per co-scientist run |
| `--token-budget` | unlimited | Combined token budget across both agents |

**Monitoring, MCP, and logging**

| Flag | Default | Description |
|------|---------|-------------|
| `--monitor-interval` | `300` | Seconds between background monitoring polls |
| `--no-monitor` | `false` | Disable autonomous background monitoring |
| `--server-command` | `dd-mcp` | Command used to launch the MCP server subprocess |
| `--server-args` | — | Extra arguments forwarded to the MCP server |
| `--log-dir` | — | Directory for session `.log` and `.jsonl` transcripts |
| `--verbose`, `-v` | `false` | DEBUG-level logging on stderr |
| `--version` | — | Show version and exit |

**Examples**

```bash
# Single-agent mode with an OpenAI-compatible endpoint
dd-agent --no-multi-agent --backend openai --model gpt-4o \
         --base-url https://my-endpoint.example.com/v1

# OpenAI model on a gateway (Azure deployments route)
dd-agent --backend openai --model GPT-5.4 \
         --base-url https://<your-gateway-host> \
         --gateway-key "$LLM_GATEWAY_KEY" --no-verify-ssl

# Anthropic with extended thinking and a larger response budget
dd-agent --thinking-budget 10000 --max-tokens 8192

# Autonomous two-agent run with a hard budget
dd-coscientist --goal "Design selective JAK2 inhibitors with good oral bioavailability" \
               --max-turns 12 --token-budget 400000
```

---

## Multi-Agent Mode (Driver + Reviewer)

The platform runs two agents that share one MCP session but keep completely independent conversation state, system prompts, and token accounting.

| | **Driver** | **Reviewer** |
|---|---|---|
| Role | Medicinal chemist doing the forward workflow | Skeptical gatekeeper and critic |
| Tools | All 24 MCP tools | 10 read-only tools |
| Mutates state | Yes | No — never |
| Prompt | `DRIVER_SYSTEM_PROMPT` | `REVIEWER_SYSTEM_PROMPT` |

Independence is the point: a Reviewer with a fresh context resists rationalizing a molecule the Driver just produced. The read-only guarantee is structural, not just prompted — the Reviewer's tool list is filtered to an allowlist before it ever reaches the LLM, so it physically cannot call `create_ttp`, `run_optimization`, `stop_optimization`, or any other mutating tool:

```
read_results             get_top_candidates       get_ttp
list_ttps                get_optimization_status  list_jobs
list_scoring_components  list_available_models    list_transforms
list_external_services
```

### Toggling it on and off

Multi-agent mode is **on by default**. Three ways to control it, in increasing precedence:

```bash
# 1. env.local.sh (applies to every launch path)
export REINVENT4_AGENT_MULTI_AGENT="false"

# 2. docker-compose.yml reads the same variable
REINVENT4_AGENT_MULTI_AGENT=false docker compose run --rm reinvent4-mcp

# 3. CLI flag, overriding both
dd-agent --no-multi-agent
dd-agent --multi-agent
```

The agent reports which mode it started in:

```
Multi-agent mode: Driver + read-only Reviewer (/review enabled).
Single-agent mode: Driver only (Reviewer off).
```

In single-agent mode the Reviewer is never constructed, `/review` returns a hint instead of running, and no Reviewer monitor starts. The Driver's own job-alert monitor still runs, and no other behaviour changes — so turning multi-agent off is the straightforward way to roughly halve token spend on routine work.

### How the two agents exchange work

Handoffs are typed payloads rather than free-form chat: each agent appends a fenced `json` block that the coordinator parses, which keeps the exchange cheap and machine-checkable.

The Driver hands off a batch:

```json
{
  "candidates": ["<SMILES>", "..."],
  "job_id": "<optimization job id or null>",
  "csv_path": "<path to the results CSV or null>",
  "rationale": "<why these candidates, in 1-3 sentences>",
  "ready_for_review": true
}
```

The Reviewer returns a verdict:

```json
{
  "verdict": "approve | revise | reject",
  "ranking": ["<SMILES best-first>", "..."],
  "liabilities": ["<concise liability>", "..."],
  "requested_changes": ["<specific, actionable change for the Driver>", "..."],
  "rationale": "<1-3 sentence justification grounded in the evidence>"
}
```

On `approve` the run finalizes; on `reject` it stops; on `revise` the requested changes are fed back to the Driver for another round. An unparseable verdict degrades to `revise` with the raw critique forwarded, so a malformed block never crashes the loop.

Before judging, the Reviewer is required to pull the real scored rows itself with its read-only tools rather than trusting the Driver's summary, and to apply a hard **Boltz2 confidence gate** — `confidence_score > 0.6`, `ligand_iptm > 0.5`, and PoseBusters `all_pass == True`. If any of the three fails, the predicted affinity is not credited. It approves only when at least one candidate clears the gate with no disqualifying medchem liability.

### On-demand review

In an interactive session, `/review` summons the Reviewer at any point:

```
You> /review
You> /review abc123          # focus a specific optimization job
```

Without a job id the Reviewer finds the most recent optimization itself. The review runs under the Driver's chat lock, so it never interleaves with the Driver's turns or the background monitors on the shared MCP session.

### Background campaign monitoring

Two independent monitors run during a session, both disabled by `--no-monitor`:

- **Job-alert monitor** (always on): the Driver polls `get_job_alerts` every `--monitor-interval` seconds and, on a new alert, is prompted with `[MONITOR] Alert (severity): ...` so it can investigate and act. Alerts are deduplicated by `(job_id, type)`, and the actions taken are appended to the job's `monitor_actions.jsonl`.
- **Reviewer monitor** (multi-agent only): every `--monitor-interval` seconds, if an optimization is running, the Reviewer independently assesses its trajectory, convergence status, and per-component health, then returns a verdict that the Driver acts on — `approve` to let it continue, `revise` to apply concrete changes, `reject` to stop the job now.

This is what closes the loop on a long RL campaign: the Reviewer notices a plateau or a failing component, and the Driver stops the run, adjusts the TTP, and resumes from the checkpoint without the user intervening.

### Autonomous co-scientist runs

`dd-coscientist` runs the same two agents non-interactively against a single goal, looping Driver → Reviewer → Driver until approval, rejection, or budget exhaustion, then entering campaign monitoring:

```bash
dd-coscientist --goal "Design selective JAK2 inhibitors with good oral bioavailability"
```

A global turn and token budget guarantees termination. The terminal shows per-agent turn banners (Driver in cyan, Reviewer in magenta, coordinator in blue) and a running `turns n/max | tokens n/budget` counter. Each agent writes its own JSONL transcript (`session_<timestamp>_<name>.jsonl`) so the two streams stay separable after the fact.

---

## MCP Tools

24 tools are exposed to the Driver. The 10 marked read-only are also available to the Reviewer.

| Tool | Read-only | Description |
|------|:---------:|-------------|
| `list_scoring_components` | ✓ | Discover all native REINVENT4 scoring components |
| `list_available_models` | ✓ | Check which generative model priors are available |
| `list_transforms` | ✓ | List score transform types (sigmoid, step, etc.) |
| `list_external_services` | ✓ | Health-check the five external scoring services |
| `list_shared_assets` | | List real files on `/shared` (structures, MSAs, cofold outputs) |
| `fetch_target_info` | | Resolve a UniProt accession or free text to a protein sequence |
| `fetch_compound` | | Resolve a compound name to a real SMILES via PubChem |
| `score_molecules` | | Score SMILES with components directly or via a TTP |
| `predict_admet` | | ADMET consensus predictions with applicability-domain flags |
| `predict_retrodmta` | | Per-target potency / ADME predictions |
| `profile_interactions` | | Fragment taxonomy and protein-ligand interaction profiling |
| `create_ttp` | | Create a Therapeutic Target Profile |
| `update_ttp` | | Update a TTP (auto-increments version) |
| `get_ttp` | ✓ | Retrieve a TTP by ID, optionally at a specific version |
| `list_ttps` | ✓ | List all stored TTPs |
| `delete_ttp` | | Delete a TTP and its version history |
| `run_optimization` | | Launch an RL campaign with full TTP scoring |
| `get_optimization_status` | ✓ | Trajectories, per-component stats, convergence diagnostics |
| `stop_optimization` | | Stop a running job and save a checkpoint |
| `run_transfer_learning` | | Fine-tune a generative model on a SMILES dataset |
| `get_top_candidates` | ✓ | Extract the best molecules from a run |
| `read_results` | ✓ | Parse a REINVENT4 output CSV into summary statistics |
| `list_jobs` | ✓ | List all tracked jobs and their states |
| `get_job_alerts` | | Check running jobs for problems needing intervention |

---

## External Scoring Services

Five services wrap domain-specific tools behind a uniform REST API.

| Service | Purpose | GPU | Host port |
|---------|---------|:---:|-----------|
| **Boltz2** | Protein-ligand cofolding + binding affinity prediction | Yes | 8091 |
| **PoseBusters** | Geometric and chemical quality checks on predicted poses | No | 8082 |
| **Interaction Profiler** | Fragment taxonomy + typed protein-ligand interactions | No | 8083 |
| **ADMET Predictor** | Consensus ADMET regression with applicability domain | No | 8084 |
| **RetroDMTA Predictor** | Per-target, per-window potency and ADME endpoints | No | 8085 |

### REST API contract

Every service implements the same interface. `GET /health` returns `{"status": "ok"}` when ready, and `POST /score` takes:

```json
{ "smiles": ["CCO", "c1ccccc1"], "params": { } }
```

returning:

```json
{ "scores": [0.85, 0.42], "metadata": { }, "artifacts": { } }
```

- `scores` — one float per SMILES (`null` on per-molecule failure, converted to `NaN` by the scorer plugin).
- `metadata` — per-molecule detail (sub-scores, confidence, per-check booleans), as lists aligned to `smiles`.
- `artifacts` — file references produced by this service, consumed by downstream services.

ADMET, RetroDMTA, and the Interaction Profiler additionally expose richer out-of-band endpoints (`/predict`, `/predict_csv`, `/targets`, `/analyze`, `/profile`, `/schema`) that the agent calls directly through `predict_admet`, `predict_retrodmta`, and `profile_interactions`.

### How the scoring pipeline fits together

When a TTP includes external objectives:

1. The agent calls `score_molecules` with a TTP that has `type: "external"` objectives.
2. The MCP server health-checks the required services via `GET /health`.
3. If all are healthy a `PipelineAwareScorer` is created; if any is unreachable, scoring falls back to native-only components with a warning noting which objectives were skipped.
4. The TTP mapper topologically sorts objectives so upstream services run first.
5. Boltz2 predicts complex structures and affinities, writing structure files to `/shared`.
6. The pipeline scorer reads those artifacts and injects `structure_paths` into the PoseBusters and Interaction Profiler requests.
7. Downstream services validate poses and count interactions on the real structures.
8. All native and external scores are aggregated into one total score per molecule.

### Boltz2

Wraps the `boltz predict` CLI. For each batch it generates one Boltz YAML per molecule, runs prediction as a subprocess, then parses affinity JSONs, confidence JSONs, and structure files from the output directory.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `protein_sequence` | (required) | Full amino-acid sequence of the target |
| `msa_path` | `null` | Path to a pre-computed `.a3m` MSA |
| `use_msa_server` | `false` | Generate the MSA via the mmseqs2 server (cached after the first call) |
| `msa_server_url` | `https://api.colabfold.com` | MSA server URL (only with `use_msa_server`) |
| `affinity_mode` | `"pred_value"` | `"pred_value"` (log10 IC50 µM) or `"probability_binary"` (0-1) |
| `recycling_steps` | `3` | Structure prediction recycling iterations |
| `diffusion_samples` | `1` | Structure samples per molecule |

One of `msa_path` or `use_msa_server: true` is required; the wrapper returns HTTP 400 if neither is set.

**MSA resolution and caching.** An explicit `msa_path` is read directly and the server is never contacted. With `use_msa_server: true`, the wrapper queries the MSA server exactly once for a given protein sequence, writes the alignment to `/shared/msa_cache/<sha1>.a3m`, and every later RL step reads from that file. The cache key is a hash of the sequence, so distinct targets cache independently; if the one-time query fails, the wrapper lets Boltz query the server itself so the run is not lost. RL loops are therefore efficient out of the box.

### PoseBusters

Validates predicted ligand poses against geometric and chemical checks (bond lengths and angles, steric clashes, ring flatness, protein overlap), returning the fraction of checks passed, with per-check booleans in metadata.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `"dock"` | `dock`, `redock`, `mol`, or `gen` |
| `protein_pdb_path` | (required for `dock`) | Protein structure PDB on the shared volume |
| `structure_paths` | (required) | Per-molecule structure paths, injected from Boltz2 artifacts |

### Interaction Profiler

Builds a hierarchical medchem fragment taxonomy from SMILES, and — when a structure is supplied — detects typed protein-ligand interactions (H-bonds, hydrophobic contacts, π-stacking, ionic, halogen, metal) grouped into `(fragment, residue, interaction)` triples. `POST /score` returns the interaction count for a chosen family, so it can be chained after a structure producer as an RL objective.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `interaction_type` | `"all"` | Family to count (`hbond`, `hydrophobic`, `ionic`, `pi_stacking`, …, or `all`) |
| `structure_paths` | `null` | Per-SMILES structure paths, injected from an upstream cofold |
| `ph` | `7.4` | Protonation pH |
| `radius` | `8.0` | Pocket crop radius (Å) |

### ADMET Predictor

Consensus regression over XGBoost, Chemprop-MTL, and CheMeleon-MTL for seven endpoints — `logp`, `logd`, `aqueous_solubility`, `kinetic_solubility`, `caco2_permeability`, `efflux_mdck`, `efflux_caco2` — reported as mean ± std on standardized SMILES, with a double applicability-domain flag. `POST /score` takes `endpoint` (default `"logp"`) and returns the ensemble mean per molecule, with `std` and `ad_pass` in metadata. Weights come from `models/admet/`.

### RetroDMTA Predictor

Per-target, per-temporal-window project-endpoint predictions using the same three-model ensemble. Targets are FXA (pKi, logD, logS), MMP8 (pIC50), PPARD (pEC50), and RENIN (pIC50); `GET /targets` lists them.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target` | (required) | `FXA`, `MMP8`, `PPARD`, or `RENIN` |
| `endpoint` | first endpoint of the target | e.g. `FXA_pKi`; required when a target has several |
| `window` | `"latest"` | Temporal window index, or `"latest"` |
| `method` | `"ensemble"` | `ensemble`, `xgboost`, `chemprop`, or `chemeleon` |

Weights come from `models/retrodmta/`. Unlike ADMET, no applicability domain is reported.

### Service configuration

Service URLs and parameters live in `reinvent4_mcp_config.yaml` under `scoring`:

```yaml
scoring:
  aggregation: "geometric_mean"
  boltz:
    enabled: true
    service_url: "http://boltz:8080/score"
    recycling_steps: 3
    diffusion_samples: 1
    affinity_mode: "pred_value"
    use_msa_server: false
  posebusters:
    enabled: true
    service_url: "http://posebusters:8080/score"
    mode: "dock"
  admet:
    enabled: true
    service_url: "http://admet-predictor:8084/score"
  retrodmta:
    enabled: true
    service_url: "http://retrodmta-predictor:8085/score"
    method: "ensemble"
    window: "latest"
  interaction_profiler:
    enabled: true
    service_url: "http://interaction-profiler:8080/score"
    interaction_type: "all"
    radius: 8.0
    ph: 7.4
```

Set `enabled: false` to disable a service entirely. Under Compose these URLs use service names resolved by Docker's internal DNS; under Apptainer the equivalent `reinvent4_mcp_config.singularity.yaml` points every URL at `http://localhost:<port>`.

### Defining external objectives in a TTP

Add `type: "external"` objectives. A service that consumes upstream artifacts declares `depends_on` naming the objective that produces them, so the pipeline scorer injects structure paths automatically:

```json
{
  "id": "jak2-structure",
  "name": "JAK2 Structure-Based Optimization",
  "target": {
    "protein_sequence": "MVLSPADKTN...",
    "protein_pdb_path": "/shared/protein.pdb"
  },
  "scoring": {
    "objectives": [
      {
        "name": "Boltz Affinity",
        "type": "external",
        "service": "boltz",
        "service_url": "http://boltz:8080/score",
        "weight": 1.0
      },
      {
        "name": "Pose Quality",
        "type": "external",
        "service": "posebusters",
        "service_url": "http://posebusters:8080/score",
        "depends_on": "Boltz Affinity",
        "weight": 0.5
      },
      {
        "name": "Drug-likeness",
        "type": "native",
        "component": "QED",
        "weight": 0.6
      }
    ]
  }
}
```

---

## RL Optimization Workflows

The agent defines a TTP, launches an RL campaign, monitors convergence by polling score trajectories, and intervenes when the run stalls or diverges.

1. `run_optimization` is called with a TTP and RL parameters (model, max_steps, batch_size).
2. The MCP server composes a Scorer from the TTP, generates a REINVENT4 TOML config, and launches a background subprocess.
3. REINVENT4 runs the RL loop: generate molecules, score them all, compute the RL loss, update the model, repeat.
4. `get_optimization_status` returns the score trajectory, per-component statistics, and automated convergence diagnostics.
5. On a plateau, divergence, or component failure, the agent calls `stop_optimization` (saving a checkpoint), adjusts the TTP via `update_ttp`, and resumes with `run_optimization(resume_from=...)`.
6. On completion, `get_top_candidates` extracts the best molecules ranked by total score with full component breakdowns.

### Convergence diagnostics

`get_optimization_status` returns diagnostics in `trajectory.diagnostics`:

| Field | Description |
|-------|-------------|
| `convergence_status` | `early`, `improving`, `plateau`, `diverging`, `stable`, or `completed` |
| `plateau_detected` | Scores stopped improving (range below threshold in the trailing window) |
| `diverging` | Recent scores declining below the mid-run mean |
| `improvement_rate` | Linear regression slope over the recent window |
| `trend` | Early/mid/recent mean scores and overall improvement |
| `component_health` | Per-component NaN rates and status (`healthy`, `degraded`, `failing`) |
| `warnings` | Actionable alerts with type, severity, and message |
| `recommendations` | Suggested next actions |

### Example workflow

```
Goal: "Design JAK2 inhibitors with good oral bioavailability"

1. create_ttp(...)          -> QED, Lipinski, Boltz affinity, PoseBusters, SA
2. run_optimization(...)    -> job_id="abc123"
3. get_optimization_status  -> step=100, mean_score=0.35, convergence="improving"
4. get_optimization_status  -> step=150, mean_score=0.38, convergence="plateau"
   warnings: "Score has plateaued since step 130"
   recommendations: "Stop and adjust TTP weights"
5. stop_optimization        -> checkpoint saved
6. update_ttp(lower the SA weight from 0.15 to 0.05)
7. run_optimization(resume_from="checkpoint.chkpt")
8. get_top_candidates(top_n=50)
```

In multi-agent mode steps 3-6 also happen without prompting: the Reviewer monitor spots the plateau and the Driver acts on its verdict.

### Background job management

Jobs run as background subprocesses of the MCP server. The `JobManager` tracks job states (`queued` → `running` → `completed`/`failed`/`stopped`), subprocess lifecycle (PID tracking, graceful SIGINT for checkpoint saves, SIGTERM fallback), and metadata as JSON files in each job's working directory, so jobs survive a server restart. Concurrency is capped by `jobs.max_concurrent_jobs` in the process config.

### Transfer learning

Transfer learning fine-tunes a generative model on a focused SMILES dataset before RL, narrowing the chemical space the model explores:

```
1. run_transfer_learning(model=".reinvent", smiles=[...], num_epochs=50)
2. get_optimization_status(job_id)          # monitor progress
3. run_optimization(model="<output_model_path>", ttp_id="...")
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `.reinvent` | Prior to fine-tune (registry key or file path) |
| `smiles` / `smiles_file` | (required) | Training SMILES, inline or as a `.smi` path |
| `num_epochs` | `50` | Training epochs |
| `batch_size` | `128` | Training batch size |
| `save_every_n_epochs` | `5` | Checkpoint frequency |
| `validation_smiles` / `validation_smiles_file` | — | Optional validation data |
| `output_model_file` | `tl_model.model` | Base name for the output model |

Epoch checkpoints are saved as `<output_model_file>.<epoch>.chkpt` in the job working directory.

---

## Testing

The ~316 unit tests mock REINVENT4 imports, so no GPU or REINVENT4 installation is required:

```bash
cd examples/dd_agentic
python -m pytest tests/ -v
```

To run them inside the agent container, which has every dependency:

```bash
docker compose run --rm --no-deps reinvent4-mcp python -m pytest tests/ -v
```

See [TESTING.md](TESTING.md) for end-to-end scenarios that exercise the live services.

---

## Environment Variables Reference

All of these are set in `services/env.local.sh` and forwarded into the containers by `docker-compose.yml` and `services/singularity/run.sh`.

**Credentials** — set exactly one of the first two.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_GATEWAY_KEY` | — | LLM gateway subscription key (`Ocp-Apim-Subscription-Key`) |
| `REINVENT4_AGENT_API_KEY` | — | Direct API key for an endpoint without gateway headers (Azure, Anthropic, …) |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | — | SDK-standard fallbacks; not forwarded into containers |

**LLM backend**

| Variable | Default | Description |
|----------|---------|-------------|
| `REINVENT4_AGENT_BACKEND` | `anthropic` | `anthropic` or `openai` |
| `REINVENT4_AGENT_MODEL` | `claude-sonnet-4.6` | Model name |
| `REINVENT4_AGENT_BASE_URL` | — (SDK default) | LLM API base URL, e.g. `https://<your-gateway-host>/Anthropic`; use the gateway root for the OpenAI backend |
| `REINVENT4_AGENT_API_VERSION` | `2024-12-01-preview` on Azure routes | Azure `api-version` (OpenAI); override for deployments needing a newer one |
| `REINVENT4_AGENT_NO_VERIFY_SSL` | `true` | Disable TLS verification |
| `REINVENT4_AGENT_THINKING_BUDGET` | `1024` | Extended-thinking token budget (Anthropic) |
| `REINVENT4_AGENT_REASONING_EFFORT` | — | `low`/`medium`/`high` for o-series deployments (OpenAI) |
| `REINVENT4_AGENT_MAX_TOKENS` | `128000` for `gpt-5`, `100000` for the o-series, else `4096` | Maximum tokens per response. On reasoning models the cap also covers hidden reasoning tokens, so `env.sh` defaults it to the model's full output limit and the backend floors it at 32768 |

**Multi-agent**

| Variable | Default | Description |
|----------|---------|-------------|
| `REINVENT4_AGENT_MULTI_AGENT` | `true` | Run the read-only Reviewer alongside the Driver |
| `REINVENT4_AGENT_MAX_TURNS` | `8` | Turn cap for a `dd-coscientist` run |
| `REINVENT4_AGENT_TOKEN_BUDGET` | unlimited | Combined token budget across both agents |

**Monitoring and logging**

| Variable | Default | Description |
|----------|---------|-------------|
| `REINVENT4_AGENT_MONITOR_INTERVAL` | `300` | Seconds between background monitoring polls |
| `REINVENT4_AGENT_NO_MONITOR` | `false` | Disable background monitoring entirely |
| `REINVENT4_AGENT_LOG_DIR` | `$REINVENT4_PROJECT_DIR/logs` | Session logs + JSONL transcripts (fixed to `/logs` in-container) |
| `REINVENT4_AGENT_LOG_TIMESTAMPS` | — | Force timestamps on captured output (`1`/`0`) |

**Paths, GPUs, and Slurm**

| Variable | Default | Description |
|----------|---------|-------------|
| `REINVENT4_PROJECT_DIR` | `$HOME/dd_agentic` | Project root on the cluster |
| `REINVENT4_ROOT` | `$REINVENT4_PROJECT_DIR/REINVENT4` | REINVENT4 source tree (added to `sys.path`) |
| `REINVENT4_MCP_CONFIG` | `$REINVENT4_PROJECT_DIR/reinvent4_mcp_config.yaml` | Process config path |
| `REINVENT4_MCP_COMMAND` | `dd-mcp` | Command used to launch the MCP server |
| `DATA_DIR` | `./data` | Host directory bind-mounted into the containers |
| `REINVENT_GPU` | first of `SLURM_STEP_GPUS`, else `0` | GPU for REINVENT / the agent |
| `BOLTZ_GPU` | second of `SLURM_STEP_GPUS`, else `1` | GPU for Boltz2 |
| `SLURM_PARTITION` | `batch` | Cluster partition |
| `SLURM_CPUS` / `SLURM_MEM` / `SLURM_TIME` | `4` / `16G` / `4:00:00` | Slurm defaults |

## License

See the root repository for license information.
