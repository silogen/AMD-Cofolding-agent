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
│                    │   dd-mcp    │  59 tools                          │
│                    │ REINVENT4   │  scoring · TTPs · campaigns · jobs │
│                    └──────┬──────┘                                    │
└───────────────────────────┼───────────────────────────────────────────┘
                            │
          ┌─────────────────┼───────────────────────┐
          │                 │                       │
          │ POST /score     │ task-specific REST    │ HTTPS data access
          ▼                 ▼                       ▼
┌───────────────────┐ ┌───────────────────┐ ┌──────────────────────────┐
│ Scoring services  │ │ Support services  │ │ Public scientific APIs   │
│                   │ │                   │ │                          │
│ Boltz2       :8091│ │ Synthesis    :8086│ │ RCSB PDB                 │
│ PoseBusters  :8082│ │ StructPrep   :8087│ │ ChEMBL                   │
│ Interaction  :8083│ │ Annotation   :8088│ │ UniProt / InterPro       │
│ ADMET        :8084│ │                   │ │ PubChem                  │
│ RetroDMTA    :8085│ │                   │ │                          │
└─────────┬─────────┘ └─────────┬─────────┘ └──────────────────────────┘
          │                     │
          └──────────┬──────────┘
                     ▼
           ┌───────────────────────┐
           │    /shared volume     │
           │ structures · MSAs     │
           │ poses · annotations   │
           │ preparation artifacts│
           └───────────────────────┘
```

The agent spawns the MCP server as a subprocess over stdio. The server imports REINVENT4 for native scoring and Reinvent/LibInvent/LinkInvent/Mol2Mol campaigns, manages TTPs and background jobs, and exposes the structure, synthesis, annotation, and public-data tools through one 59-tool MCP catalog.

The five scoring services expose the uniform `POST /score` contract used by REINVENT's `ExternalRESTScorer`. The three support services expose task-specific preparation, retrosynthesis, availability, and annotation endpoints. A `PipelineAwareScorer` passes artifacts between objectives: Structure Preparation emits validated receptors/complexes, Boltz2 can cofold from sequence or condition on an input structure, and PoseBusters plus the Interaction Profiler consume the resulting `/shared` structures. The MCP fetchers call RCSB PDB, ChEMBL, UniProt, InterPro, and PubChem directly and cache reusable artifacts or annotations where appropriate.

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
| 9 | Structure-conditioned Boltz cofolding from a prepared input structure | Done |
| 10 | AiZynthFinder retrosynthesis prediction and building-block availability | Done |
| 11 | Structure Preparation service + expanded protein/fragment Interaction Profiler | Done |
| 12 | RCSB PDB, ChEMBL, UniProt, InterPro, PubChem, and nucleic-acid annotation fetchers | Done |
| 13 | Expanded ADMET and RetroDMTA ML endpoints for off-target, distribution, metabolism, elimination, and toxicity | Done |
| 14 | REINVENT LibInvent/LinkInvent transformer support and unified Mol2Mol campaigns | Done |
| 15 | Per-molecule evaluation detail tools for the Reviewer (`get_evaluation_detail`, `read_structure_file`) | Planned |
| 16 | OpenTelemetry tracing of agent turns + tool calls | Planned |

## Prerequisites

- A Slurm cluster with AMD GPUs (ROCm) or NVIDIA GPUs (CUDA)
- **Either** Docker with Compose v2 on the compute nodes, **or** Apptainer/Singularity with `--fakeroot`
- An LLM API key, or any Anthropic / OpenAI-compatible endpoint (see [Configure credentials](#3-configure-credentials))

Everything runs inside containers. No local Python environment, venv, or pip install is needed.

## Repository Layout

```
AMD-Cofolding-agent/
├── reinvent4_mcp/                       # MCP server package (dd-mcp), 59 tools
│   ├── server.py                        # FastMCP entry point (stdio transport)
│   ├── registry.py                      # Extension registration (blocks 01–03, 06–10)
│   ├── config.py                        # Process/service configuration
│   ├── scoring_pipeline.py              # Artifact-aware external-scoring pipeline
│   ├── tools/
│   │   ├── discovery.py                 # Components, models, transforms, model manifests
│   │   ├── scoring.py                   # Direct and TTP-based molecular scoring
│   │   ├── optimization.py              # RL and transfer-learning launch paths
│   │   ├── results.py                   # Status, trajectories, alerts, candidates, CSV results
│   │   ├── ttp.py                       # TTP create/update/get/list/delete
│   │   ├── external.py                  # ADMET, RetroDMTA, profiler, assets, service discovery
│   │   ├── rcsb.py                      # RCSB PDB search, metadata, and structure download
│   │   ├── chembl.py                    # ChEMBL molecules, targets, assays, and activities
│   │   ├── uniprot.py                   # Core UniProt target/sequence lookup
│   │   ├── uniprot_features.py          # Extended UniProt feature annotations
│   │   ├── interpro.py                  # InterPro domains + nucleic-acid annotations
│   │   ├── pubchem.py                   # PubChem compound name -> canonical SMILES
│   │   ├── structure_prep.py            # Validate/prepare structures and extract ligands
│   │   ├── interaction_profiler.py      # Complex interfaces, sites, fragments, interactions
│   │   ├── boltz_structure.py           # Structure-conditioned Boltz cofolding
│   │   ├── synthesis.py                 # AiZynthFinder planning and availability tools
│   │   └── generation.py                # Reinvent/LibInvent/LinkInvent/Mol2Mol campaigns
│   ├── jobs/
│   │   ├── manager.py                   # Serialized GPU jobs, persistence, monitoring
│   │   ├── toml_generator.py            # Reinvent, LibInvent, LinkInvent, Mol2Mol, TL TOML
│   │   └── _run_worker.py               # REINVENT subprocess entry point
│   ├── ttp/
│   │   ├── schema.py                    # TTP + generation campaign contracts
│   │   ├── store.py                     # JSON persistence and version history
│   │   └── mapper.py                    # TTP -> scorer/generation configuration
│   ├── _plugins/reinvent_plugins/components/
│   │   └── comp_external_rest.py        # ExternalRESTScorer implementation
│   └── pyproject.toml
├── reinvent4_agent/                     # Driver/Reviewer CLI (dd-agent, dd-coscientist)
│   ├── cli.py
│   ├── agent.py                         # Tool loop and background monitoring
│   ├── coordinator.py                   # Driver/Reviewer handoff and campaign monitor
│   ├── system_prompt.py                 # Medchem + operational tool guidance
│   ├── mcp_client.py
│   ├── display.py
│   └── backends/                        # Anthropic and OpenAI-compatible backends
├── admet-predictor/                     # Drug (17) + agro (25) endpoint ensembles
├── retrodmta-predictor/                 # Legacy + Mpro/CHIKV/EV-A71/FatA target models
├── interaction-profiler/                # Protein/fragment taxonomy and interface metrics
├── services/
│   ├── env.sh
│   ├── dd-agent/                        # Agent image, pinned REINVENT4, compatibility patch
│   ├── boltz2/                          # Cofolding, affinity, structure conditioning
│   ├── posebusters/                     # Pose validation
│   ├── admet-predictor/                 # ADMET REST service
│   ├── retrodmta-predictor/             # RetroDMTA REST service
│   ├── interaction-profiler/            # Interaction/interface REST service
│   ├── structure-prep/                  # Structure preparation REST service + Compose overlay
│   ├── synthesis-planner/               # AiZynthFinder routes, stock checks, async jobs
│   ├── annotation/                      # Cached UniProt/InterPro annotation proxy
│   ├── singularity/                     # Apptainer/Singularity definitions and runner
│   └── slurm/
│       ├── run_interactive.sh           # Canonical complete-stack interactive launcher
│       ├── llm_tunnel_relay.py          # Docker bridge -> host reverse-tunnel relay
│       ├── agent.sbatch
│       └── setup.sh
├── scripts/
│   └── generate_admet_training_ranges.py # Endpoint p01–p99 response-domain metadata
├── tests/                               # 525 integrated unit/contract tests
│   ├── test_admet_model_sets.py
│   ├── test_admet_response_domain.py
│   ├── test_boltz_structure_conditioned.py
│   ├── test_campaign_contracts.py
│   ├── test_generation_campaign_runtime.py
│   ├── test_generation_mapping.py
│   ├── test_synthesis_mcp.py
│   └── ...                              # Existing scoring, TTP, jobs, wrappers, agent tests
├── docker-compose.yml                   # Integrated scoring/support-service stack
├── docker-compose.tunnel.yml            # Optional AMD gateway reverse-tunnel overlay
├── reinvent4_mcp_config.yaml
├── TENSORWAVE_INSTRUCTIONS.md
├── TESTING.md
├── STACK_MANIFEST.json
│
│  # --- Gitignored runtime/build inputs ---
├── REINVENT4/                           # Pinned source restored by run_interactive.sh
├── boltz2/                              # Pinned source restored by run_interactive.sh
├── models/                              # ADMET + RetroDMTA model bundle and checksums
├── data/                                # Shared assets, TTPs, jobs, logs, caches
└── services/env.local.sh                # Local credentials and cluster overrides
```

## Setup

### 1. Restore the gitignored dependencies

`REINVENT4/`, `boltz2/`, `posebusters/`, and `models/` are gitignored because they are large external artifacts. From the project root:

```bash
cd refm-core-private/examples/dd_agentic

git clone https://github.com/MolecularAI/REINVENT4.git   # native scoring + RL
git clone https://github.com/jwohlwend/boltz.git boltz2   # cofolding + affinity
git clone https://github.com/maabuu/posebusters.git       # pose validation
```

The ADMET and RetroDMTA services need trained model weights, distributed as a SharePoint zip rather than in git. Unzip it so that `models/admet/` and `models/retrodmta/` exist — the Dockerfiles bake them into the images at build time:

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

**AMD LLM Gateway** — set `LLM_GATEWAY_KEY`. It is sent as the `Ocp-Apim-Subscription-Key` header, together with the `user` and `anthropic-version` headers the gateway requires:

```bash
export LLM_GATEWAY_KEY="your-gateway-subscription-key"
export REINVENT4_AGENT_BACKEND="anthropic"
export REINVENT4_AGENT_MODEL="claude-sonnet-4.6"
export REINVENT4_AGENT_BASE_URL="https://llm-api.amd.com/Anthropic"
```

The gateway publishes OpenAI models as **Azure deployments**, so point the base URL at the gateway root and use the deployment id as the model name. The backend appends `/openai/deployments/<model>` and sends `api-version=2024-12-01-preview` by default. Authentication uses the same `Ocp-Apim-Subscription-Key` and `user` headers as the Anthropic route:

```bash
export LLM_GATEWAY_KEY="your-gateway-subscription-key"
export REINVENT4_AGENT_BACKEND="openai"
export REINVENT4_AGENT_MODEL="GPT-5.4"            # Azure deployment id
export REINVENT4_AGENT_BASE_URL="https://llm-api.amd.com"
```

Override `REINVENT4_AGENT_API_VERSION` if a deployment needs a newer version. If your gateway also exposes the plain OpenAI-compatible route, name it in the base URL (`https://llm-api.amd.com/openai/v1`) and the backend uses the standard client instead.

**Any non-AMD endpoint (Azure OpenAI, Anthropic direct, or another OpenAI-compatible gateway)** — leave `LLM_GATEWAY_KEY` empty and set `REINVENT4_AGENT_API_KEY` instead. This is the plain bearer/API key for the endpoint and is passed straight to the SDK; no gateway headers are added:

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

If the compute node cannot reach the AMD gateway directly, see [TENSORWAVE_INSTRUCTIONS.md](TENSORWAVE_INSTRUCTIONS.md) for the SSH reverse-tunnel setup.

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

The container reaches the LLM gateway directly by DNS. Only if the node cannot, add the tunnel overlay (see [TENSORWAVE_INSTRUCTIONS.md](TENSORWAVE_INSTRUCTIONS.md)), which redirects the gateway hostname to the SSH relay on the Docker host:

```bash
docker compose -f docker-compose.yml -f docker-compose.tunnel.yml run --rm reinvent4-mcp
```

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
| `docker-compose.tunnel.yml` LLM relay | host loopback is shared directly — no relay needed |

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
| `--gateway-loopback` | Map the gateway hostname to `127.0.0.1` for an SSH tunnel |
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

For the SSH-tunnel workflow and validation steps, see [services/singularity/README.md](services/singularity/README.md).

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
| `--gateway-key` | — | AMD LLM Gateway subscription key (`Ocp-Apim-Subscription-Key`) |
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

# OpenAI model on the AMD gateway (Azure deployments route)
dd-agent --backend openai --model GPT-5.4 \
         --base-url https://llm-api.amd.com \
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

59 tools are exposed to the Driver. Tools marked read-only are also available to the Reviewer.

| Tool | Read-only | Description |
|------|:---------:|-------------|
| `list_scoring_components` | ✓ | Discover all native REINVENT4 scoring components |
| `list_available_models` | ✓ | Check which generative model priors are available |
| `list_transforms` | ✓ | List score transform types (sigmoid, step, etc.) |
| `list_external_services` | ✓ | Discover and health-check configured scoring services |
| `list_shared_assets` | ✓ | List real files on `/shared` (structures, MSAs, cofold outputs) |
| `score_molecules` | | Score SMILES directly or through a TTP |
| `predict_admet` | | ADMET consensus predictions with applicability-domain flags |
| `predict_retrodmta` | | Per-target potency / ADME predictions |
| `profile_interactions` | | Fragment taxonomy and protein-ligand interaction profiling |
| `create_ttp` | | Create a Therapeutic Target Profile |
| `update_ttp` | | Update a TTP (auto-increments version) |
| `get_ttp` | ✓ | Retrieve a TTP by ID, optionally at a specific version |
| `list_ttps` | ✓ | List all stored TTPs |
| `delete_ttp` | | Delete a TTP and its version history |
| `run_optimization` | | Launch a standard REINVENT or Mol2Mol RL campaign |
| `get_optimization_status` | ✓ | Trajectories, per-component stats, convergence diagnostics |
| `stop_optimization` | | Stop a running job and save a checkpoint |
| `run_transfer_learning` | | Fine-tune a generative model on a SMILES dataset |
| `get_top_candidates` | ✓ | Extract the best molecules from a run |
| `read_results` | ✓ | Parse a REINVENT4 output CSV into summary statistics |
| `list_jobs` | ✓ | List all tracked jobs and their states |
| `get_job_alerts` | ✓ | Check tracked jobs for actionable warnings |
| `fetch_target_info` | ✓ | Resolve a UniProt accession or free text to a protein sequence |
| `fetch_compound` | ✓ | Resolve a compound name to a real SMILES via PubChem |
| `rcsb_search` | ✓ | Search RCSB PDB with structured queries and facets |
| `rcsb_count` | ✓ | Count matching RCSB PDB entries |
| `rcsb_get_entries` | ✓ | Retrieve enriched PDB entry/entity metadata |
| `rcsb_fetch_files` | ✓ | Download selected PDB/mmCIF structures into `/shared` |
| `search_chembl_molecules` | ✓ | Search ChEMBL molecules |
| `get_chembl_molecule` | ✓ | Retrieve one ChEMBL molecule record |
| `search_chembl_targets` | ✓ | Search ChEMBL biological targets |
| `get_chembl_target` | ✓ | Retrieve one ChEMBL target record |
| `search_chembl_activities` | ✓ | Query filtered ChEMBL bioactivities |
| `fetch_all_activities` | ✓ | Fetch paginated activities for a target or assay |
| `search_chembl_assays` | ✓ | Search ChEMBL assay records |
| `get_chembl_mechanisms` | ✓ | Retrieve drug mechanism records |
| `get_chembl_indications` | ✓ | Retrieve indication records |
| `get_chembl_documents` | ✓ | Retrieve source publication metadata |
| `search_chembl_similarity` | ✓ | Search ChEMBL by molecular similarity |
| `search_chembl_substructure` | ✓ | Search ChEMBL by substructure |
| `validate_structure` | ✓ | Inspect a PDB/mmCIF and propose a preparation profile |
| `prepare_structure` | | Run the structure-preparation pipeline |
| `get_preparation_status` | ✓ | Poll preparation status and retrieve artifact paths |
| `extract_ligand_sdf` | ✓ | Extract a selected ligand into an SDF artifact |
| `profile_complex_interaction` | ✓ | Profile protein-ligand or protein-protein interfaces |
| `get_interaction_profiler_schema` | ✓ | Retrieve interaction-profiler response schemas |
| `characterize_site_profile` | ✓ | Characterize pocket geometry and interaction features |
| `plan_synthesis` | ✓ | Predict retrosynthetic routes with AiZynthFinder |
| `check_building_block_availability` | ✓ | Query configured synthesis-stock availability |
| `submit_synthesis_job` | | Submit an asynchronous retrosynthesis batch |
| `get_synthesis_job_status` | ✓ | Poll an asynchronous synthesis job |
| `list_generation_models` | ✓ | Discover Reinvent, LibInvent, LinkInvent, and Mol2Mol priors |
| `validate_generation_campaign` | ✓ | Validate a generation campaign without launching |
| `create_generation_campaign` | | Persist a generation-capable TTP |
| `run_generation_campaign` | | Run Reinvent, LibInvent, LinkInvent, or Mol2Mol campaigns |
| `predict_structure_conditioned` | | Cofold a ligand while conditioning on an input structure |
| `fetch_uniprot_features` | ✓ | Retrieve detailed UniProt residue and sequence features |
| `fetch_interpro_annotations` | ✓ | Retrieve InterPro family/domain annotations |
| `fetch_nucleic_annotation` | ✓ | Retrieve nucleic-acid feature annotations |

Generation campaigns support the RNN and PubChem-trained transformer variants of LibInvent and LinkInvent (`.libinvent_transformer`, `.linkinvent_transformer`) as well as seed-conditioned Mol2Mol models (`.m2m_*`). The campaign tools validate mode-specific scaffold, fragment, or seed inputs, generate the required auxiliary files, and expose one tracked submission/status/results lifecycle.

---

## External Scoring Services

Five external scoring services and three MCP support services provide the scientific runtime.

| Service | Purpose | GPU | Host port |
|---------|---------|:---:|-----------|
| **Boltz2** | Protein-ligand cofolding, affinity, and input-structure conditioning | Yes | 8091 |
| **PoseBusters** | Geometric and chemical quality checks on predicted poses | No | 8082 |
| **Interaction Profiler** | Fragment taxonomy, protein interfaces, sites, and typed interactions | No | 8083 |
| **ADMET Predictor** | Drug/agro ADMET ensembles with chemical/response-domain flags | No | 8084 |
| **RetroDMTA Predictor** | Legacy and FBDD target potency/ADME endpoints | No | 8085 |
| **Synthesis Planner** | AiZynthFinder routes, stock availability, and async planning | No | 8086 |
| **Structure Preparation** | PDB/mmCIF validation, repair, protonation, and ligand extraction | No | 8087 |
| **Annotation Service** | Cached UniProt/InterPro annotations for profiler enrichment | No | 8088 |

### REST API contract

The five scoring services implement the shared scorer interface. `GET /health` reports readiness, and `POST /score` takes:

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

ADMET, RetroDMTA, and the Interaction Profiler additionally expose richer out-of-band prediction/profile endpoints. Structure Preparation, Synthesis Planner, and Annotation are MCP support services with task-specific validation, job, preparation, route, availability, and annotation endpoints rather than REINVENT scalar-scoring contracts.

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

`predict_structure_conditioned` adds the prepared protein structure as a Boltz template and places the ligand while preserving the selected receptor chain geometry. It accepts `/shared` PDB/mmCIF paths from the Structure Preparation service, optional template-chain mapping and SHA-256 verification, soft or forced backbone conditioning, multiple diffusion samples ranked by confidence, and optional PoseBusters validation of the best result.

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

Builds a hierarchical medchem fragment taxonomy from SMILES and profiles prepared complexes in ligand-target, binder-target, or generalized biomolecular-interface modes. Structure-aware analysis includes typed interactions (H-bonds, hydrophobic, π-stacking, ionic, halogen, metal), `(fragment, residue, interaction)` triples, site characterization, interface metrics, ligand strain/efficiency, secondary structure, and UniProt/InterPro enrichment. `POST /score` still returns an interaction-family count for use after a structure-producing RL objective.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `interaction_type` | `"all"` | Family to count (`hbond`, `hydrophobic`, `ionic`, `pi_stacking`, …, or `all`) |
| `structure_paths` | `null` | Per-SMILES structure paths, injected from an upstream cofold |
| `ph` | `7.4` | Protonation pH |
| `radius` | `8.0` | Pocket crop radius (Å) |

### Structure Preparation

Validates raw PDB/mmCIF inputs before any cofolding or interaction analysis, inventories chains/ligands/additives/missing residues, and proposes a preparation profile. Asynchronous preparation can curate entities, resolve alternate locations, repair atoms, protonate with PDB2PQR/PROPKA, parameterize ligands, optionally minimize, and emit prepared complexes, receptors, ligand SDFs, FASTA sequences, and provenance under `/shared`.

### Synthesis Planner

Wraps AiZynthFinder for synchronous route prediction, asynchronous batch planning, and building-block availability checks. `plan_synthesis` returns ranked reaction trees and leaf reactants; `submit_synthesis_job` / `get_synthesis_job_status` handle larger batches. Stock-dependent conclusions are reported separately so an unavailable stock catalog does not masquerade as verified purchasability.

### Annotation and Data Fetchers

RCSB PDB and ChEMBL tools provide structures, targets, assays, activities, mechanisms, indications, and similarity/substructure search. UniProt, InterPro, PubChem, and nucleic-acid annotation tools provide canonical sequences, residue/domain features, compound SMILES, and biological context. The annotation service caches UniProt/InterPro responses for reuse by the Interaction Profiler.

### ADMET Predictor

Manifest-driven XGBoost + Chemprop-MTL ensembles with selectable `model_set="drug"` (17 endpoints) or `model_set="agro"` (25 endpoints). The expanded endpoints cover physicochemical properties, distribution/permeability and efflux, CYP/hERG off-target liabilities, microsomal/plasma metabolism and elimination, protein binding, and agrochemical bioaccumulation, environmental fate, volatility, and acute toxicity.

Each result reports accepted model members, consensus mean ± std, Avalon chemical-space applicability, physical-bound member exclusions, and a per-endpoint p01–p99 training-response-domain flag. `POST /score` returns the selected endpoint for REINVENT objectives; `/predict` returns the complete self-describing profile. Weights and endpoint manifests come from `models/admet/{drug,agro}/`.

### RetroDMTA Predictor

Per-target, per-temporal-window project predictions for the legacy `FXA`, `MMP8`, `PPARD`, and `RENIN` models plus the new `Mpro`, `CHIKV_Mac`, `EV-A71_2A`, and `FatA` FBDD targets. Legacy bundles retain their available XGBoost/Chemprop/CheMeleon members; current FBDD bundles use their exported XGBoost endpoints and ensemble fallback policy. `GET /targets` exposes the exact endpoint, window, method, unit, and model availability contract for each target.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target` | (required) | `FXA`, `MMP8`, `PPARD`, `RENIN`, `Mpro`, `CHIKV_Mac`, `EV-A71_2A`, or `FatA` |
| `endpoint` | first endpoint of the target | Target-specific pKi/pIC50/pEC50/pKd or auxiliary endpoint |
| `window` | `"latest"` | Temporal window index, or `"latest"` |
| `method` | `"ensemble"` | `ensemble`, `xgboost`, `chemprop`, or `chemeleon` |

Weights and target/window manifests come from `models/retrodmta/`. Unlike ADMET, no applicability domain is currently reported.

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
| `LLM_GATEWAY_KEY` | — | AMD LLM Gateway subscription key (`Ocp-Apim-Subscription-Key`) |
| `REINVENT4_AGENT_API_KEY` | — | Direct API key for any non-AMD endpoint (Azure, Anthropic, …) |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | — | SDK-standard fallbacks; not forwarded into containers |

**LLM backend**

| Variable | Default | Description |
|----------|---------|-------------|
| `REINVENT4_AGENT_BACKEND` | `anthropic` | `anthropic` or `openai` |
| `REINVENT4_AGENT_MODEL` | `claude-sonnet-4.6` | Model name |
| `REINVENT4_AGENT_BASE_URL` | `https://llm-api.amd.com/Anthropic` | LLM API base URL; use the gateway root `https://llm-api.amd.com` for the OpenAI backend |
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
