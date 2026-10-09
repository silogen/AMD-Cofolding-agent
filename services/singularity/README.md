# Apptainer / Singularity stack (AMD + NVIDIA)

Run the full 6-service `dd_agentic` stack under **Apptainer/Singularity** on a
Slurm/HPC cluster with **no Docker daemon**, with separate GPU image variants
for **AMD (ROCm)** and **NVIDIA (CUDA)**. The Docker Compose setup
(`../../docker-compose.yml`) is left untouched; everything here is additive.

## What this replaces (vs. Compose)

| Compose feature            | Apptainer replacement                                        |
| -------------------------- | ----------------------------------------------------------- |
| `build:` from Dockerfiles  | native `.def` files, `apptainer build --fakeroot`           |
| bridge DNS `http://boltz:8080` | **host networking** + `http://localhost:<port>`         |
| `depends_on` + healthchecks | `apptainer instance start` + `/health` polling in `run.sh` |
| named volumes / mounts     | `--bind` (all services share one `/shared`)                 |
| `runtime: amd`             | `--rocm` (AMD) / `--nv` (NVIDIA), chosen by `--vendor`       |
| `extra_hosts` LLM relay    | host loopback is shared directly (see *LLM gateway* below)  |

## Services / ports / GPU

| service                | port | GPU        | image                              |
| ---------------------- | ---- | ---------- | ---------------------------------- |
| `dd-agent` (agent)     | —    | yes (GPU0) | `dd-agent-<vendor>.sif`            |
| `boltz`                | 8091 | yes (GPU1) | `boltz-<vendor>.sif`               |
| `admet-predictor`      | 8084 | no (CPU)   | `admet-predictor-<vendor>.sif`     |
| `posebusters`          | 8082 | no         | `posebusters.sif`                  |
| `retrodmta-predictor`  | 8085 | no         | `retrodmta-predictor.sif`          |
| `interaction-profiler` | 8083 | no         | `interaction-profiler.sif`         |

Ports mirror the Compose *published* ports. Because `boltz`, `posebusters`, and
`interaction-profiler` all listen on `:8080` internally, the listen port is made
env-driven (`PORT`) in each `.def` `%startscript`, and `run.sh` passes
`--env PORT=<mapped>` so there is no collision under host networking.

The GPU/vendor images differ only in their base image (and, for `admet`, the
torch wheel index + `PRED_DEVICE`). `device: "cuda:0"` is unchanged for AMD —
HIP presents as `torch.cuda`.

## Build (no Docker required)

> Requires the `apptainer` (or legacy `singularity`) command with `--fakeroot`.
> On HPC it's usually a module: `module load apptainer` (or `module load
> singularity`). Both scripts auto-detect either name; override with
> `APPTAINER_BIN=/path/to/apptainer`.

```bash
# from the dd_agentic/ root, on a machine with apptainer + --fakeroot
bash services/singularity/build.sh nvidia            # all NVIDIA images
bash services/singularity/build.sh amd               # all AMD images
bash services/singularity/build.sh amd boltz         # one service
bash services/singularity/build.sh nvidia --dry-run  # print commands only
```

Vendor base images / wheel indexes live in [`vendors.sh`](vendors.sh):

- **AMD**  — `rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.8.0`
  (exact base the current Dockerfiles use), admet torch from `whl/rocm6.3`.
- **NVIDIA** — `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` (torch 2.8.0 parity
  with the ROCm base; `-devel` needed for boltz's `pip install -e .`).
- **admet-predictor** runs on **CPU** for both vendors (as under Docker Compose):
  it installs the CPU torch wheel (`whl/cpu`) with `PRED_DEVICE=cpu`, so there is
  no GPU torch / ROCm-driver ABI risk. `run.sh` starts it without a GPU flag. The
  `TORCH_INDEX_URL` / `PRED_DEVICE` build-args remain if you later want a GPU wheel.

> Per-vendor GPU builds must run on (or for) the matching hardware; use
> `--dry-run` to preview the resolved `apptainer build` commands anywhere.

## Run

```bash
# on a 2-GPU allocation (agent -> GPU0, boltz -> GPU1)
bash services/singularity/run.sh --vendor nvidia
# AMD is identical:
bash services/singularity/run.sh --vendor amd

# pipe a prompt (batch mode)
echo "Score aspirin CC(=O)OC1=CC=CC=C1C(=O)O for drug-likeness" \
  | bash services/singularity/run.sh --vendor amd

# services only (no agent), useful for debugging
bash services/singularity/run.sh --vendor amd --skip-agent
```

`run.sh` starts the 5 service instances (binds, `--env PORT`, per-instance GPU
visibility, `--rocm`/`--nv`), polls each `/health`, then `apptainer run`s the
agent with the localhost config. All instances are stopped on exit (EXIT/INT/TERM
trap). Useful flags: `--data-dir`, `--reinvent-gpu`, `--boltz-gpu`, `--dry-run`.

### Config

The agent uses [`reinvent4_mcp_config.singularity.yaml`](reinvent4_mcp_config.singularity.yaml)
— a copy of `../../reinvent4_mcp_config.yaml` with every `service_url` pointed at
`http://localhost:<port>/score`. The `dd-agent` image defaults
`REINVENT4_MCP_CONFIG` to this file; `run.sh` also sets it explicitly. The
original config is never modified.

### Writable binds (SIF is read-only)

`run.sh` binds these host dirs under `--data-dir` (default `./data`), creating
them if missing: `/shared`, `/data/reinvent4_mcp/{jobs,logs,tb_logs}`,
`~/.reinvent4_mcp/ttps`, and `/logs` (agent transcripts). boltz's `/cache` is
in-container; bind a host path with `--bind` if you want it to persist.

### Environment / `env.local.sh`

**Yes — `source services/env.local.sh` (or `env.sh`) before running
`run.sh`.** Apptainer forwards host env vars into the container by default, the
image defaults are all guarded with `:-`, and `run.sh` explicitly forwards the
`REINVENT4_AGENT_*` vars + `LLM_GATEWAY_KEY` when set — so whatever you export
(gateway URL/port, key, model, GPU indices) is what the agent uses. `run.sh`
overrides only the two path-bound vars (`REINVENT4_MCP_CONFIG` -> the singularity
config, `REINVENT4_AGENT_LOG_DIR` -> `/logs`); the host values for those (which
are host paths) are intentionally ignored.

```bash
source services/env.local.sh    # LLM_GATEWAY_KEY, REINVENT4_AGENT_*, GPUs
bash services/singularity/run.sh --vendor amd
```

### LLM gateway reachability (SSH tunnel)

The Docker SSH tunnel works here too — **and it's simpler**: with host networking
the container shares the host's loopback, so the `llm_tunnel_relay.py` bridge is
**no longer needed** (it only existed to bridge the Docker bridge IP to the host
loopback). Compare:

```
# Docker:   container -> 172.17.0.1:PORT (relay) -> 127.0.0.1:PORT (ssh -R) -> laptop -> gateway
# Apptainer: container -> 127.0.0.1:PORT (ssh -R) -> laptop -> gateway      (no relay)
```

Workflow (see `../../LLM_GATEWAY_TUNNEL.md` for the full Docker version):

1. Open the reverse tunnel to the **compute node that runs Apptainer** (loopback
   is per-host, so it must be the same node — same caveat as Docker):
   ```bash
   ssh -R 127.0.0.1:9579:<gateway-host>:443 <node>
   ```
2. Point the agent at the tunnel port and keep TLS verification on (the hostname
   matches the real cert; connecting by IP/`localhost` would fail cert checks):
   ```bash
   export LLM_GATEWAY_KEY=...
   export REINVENT4_AGENT_BASE_URL="https://<gateway-host>:9579/Anthropic"
   export REINVENT4_AGENT_NO_VERIFY_SSL="false"
   ```
   (These live in `env.local.sh`; sourcing it is enough.)
3. Run with `--gateway-loopback` so `run.sh` binds a `/etc/hosts` mapping
   `<gateway-host> -> 127.0.0.1` into the agent (replacing Compose's
   `extra_hosts`), letting the hostname resolve to the loopback tunnel:
   ```bash
   bash services/singularity/run.sh --vendor amd --gateway-loopback
   ```

If the gateway is **directly reachable** from the compute node, skip the tunnel
and `--gateway-loopback` entirely and just set `LLM_GATEWAY_KEY` (+ a direct
`REINVENT4_AGENT_BASE_URL`).

## Validation

1. **CPU smoke test** (no GPU, no Docker needed):
   ```bash
   bash services/singularity/build.sh amd posebusters      # vendor-neutral image
   apptainer instance start --env PORT=8082 \
     services/singularity/posebusters.sif dd_posebusters
   curl -fsS http://localhost:8082/health
   apptainer instance stop dd_posebusters
   ```
2. **Per-vendor build dry-run** (no hardware):
   ```bash
   bash services/singularity/build.sh amd    --dry-run
   bash services/singularity/build.sh nvidia --dry-run
   ```
3. **End-to-end**: `run.sh --vendor <v>` health-gates all services, then the
   agent scores a known SMILES. Use `--dry-run` to inspect the exact
   `apptainer instance start` / `apptainer run` commands first.
```
bash services/singularity/run.sh --vendor amd --dry-run
```

## Files

| file | purpose |
| ---- | ------- |
| `dd-agent.def`, `boltz.def`, `admet-predictor.def` | GPU `.def`, vendor build-arg templating (`BASE_IMAGE`/`TORCH_INDEX_URL`/`PRED_DEVICE`) |
| `posebusters.def`, `retrodmta-predictor.def`, `interaction-profiler.def` | vendor-neutral CPU `.def` |
| `vendors.sh` | AMD/NVIDIA base image + torch index + device matrix |
| `build.sh` | `apptainer build --fakeroot` driver (`<amd\|nvidia> [service...] [--dry-run]`) |
| `run.sh` | starts instances, health-gates, runs agent, stops on exit |
| `reinvent4_mcp_config.singularity.yaml` | localhost service URLs |

The `.def` `%post` sections mirror `../*/Dockerfile`; the Dockerfiles and
`../../docker-compose.yml` remain the source of truth for build steps and
binds/ports.
