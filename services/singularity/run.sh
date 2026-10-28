#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ──────────────────────────────────────────────────────────────────────
# Run the full dd_agentic stack under Apptainer/Singularity (no Docker).
#
# Starts the 5 service instances (boltz + 4 REST predictors) with host
# networking, health-gates them, then runs the agent. All instances are
# stopped on exit.
#
# Usage:
#   bash services/singularity/run.sh --vendor <amd|nvidia> [-- <agent args>]
#
# Examples:
#   bash services/singularity/run.sh --vendor nvidia
#   echo "Score aspirin ..." | bash services/singularity/run.sh --vendor amd
#   bash services/singularity/run.sh --vendor amd -- --some-agent-flag
#
# Options:
#   --vendor <amd|nvidia>   REQUIRED. Selects --rocm/--nv and the *-<vendor>.sif.
#   --data-dir <path>       Host data dir to bind (default: ./data).
#   --reinvent-gpu <id>     GPU index for the agent (default: $REINVENT_GPU or 0).
#   --boltz-gpu <id>        GPU index for boltz    (default: $BOLTZ_GPU or 1).
#   --boltz-gpus <ids>      Comma-separated GPUs for boltz, e.g. 5,6,7; overrides
#                           --boltz-gpu (default: $BOLTZ_GPUS, else the one GPU).
#                           Boltz runs $BOLTZ_WORKERS_PER_GPU (default 2) resident
#                           workers on each.
#   --skip-agent            Start + health-gate services only, then wait (Ctrl-C).
#   --web                   Run the dd-web gateway for the browser UI instead of
#                           the terminal agent ($DD_WEB_HOST:$DD_WEB_PORT,
#                           default 127.0.0.1:8000). Set REINVENT4_WEB_TOKEN
#                           whenever DD_WEB_HOST is reachable by other users.
#   --dry-run               Print the apptainer commands without running them.
#
# APPTAINER_GPU_FLAG overrides --rocm/--nv. On LUMI set it to "": the GPUs are
# visible without it, and --rocm would mount the host's older ROCm libraries.
#
# GPU assignment mirrors services/env.sh: agent -> GPU0, boltz -> GPU1.
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"   # dd_agentic/

# --- Defaults ---------------------------------------------------------
VENDOR=""
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
REINVENT_GPU="${REINVENT_GPU:-0}"
BOLTZ_GPU="${BOLTZ_GPU:-1}"
BOLTZ_GPUS="${BOLTZ_GPUS:-}"
SKIP_AGENT=0
WEB=0
DRY_RUN=0
AGENT_ARGS=()

# --- Parse args -------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --vendor)           VENDOR="$2"; shift 2 ;;
        --data-dir)         DATA_DIR="$2"; shift 2 ;;
        --reinvent-gpu)     REINVENT_GPU="$2"; shift 2 ;;
        --boltz-gpu)        BOLTZ_GPU="$2"; shift 2 ;;
        --boltz-gpus)       BOLTZ_GPUS="$2"; shift 2 ;;
        --skip-agent)       SKIP_AGENT=1; shift ;;
        --web)              WEB=1; shift ;;
        --dry-run)          DRY_RUN=1; shift ;;
        --)                 shift; AGENT_ARGS=("$@"); break ;;
        -h|--help)          sed -n '5,39p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)                  echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

case "${VENDOR}" in
    amd)    GPU_FLAG="--rocm" ;;
    nvidia) GPU_FLAG="--nv" ;;
    *)      echo "error: --vendor must be amd or nvidia" >&2; exit 1 ;;
esac
GPU_FLAG="${APPTAINER_GPU_FLAG-${GPU_FLAG}}"
BOLTZ_GPUS="${BOLTZ_GPUS:-${BOLTZ_GPU}}"

run() { echo "+ $*"; [ "${DRY_RUN}" -eq 0 ] && "$@"; return 0; }

# --- Resolve the container runtime (apptainer or singularity) ---------
# HPC clusters often expose it as a module (`module load apptainer`) or as the
# legacy `singularity` command. Override explicitly with APPTAINER_BIN=/path.
APPTAINER="${APPTAINER_BIN:-}"
if [ -z "${APPTAINER}" ]; then
    if command -v apptainer >/dev/null 2>&1; then
        APPTAINER=apptainer
    elif command -v singularity >/dev/null 2>&1; then
        APPTAINER=singularity
    else
        APPTAINER=apptainer   # keep --dry-run output readable
        if [ "${DRY_RUN}" -eq 0 ]; then
            echo "error: neither 'apptainer' nor 'singularity' found on PATH." >&2
            echo "  On HPC try:  module load apptainer   (or: module load singularity)" >&2
            echo "  Or set:      APPTAINER_BIN=/path/to/apptainer" >&2
            exit 127
        fi
    fi
fi

# Emit the vendor-appropriate GPU-visibility --env flag for a given GPU id.
gpu_env_flag() {
    local gpu="$1"
    if [ "${VENDOR}" = "amd" ]; then
        # ROCR renumbers the GPUs it keeps from 0, so a HIP_VISIBLE_DEVICES with the
        # same id would then point past the single visible device.
        echo "--env ROCR_VISIBLE_DEVICES=${gpu}"
    else
        echo "--env CUDA_VISIBLE_DEVICES=${gpu}"
    fi
}

SHARED_BIND="${DATA_DIR}/shared:/shared"
mkdir -p "${DATA_DIR}/shared"

# Boltz writes its model-weight/CCD cache to /cache, but the Apptainer container
# root is read-only. Bind a persistent host dir (mirrors the Compose boltz_cache
# volume) so weights download once and are reused across runs.
BOLTZ_CACHE_BIND="${DATA_DIR}/boltz_cache:/cache"
mkdir -p "${DATA_DIR}/boltz_cache"

# --- Service table ----------------------------------------------------
# name  port  sif                            gpu?   shared-bind?
# boltz is the only GPU service instance; admet/retrodmta run CPU (add --nv/
# --rocm + a free GPU manually if you want admet on GPU).
INSTANCES=(
    "boltz|8091|boltz-${VENDOR}.sif|gpu|shared"
    "posebusters|8082|posebusters.sif|cpu|shared"
    "admet-predictor|8084|admet-predictor-${VENDOR}.sif|cpu|shared"
    "retrodmta-predictor|8085|retrodmta-predictor.sif|cpu|shared"
    "interaction-profiler|8083|interaction-profiler.sif|cpu|shared"
)

STARTED=()

cleanup() {
    echo ""
    echo "=== Stopping instances ==="
    for name in ${STARTED[@]+"${STARTED[@]}"}; do
        run "${APPTAINER}" instance stop "dd_${name}" || true
    done
}
trap cleanup EXIT INT TERM

# --- Health poll ------------------------------------------------------
health_ok() {
    local port="$1"
    if command -v curl >/dev/null 2>&1; then
        curl -fsS "http://localhost:${port}/health" >/dev/null 2>&1
    else
        python3 -c "import sys,urllib.request; urllib.request.urlopen('http://localhost:${port}/health')" 2>/dev/null
    fi
}

wait_health() {
    local name="$1" port="$2" tries=60
    echo -n "  waiting for ${name} on :${port} "
    if [ "${DRY_RUN}" -eq 1 ]; then echo "(dry-run: skipped)"; return 0; fi
    for _ in $(seq "${tries}"); do
        if health_ok "${port}"; then echo "OK"; return 0; fi
        echo -n "."; sleep 5
    done
    echo " FAILED"
    echo "error: ${name} did not become healthy on :${port}" >&2
    return 1
}

echo "Vendor    : ${VENDOR} (${GPU_FLAG})"
echo "Data dir  : ${DATA_DIR}"
echo "GPUs      : agent=${REINVENT_GPU}  boltz=${BOLTZ_GPUS}"
echo ""

# --- Start service instances -----------------------------------------
echo "=== Starting service instances ==="
for row in "${INSTANCES[@]}"; do
    IFS='|' read -r name port sif kind share <<<"${row}"
    sif_path="${SCRIPT_DIR}/${sif}"
    if [ "${DRY_RUN}" -eq 0 ] && [ ! -f "${sif_path}" ]; then
        echo "error: image not found: ${sif_path} (build it with build.sh ${VENDOR})" >&2
        exit 1
    fi

    opts=(--env "PORT=${port}")
    [ "${share}" = "shared" ] && opts+=(--bind "${SHARED_BIND}")
    # boltz needs a writable /cache (read-only container root otherwise).
    # The image bakes only app.py; the worker-pool modules come from this
    # checkout, so the whole wrapper dir is bound over the baked copy.
    if [ "${name}" = "boltz" ]; then
        opts+=(--bind "${BOLTZ_CACHE_BIND}")
        opts+=(--bind "${ROOT_DIR}/services/boltz2:/app/wrapper")
        if [ -n "${BOLTZ_WORKERS_PER_GPU:-}" ]; then
            opts+=(--env "BOLTZ_WORKERS_PER_GPU=${BOLTZ_WORKERS_PER_GPU}")
        fi
    fi
    # The image installs the profiler package at build time; bind the checkout
    # over it so profiler fixes deploy without a rebuild.
    if [ "${name}" = "interaction-profiler" ]; then
        opts+=(--bind "${ROOT_DIR}/interaction-profiler/src/interaction_profiler:/opt/conda/lib/python3.11/site-packages/interaction_profiler")
    fi
    if [ "${kind}" = "gpu" ]; then
        opts+=(${GPU_FLAG})
        # shellcheck disable=SC2046
        opts+=($(gpu_env_flag "${BOLTZ_GPUS}"))
    fi

    # Reusing a healthy instance lets the gateway be restarted with new settings
    # without taking the scoring services (and Boltz's model load) down with it.
    if [ "${DRY_RUN}" -eq 0 ] && "${APPTAINER}" instance list 2>/dev/null | grep -qE "^dd_${name}\b"; then
        echo "  reusing running instance dd_${name}"
        continue
    fi
    run "${APPTAINER}" instance start "${opts[@]}" "${sif_path}" "dd_${name}"
    STARTED+=("${name}")
done

echo ""
echo "=== Health-gating services ==="
for row in "${INSTANCES[@]}"; do
    IFS='|' read -r name port _ _ _ <<<"${row}"
    wait_health "${name}" "${port}"
done

if [ "${SKIP_AGENT}" -eq 1 ]; then
    echo ""
    echo "All services healthy. --skip-agent set; press Ctrl-C to stop."
    [ "${DRY_RUN}" -eq 0 ] && sleep infinity
    exit 0
fi

# --- Run the agent ----------------------------------------------------
echo ""
echo "=== Running agent ==="
AGENT_SIF="${SCRIPT_DIR}/dd-agent-${VENDOR}.sif"
if [ "${DRY_RUN}" -eq 0 ] && [ ! -f "${AGENT_SIF}" ]; then
    echo "error: agent image not found: ${AGENT_SIF} (build it with build.sh ${VENDOR})" >&2
    exit 1
fi

mkdir -p "${DATA_DIR}/ttps" "${DATA_DIR}/jobs" "${DATA_DIR}/logs" \
         "${DATA_DIR}/tb_logs" "${DATA_DIR}/agent_logs"

AGENT_BINDS=(
    --bind "${SHARED_BIND}"
    # Overlay the baked MCP config with the current source so config edits
    # (e.g. service ports/URLs) take effect without rebuilding the agent SIF.
    --bind "${SCRIPT_DIR}/reinvent4_mcp_config.singularity.yaml:/app/reinvent4_mcp_config.singularity.yaml"
    --bind "${DATA_DIR}/ttps:/data/reinvent4_mcp/ttps"
    --bind "${DATA_DIR}/jobs:/data/reinvent4_mcp/jobs"
    --bind "${DATA_DIR}/logs:/data/reinvent4_mcp/logs"
    --bind "${DATA_DIR}/tb_logs:/data/reinvent4_mcp/tb_logs"
    --bind "${DATA_DIR}/agent_logs:/logs"
)
# The model supervisor hands the gateway the running model through this directory.
# A path inside the container (e.g. /logs/model-control) is already covered by
# the log bind; only a host directory needs its own bind.
if [ -n "${REINVENT4_WEB_MODEL_CONTROL:-}" ] && [ -d "${REINVENT4_WEB_MODEL_CONTROL}" ]; then
    AGENT_BINDS+=(--bind "${REINVENT4_WEB_MODEL_CONTROL}:${REINVENT4_WEB_MODEL_CONTROL}")
fi
# DEV_SOURCE=1 runs the agent and MCP server from this checkout instead of the
# copies baked into the image, so Python edits need a restart, not a rebuild.
# Dependency changes (pyproject.toml) still need build.sh.
if [ "${DEV_SOURCE:-0}" = 1 ]; then
    SITE=/opt/venv/lib/python3.12/site-packages
    AGENT_BINDS+=(
        --bind "${ROOT_DIR}/reinvent4_agent:${SITE}/reinvent4_agent"
        --bind "${ROOT_DIR}/reinvent4_mcp:${SITE}/reinvent4_mcp"
    )
fi

# Forward agent env into the container. REINVENT4_MCP_CONFIG + LOG_DIR are fixed
# to the in-container paths; the rest (gateway URL, key, model, ...) pass through
# from the host only when set — so `source services/env.local.sh` before
# run.sh is enough to carry REINVENT4_AGENT_BASE_URL.
AGENT_ENV=(
    --env "REINVENT4_MCP_CONFIG=/app/reinvent4_mcp_config.singularity.yaml"
    --env "REINVENT4_AGENT_LOG_DIR=/logs"
)
if [ -z "${LLM_GATEWAY_KEY:-}" ] && [ -z "${REINVENT4_AGENT_API_KEY:-}" ]; then
    echo "warning: neither LLM_GATEWAY_KEY nor REINVENT4_AGENT_API_KEY is set" \
         "(source services/env.local.sh first)" >&2
fi
for _v in LLM_GATEWAY_KEY REINVENT4_AGENT_API_KEY \
          REINVENT4_AGENT_BACKEND REINVENT4_AGENT_MODEL REINVENT4_AGENT_BASE_URL \
          REINVENT4_AGENT_API_VERSION REINVENT4_AGENT_NO_VERIFY_SSL \
          REINVENT4_AGENT_THINKING_BUDGET REINVENT4_AGENT_REASONING_EFFORT \
          REINVENT4_AGENT_MULTI_AGENT REINVENT4_AGENT_LOG_TIMESTAMPS \
          REINVENT4_WEB_MODEL_CONTROL; do
    [ -n "${!_v:-}" ] && AGENT_ENV+=(--env "${_v}=${!_v}")
done

if [ "${WEB}" -eq 1 ]; then
    # The gateway is restarted whenever it exits, so a code change (DEV_SOURCE=1)
    # needs only `pkill -f '[d]d-web --host'`, not a restart of the services.
    # Ctrl-C or a TERM to this script ends the loop.
    trap 'exit 0' INT TERM
    while true; do
        # shellcheck disable=SC2046,SC2086
        run "${APPTAINER}" exec \
            ${GPU_FLAG} \
            $(gpu_env_flag "${REINVENT_GPU}") \
            "${AGENT_ENV[@]}" \
            "${AGENT_BINDS[@]}" \
            "${AGENT_SIF}" dd-web --host "${DD_WEB_HOST:-127.0.0.1}" --port "${DD_WEB_PORT:-8000}" || true
        [ "${DRY_RUN}" -eq 1 ] && exit 0
        echo "dd-web exited at $(date +%T); restarting in 3 s" >&2
        sleep 3
    done
fi

# shellcheck disable=SC2046,SC2086
run "${APPTAINER}" run \
    ${GPU_FLAG} \
    $(gpu_env_flag "${REINVENT_GPU}") \
    "${AGENT_ENV[@]}" \
    "${AGENT_BINDS[@]}" \
    "${AGENT_SIF}" ${AGENT_ARGS[@]+"${AGENT_ARGS[@]}"}
