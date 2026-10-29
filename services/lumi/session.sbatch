#!/bin/bash
#SBATCH --job-name=cofolding-session
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# One LUMI-G node running the whole stack for the browser UI:
#   GCD 0-3  vLLM serving the chat model on 127.0.0.1:8001
#   GCD 4    REINVENT (dd-mcp inside dd-web)
#   GCD 5-7  Boltz, two resident workers per GCD; the other scoring services run on CPU
#   :8000    dd-web, the gateway the browser UI talks to (token required)
#
# Submit from the repository root (see services/lumi/README.md):
#   mkdir -p logs && sbatch --account=<project> services/lumi/session.sbatch
#
# CHAT_MODEL=qwen (default) or gemma picks the local model. CHAT_MODEL=external
# skips vLLM and uses the endpoint and key in services/env.local.sh instead.

set -uo pipefail
REPO=${REPO:-$SLURM_SUBMIT_DIR}
[[ -f $REPO/services/singularity/run.sh ]] || { echo "Submit from the repository root (got $REPO)"; exit 1; }
JOB=$SLURM_JOB_ID
export DATA_DIR=${DATA_DIR:-$REPO/data}
LOGS=$REPO/logs/session-$JOB
mkdir -p "$LOGS"
CHAT_MODEL=${CHAT_MODEL:-qwen}

if [[ $CHAT_MODEL == external ]]; then
    [[ -f $REPO/services/env.local.sh ]] || { echo "CHAT_MODEL=external needs services/env.local.sh"; exit 1; }
    # shellcheck disable=SC1091
    source "$REPO/services/env.local.sh"
else
    export REINVENT4_AGENT_BACKEND=openai
    export REINVENT4_AGENT_BASE_URL=http://127.0.0.1:8001/v1
    export REINVENT4_AGENT_MODEL=Qwen/Qwen3.8-27B
    export REINVENT4_AGENT_API_KEY=local-vllm
    export REINVENT4_AGENT_REASONING_EFFORT=${REASONING_EFFORT:-low}
fi
export REINVENT4_AGENT_LOG_TIMESTAMPS=1
export REINVENT_GPU=4 BOLTZ_GPU=5 BOLTZ_GPUS=5,6,7 DD_WEB_PORT=8000
# On LUMI the GPUs are visible without --rocm, which would mount the host's older ROCm.
export APPTAINER_GPU_FLAG=""
# Run the agent from this checkout, so a code change needs a gateway restart, not a rebuild.
export DEV_SOURCE=1

# The laptop tunnel reaches dd-web through a login node, so every LUMI user
# could too: dd-web listens on the node but refuses requests without the token.
export DD_WEB_HOST=0.0.0.0
TOKEN_FILE=$HOME/.dd-web-token
[[ -s $TOKEN_FILE ]] || (umask 077; openssl rand -hex 24 > "$TOKEN_FILE")
REINVENT4_WEB_TOKEN=$(cat "$TOKEN_FILE")
export REINVENT4_WEB_TOKEN
AUTH=(-H "X-DD-Token: $REINVENT4_WEB_TOKEN")

echo "Node $(hostname)  job $JOB  start $(date +%T)"

VLLM=""
if [[ $CHAT_MODEL != external ]]; then
    # GCDs 0-3 hold one chat model at a time. dd-web asks for the other one by
    # writing model.want; this supervisor swaps it and reports in model.state.
    # Inside the agent container this directory is /logs/model-control.
    MODEL_CONTROL=$DATA_DIR/agent_logs/model-control
    export REINVENT4_WEB_MODEL_CONTROL=/logs/model-control
    mkdir -p "$MODEL_CONTROL"
    echo "$CHAT_MODEL" > "$MODEL_CONTROL/model.want"

    state() { printf '{"model": "%s", "state": "%s", "job": "%s", "at": %s}\n' "$1" "$2" "$JOB" "$(date +%s)" > "$MODEL_CONTROL/model.state"; }

    supervise_vllm() {
        local want running="" pid=""
        while true; do
            want=$(cat "$MODEL_CONTROL/model.want" 2>/dev/null || echo qwen)
            if [[ $want != "$running" ]]; then
                [[ -n $pid ]] && { kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; }
                state "$want" loading
                VLLM_MODEL=$want VLLM_GPUS=0,1,2,3 bash "$REPO/services/lumi/vllm_serve.sh" \
                    > "$LOGS/vllm-$want.log" 2>&1 &
                pid=$!
                running=$want
            elif ! kill -0 "$pid" 2>/dev/null; then
                state "$running" failed
                running=""
                sleep 20
            elif curl -sf http://127.0.0.1:8001/health >/dev/null; then
                state "$running" ready
            fi
            sleep 5
        done
    }
    supervise_vllm &
    VLLM=$!
fi

bash "$REPO/services/singularity/run.sh" --vendor amd --web > "$LOGS/stack.log" 2>&1 &
STACK=$!

trap 'kill $STACK $VLLM 2>/dev/null; pkill -f "vllm serve" 2>/dev/null; wait' EXIT TERM

wait_for() {  # name url pid
    for _ in $(seq 1 240); do
        curl -sf "${AUTH[@]}" "$2" >/dev/null && { echo "$1 up at $(date +%T)"; return 0; }
        kill -0 "$3" 2>/dev/null || { echo "$1 exited early, see $LOGS"; return 1; }
        sleep 10
    done
    echo "$1 not up after 40 min, see $LOGS"; return 1
}
[[ -n $VLLM ]] && { wait_for vLLM http://127.0.0.1:8001/health "$VLLM" || exit 1; }
wait_for dd-web http://127.0.0.1:8000/api/health "$STACK" || exit 1

curl -s -o /dev/null -w "without token: HTTP %{http_code}\n" "http://$(hostname):8000/api/session"
echo "READY node=$(hostname)"
wait $STACK
