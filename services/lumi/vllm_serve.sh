#!/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Serve the chat model with vLLM on a LUMI-G node. session.sbatch starts it.
#
#   VLLM_MODEL         qwen (Qwen/Qwen3.8-27B, default) or gemma (google/gemma-4-31B-it)
#   VLLM_GPUS          GCDs to use (default 0,1,2,3; 4-7 are left for REINVENT and Boltz)
#   VLLM_PORT          OpenAI-compatible endpoint on the node (default 8001; dd-web uses 8000)
#   VLLM_SIF           vLLM image (default services/lumi/vllm.sif, made by build.sbatch)
#   HF_HOME            model weights (default data/hf-cache, filled by build.sbatch)
#   VLLM_LONG_CONTEXT  Qwen only: 1 (default) for 1M context through YaRN, 0 for the native 262,144
#   VLLM_SPEC_TOKENS   Qwen only: tokens drafted per step by its MTP head (default 2, 0 = off)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DATA_DIR=${DATA_DIR:-$REPO/data}
SIF=${VLLM_SIF:-$HERE/vllm.sif}
WHICH=${VLLM_MODEL:-qwen}
GPUS=${VLLM_GPUS:-0,1,2,3}
PORT=${VLLM_PORT:-8001}
TP=$(awk -F, '{print NF}' <<<"$GPUS")
SPEC=${VLLM_SPEC_TOKENS:-2}

case "$WHICH" in
qwen) MODEL=Qwen/Qwen3.8-27B ;;
gemma) MODEL=google/gemma-4-31B-it ;;
*)
    echo "VLLM_MODEL must be qwen or gemma, got '$WHICH'" >&2
    exit 2
    ;;
esac

# Decoding and memory are the same for both models so answers feel the same either way.
ARGS=(
    --host 127.0.0.1 --port "$PORT"
    --served-model-name "$MODEL"
    --tensor-parallel-size "$TP"
    --dtype bfloat16
    --gpu-memory-utilization 0.90
    --max-num-seqs 8
    --enable-prefix-caching
    --enable-auto-tool-choice
    --limit-mm-per-prompt '{"image": 0, "video": 0}'
)

if [[ "$WHICH" == qwen ]]; then
    ARGS+=(--tool-call-parser qwen3_coder --reasoning-parser qwen3)
    if [[ "${VLLM_LONG_CONTEXT:-1}" == 1 ]]; then
        export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
        ARGS+=(
            --max-model-len 1000000
            --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'
        )
    else
        ARGS+=(--max-model-len 262144)
    fi
    (( SPEC > 0 )) && ARGS+=(--speculative-config "{\"method\": \"qwen3_5_mtp\", \"num_speculative_tokens\": $SPEC}")
else
    # 262,144 is Gemma 4's native context; the -it checkpoint ships no MTP head, so no drafting.
    ARGS+=(--tool-call-parser gemma4 --reasoning-parser gemma4 --max-model-len 262144)
fi

TMP=$DATA_DIR/tmp
# Gemma's RMS norm goes through AITER, whose JIT copies itself into $HOME on first use;
# the tensor-parallel workers race that copy and the engine dies. Seed one copy first.
export AITER_JIT_DIR=$TMP/aiter-jit
if [[ ! -f $AITER_JIT_DIR/core.py ]]; then
    mkdir -p "$AITER_JIT_DIR"
    # A few root-owned cache blobs in the image are unreadable; AITER rebuilds those.
    singularity exec -B /pfs,/scratch "$SIF" \
        bash -c "cp -r /usr/local/lib/python3.12/dist-packages/aiter/jit/. '$AITER_JIT_DIR/' 2>/dev/null; true"
fi

export HF_HOME=${HF_HOME:-$DATA_DIR/hf-cache}
export HF_HUB_OFFLINE=1
export HIP_VISIBLE_DEVICES=$GPUS
export VLLM_CACHE_ROOT=$TMP/vllm-cache
export TRITON_CACHE_DIR=$TMP/triton-cache
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"

exec singularity exec -B /pfs,/scratch,/project,/appl "$SIF" \
    vllm serve "$MODEL" "${ARGS[@]}"
