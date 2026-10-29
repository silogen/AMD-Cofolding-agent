#!/bin/bash
#SBATCH --job-name=cofolding-build
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=224G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# One-time setup on a LUMI-C node, where /tmp is in RAM and builds are fast.
# Submit from the repository root (see services/lumi/README.md):
#   mkdir -p logs && sbatch --account=<project> services/lumi/build.sbatch [target ...]
#
# Targets (default: all of them):
#   posebusters interaction-profiler retrodmta-predictor admet-predictor boltz dd-agent
#                 the stack images, into services/singularity/*.sif
#   vllm          the vLLM image, into services/lumi/vllm.sif
#   qwen          Qwen/Qwen3.8-27B weights, into $HF_HOME (default data/hf-cache)
#   gemma         google/gemma-4-31B-it weights; gated, needs HF_TOKEN in the environment

set -uo pipefail
REPO=${REPO:-$SLURM_SUBMIT_DIR}
[[ -f $REPO/services/singularity/build.sh ]] || { echo "Submit from the repository root (got $REPO)"; exit 1; }
STACK="posebusters interaction-profiler retrodmta-predictor admet-predictor boltz dd-agent"
TARGETS=${*:-"$STACK vllm qwen"}
VLLM_IMAGE=${VLLM_IMAGE:-docker://vllm/vllm-openai-rocm:qwen38}
VLLM_SIF=$REPO/services/lumi/vllm.sif
export HF_HOME=${HF_HOME:-$REPO/data/hf-cache}

module load CrayEnv PRoot
export APPTAINER_BUILD_FLAGS="--force"
export SINGULARITY_TMPDIR=/tmp/$USER SINGULARITY_CACHEDIR=/tmp/$USER/cache
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR" "$REPO/logs/build" "$HF_HOME"

cd "$REPO" || exit 1
# Scratch files are created 660, and a PRoot build keeps that mode while the
# owner becomes root, so the container user could not read its own code.
# LUMI project directories are 2770, so this exposes nothing outside the project.
find "$REPO" -path "$REPO/data" -prune -o -path "$REPO/logs" -prune -o -exec chmod o+rX {} +

download() {  # repo id
    singularity exec -B /pfs,/scratch "$VLLM_SIF" python3 -c \
        "from huggingface_hub import snapshot_download; print(snapshot_download('$1', max_workers=8))"
}

failed=0
for target in $TARGETS; do
    echo "=== $target  $(date +%T)"
    log=$REPO/logs/build/$target.log
    case $target in
    vllm)
        # Copied rather than moved, so the image takes the project group and stays shareable.
        ok() { singularity pull --disable-cache "/tmp/$USER/vllm.sif" "$VLLM_IMAGE" && cp "/tmp/$USER/vllm.sif" "$VLLM_SIF" && rm "/tmp/$USER/vllm.sif"; } ;;
    qwen) ok() { download Qwen/Qwen3.8-27B; } ;;
    gemma) ok() { download google/gemma-4-31B-it; } ;;
    *) ok() { bash services/singularity/build.sh amd "$target"; } ;;
    esac
    if ok > "$log" 2>&1; then
        echo "    OK  $(date +%T)"
    else
        failed=1
        echo "    FAILED (see logs/build/$target.log):"
        tail -15 "$log" | sed 's/^/      /'
    fi
    rm -rf "$SINGULARITY_TMPDIR"/build-temp-* "$SINGULARITY_TMPDIR"/bundle-temp-*
done
ls -la services/singularity/*.sif "$VLLM_SIF" 2>/dev/null
exit $failed
