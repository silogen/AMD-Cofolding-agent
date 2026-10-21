#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ──────────────────────────────────────────────────────────────────────
# Vendor matrix for the Apptainer/Singularity build.
#
# Sourced by build.sh. Given a vendor ($1 = amd | nvidia), exports:
#   BASE_IMAGE       GPU pytorch base for dd-agent + boltz (torch from base).
#   TORCH_INDEX_URL  pip index for the admet torch wheel (torch==2.7.1).
#   PRED_DEVICE      admet GNN device (cuda works on ROCm too via HIP).
#
# Notes / parity:
#   - AMD keeps the exact ROCm base the Dockerfiles use today
#     (rocm7.0 / pytorch 2.8.0 / py3.12).
#   - NVIDIA uses the matching-torch CUDA base (pytorch 2.8.0, cuda12.8-devel);
#     verified to exist on Docker Hub. `-devel` is required because boltz's
#     `pip install -e .` may compile extensions.
#   - admet runs on CPU (as it does under Docker Compose today): it installs the
#     CPU torch wheel and PRED_DEVICE=cpu for BOTH vendors. run.sh starts admet
#     without a --rocm/--nv flag, so there is no GPU torch/ROCm-driver ABI risk.
# ──────────────────────────────────────────────────────────────────────

vendor_matrix() {
    local vendor="$1"
    case "${vendor}" in
        amd)
            export BASE_IMAGE="rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.8.0"
            # admet: CPU torch wheel + CPU device (matches docker-compose).
            export TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
            export PRED_DEVICE="cpu"
            ;;
        nvidia)
            export BASE_IMAGE="pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel"
            # admet: CPU torch wheel + CPU device (matches docker-compose).
            export TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
            export PRED_DEVICE="cpu"
            ;;
        *)
            echo "vendors.sh: unknown vendor '${vendor}' (expected: amd | nvidia)" >&2
            return 1
            ;;
    esac
}
