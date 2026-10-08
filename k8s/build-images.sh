#!/usr/bin/env bash
# Build the eight dd_agentic images locally. No registry push.
#
# Must be run from examples/dd_agentic/ (or pass --project-dir).
# Requires REINVENT4/, boltz2/, models/admet/, and models/retrodmta/ as
# described in the example README.
set -euo pipefail

TAG=local
PROJECT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)
            TAG="$2"
            shift 2
            ;;
        --project-dir)
            PROJECT_DIR="$2"
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${PROJECT_DIR}" ]]; then
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${PROJECT_DIR}"

# REINVENT4/ is consumed by the reinvent-runner build (it used to be the
# agent's); boltz2/ by the boltz wrapper; models/ by the two predictors.
for required in REINVENT4 boltz2 models/admet models/retrodmta; do
    if [[ ! -e "${required}" ]]; then
        echo "missing ${PROJECT_DIR}/${required}" >&2
        echo "restore gitignored deps first (see README: Restore the gitignored dependencies)" >&2
        exit 1
    fi
done

echo "building images in ${PROJECT_DIR} with tag ${TAG}"

docker build --file services/dd-agent/Dockerfile \
    --tag "dd-agent:${TAG}" .
docker build --file services/boltz2/Dockerfile \
    --tag "boltz2-wrapper:${TAG}" .
docker build --file services/posebusters/Dockerfile \
    --tag "posebusters:${TAG}" services/posebusters
docker build --file services/admet-predictor/Dockerfile \
    --tag "admet-predictor:${TAG}" .
docker build --file services/retrodmta-predictor/Dockerfile \
    --tag "retrodmta-predictor:${TAG}" .
docker build --file services/interaction-profiler/Dockerfile \
    --tag "interaction-profiler:${TAG}" .
docker build --file services/boltz-gateway/Dockerfile \
    --tag "boltz-gateway:${TAG}" services/boltz-gateway
# Owns REINVENT4 + the Zenodo priors, so this is the slow one. Context is the
# repo root, where REINVENT4/ and reinvent4_mcp/ live.
docker build --file services/reinvent-runner/Dockerfile \
    --tag "reinvent-runner:${TAG}" .

echo "built:"
echo "  dd-agent:${TAG}"
echo "  boltz2-wrapper:${TAG}"
echo "  posebusters:${TAG}"
echo "  admet-predictor:${TAG}"
echo "  retrodmta-predictor:${TAG}"
echo "  interaction-profiler:${TAG}"
echo "  boltz-gateway:${TAG}"
echo "  reinvent-runner:${TAG}"
