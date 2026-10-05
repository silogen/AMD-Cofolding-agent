#!/usr/bin/env bash
# Score two fixed prompts through deployment/dd-agentic and require the
# tool-rendered score tables (not only the model's prose).
#
# dd-agent uses prompt_toolkit on a TTY and line input otherwise. This script
# omits --tty so the prompts can be piped. It also passes --no-monitor so a
# background Reviewer cycle cannot interleave with the two turns.
#
# A human session is still:
#   kubectl --namespace demo-lux exec --stdin --tty \
#     deployment/dd-agentic --container dd-agent -- dd-agent
set -euo pipefail

NAMESPACE="demo-lux"
DEPLOYMENT="dd-agentic"
CONTAINER="dd-agent"
TIMEOUT="20m"
TRANSCRIPT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --deployment)
            DEPLOYMENT="$2"
            shift 2
            ;;
        --container)
            CONTAINER="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --transcript)
            TRANSCRIPT="$2"
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${TRANSCRIPT}" ]]; then
    TRANSCRIPT="$(mktemp --tmpdir uat-scoring.XXXXXX.log)"
fi

PROMPT_ASPIRIN='Score aspirin (CC(=O)OC1=CC=CC=C1C(=O)O) for QED and molecular weight.'
PROMPT_PAIR='Score CCO and c1ccccc1 for QED, SA score, and LogP'

echo "writing transcript to ${TRANSCRIPT}"

set +e
timeout --foreground "${TIMEOUT}" kubectl \
    --namespace "${NAMESPACE}" \
    exec \
    --stdin \
    "deployment/${DEPLOYMENT}" \
    --container "${CONTAINER}" \
    -- env PYTHONUNBUFFERED=1 dd-agent --no-monitor \
    >"${TRANSCRIPT}" 2>&1 <<EOF
${PROMPT_ASPIRIN}
${PROMPT_PAIR}
/exit
EOF
status=$?
set -e

if [[ "${status}" -ne 0 ]]; then
    echo "dd-agent exited ${status}; transcript: ${TRANSCRIPT}" >&2
    exit "${status}"
fi

fail=0
require() {
    local needle="$1"
    if ! grep --fixed-strings --quiet -- "${needle}" "${TRANSCRIPT}"; then
        echo "missing from transcript: ${needle}" >&2
        fail=1
    fi
}

# Values are the score_molecules table, which is independent of model prose.
require "CC(=O)OC1=CC=CC=C1C(=O)O"
require "180.159"
require "0.550"
require "SAScore"
require "SlogP"
require $'CCO      │ 0.002 │        0.407 │        1.980 │       -0.001'
require $'c1ccccc1 │ 0.907 │        0.443 │        1.000 │        1.687'
require "Goodbye."

if [[ "${fail}" -ne 0 ]]; then
    echo "scoring UAT failed; transcript: ${TRANSCRIPT}" >&2
    exit 1
fi

echo "scoring UAT passed; transcript: ${TRANSCRIPT}"
