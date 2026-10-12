#!/usr/bin/env bash
# Point one AIM tier at another catalog model. Implementation is
# switch-aim-model.py in this directory; it patches k8s/manifests/.
#
# --tier is required: there are two endpoints, a latency one for the conductor
# and a throughput one shared by the supervisors and the interactive agent.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! command -v python3 >/dev/null; then
    echo "python3 not found" >&2
    exit 1
fi
exec python3 "${DIR}/switch-aim-model.py" "$@"
