#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Open the UI against your running cofolding-session job on LUMI (macOS / Linux).
#
#   bash services/lumi/start-ui.sh              waits for the job to say READY
#   NODE=nid005001 bash services/lumi/start-ui.sh
#   DEV=1 bash services/lumi/start-ui.sh        live-reloading dev server
#   SSH_HOST=mylumi bash services/lumi/start-ui.sh   (default: lumi)
#
# Tunnels localhost:8000 through the login node to dd-web, then serves the UI
# on http://localhost:5173 with the access token added by Vite's proxy.
set -euo pipefail
SSH_HOST=${SSH_HOST:-lumi}
FRONTEND="$(cd "$(dirname "${BASH_SOURCE[0]}")/../ui/frontend" && pwd)"
NODE=${NODE:-}

while [[ -z $NODE ]]; do
    read -r id state node < <(ssh "$SSH_HOST" "squeue --me -h -n cofolding-session -o '%i %T %N'" | head -1) || true
    [[ -n ${id:-} ]] || { echo "No cofolding-session job. Submit services/lumi/session.sbatch first." >&2; exit 1; }
    if [[ $state == RUNNING ]] &&
        ssh "$SSH_HOST" "grep -q READY \$(scontrol show job $id | grep -oP 'StdOut=\K\S+')" 2>/dev/null; then
        NODE=$node
    else
        echo "$(date +%H:%M) job $id is $state, services still starting..."
        sleep 30
    fi
done

TOKEN=$(ssh "$SSH_HOST" 'cat ~/.dd-web-token')
[[ -n $TOKEN ]] || { echo "No ~/.dd-web-token on LUMI; session.sbatch writes it." >&2; exit 1; }
echo "Session node: $NODE"

ssh -N -o ExitOnForwardFailure=yes -L "8000:$NODE:8000" "$SSH_HOST" &
TUNNEL=$!
trap 'kill $TUNNEL 2>/dev/null' EXIT
sleep 4
kill -0 $TUNNEL 2>/dev/null || { echo "SSH tunnel failed (is local port 8000 in use?)" >&2; exit 1; }

cd "$FRONTEND"
[[ -d node_modules ]] || npm ci
export VITE_PROXY_TARGET=http://localhost:8000 VITE_PROXY_TOKEN=$TOKEN
if [[ ${DEV:-0} == 1 ]]; then
    npm run dev
else
    npx vite build
    npx vite preview --port 5173 --strictPort
fi
