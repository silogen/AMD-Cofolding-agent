#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Entrypoint wrapper for the dd-agent container (Docker + Apptainer/Singularity).
#
# Its only job is to prepend a timestamp to every line the agent writes when the
# output is being *captured* (a redirected file, `docker logs`, a Slurm .out,
# ...) rather than shown in a live interactive terminal. Interactive TTY
# sessions are left untouched so the Rich UI / input prompt stay intact.
#
# When to stamp:
#   * stdout is NOT a TTY               -> timestamps ON   (captured logs)
#   * REINVENT4_AGENT_LOG_TIMESTAMPS    -> force ON (1/true/yes/on) or
#                                          OFF (0/false/no/off), overriding the
#                                          TTY autodetection above. This is
#                                          needed under docker-compose `tty: true`,
#                                          which allocates a PTY even when the
#                                          logs are captured (so autodetection
#                                          alone would skip stamping there).
set -euo pipefail

# Default command when invoked with no args (Docker sets CMD ["dd-agent"]; the
# Singularity runscript passes `dd-agent "$@"` explicitly).
[ "$#" -eq 0 ] && set -- dd-agent

_stamp() {
    if command -v ts >/dev/null 2>&1; then
        ts '[%Y-%m-%d %H:%M:%S]'
    elif command -v gawk >/dev/null 2>&1; then
        gawk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush() }'
    else
        while IFS= read -r _line; do
            printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${_line}"
        done
    fi
}

_want_ts=0
[ -t 1 ] || _want_ts=1
case "${REINVENT4_AGENT_LOG_TIMESTAMPS:-}" in
    1|true|yes|on)  _want_ts=1 ;;
    0|false|no|off) _want_ts=0 ;;
esac

if [ "${_want_ts}" -eq 1 ]; then
    exec "$@" > >(_stamp) 2> >(_stamp >&2)
else
    exec "$@"
fi
