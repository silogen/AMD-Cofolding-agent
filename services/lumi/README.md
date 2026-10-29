<!--
Copyright Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Run the agent and its UI on LUMI

One LUMI-G node runs everything: a local chat model (vLLM), the agent
gateway, REINVENT, Boltz and the other scoring services. Your laptop serves
the UI and reaches the node through an SSH tunnel.

## You need

- A LUMI project with GPU hours (`standard-g`) and about 100 GB free on its
  scratch.
- SSH access, with an alias `lumi` in `~/.ssh/config`:

  ```
  Host lumi
      HostName lumi.csc.fi
      User <your LUMI username>
      IdentityFile ~/.ssh/<your key>
  ```

- On the laptop: git, and Node 20.19+ or 22.12+.

## 1. Set up once (on LUMI)

Clone onto scratch, not `$HOME`; the home quota is too small.

```bash
cd /scratch/<project>/$USER
git clone https://github.com/silogen/AMD-Cofolding-agent.git
cd AMD-Cofolding-agent
```

Restore the gitignored dependencies (REINVENT4, Boltz, PoseBusters and the
model weights archive) as described in
[Setup, step 1](../../README.md#1-restore-the-gitignored-dependencies).

Then build the images and download the chat model in one CPU job (about an
hour):

```bash
mkdir -p logs && sbatch --account=<project> services/lumi/build.sbatch
```

`logs/cofolding-build-<job>.out` lists each target as OK or FAILED.

## 2. Start a session (on LUMI)

```bash
sbatch --account=<project> services/lumi/session.sbatch
```

The job runs for 4 hours (`--time=` to change it). It is ready when
`logs/cofolding-session-<job>.out` says `READY`, after 10 to 20 minutes.

## 3. Open the UI (on the laptop)

Clone the same repository on the laptop, then from its root:

```powershell
.\services\lumi\start-ui.ps1        # Windows
```

```bash
bash services/lumi/start-ui.sh      # macOS, Linux
```

The script waits for `READY`, opens the tunnel, builds the UI and serves it
on <http://localhost:5173>. Behind a TLS-inspecting proxy, run `npm ci` once in
`services/ui/frontend` with `NODE_OPTIONS=--use-system-ca` first.

## Day to day

| To | Run on LUMI |
|----|-------------|
| Stop | `scancel <job>` |
| Restart only the gateway, after a code change | `srun --overlap --jobid=<job> pkill -f '[d]d-web --host'` |
| Use Gemma instead of Qwen | `sbatch --export=ALL,CHAT_MODEL=gemma ...` (build target `gemma` first, needs `HF_TOKEN`) |
| Use an external LLM endpoint | `sbatch --export=ALL,CHAT_MODEL=external ...` with `services/env.local.sh` set up ([Configure credentials](../../README.md#3-configure-credentials)) |
| Share model weights across a project | set `HF_HOME` to a project directory for both jobs |

A gateway restart ends the chat turn and any REINVENT run in progress, so
restart when nothing is running; queued runs start again afterwards.
Conversations, jobs and results are kept in `data/`; the logs of each
session are in `logs/session-<job>/`.

## Security

- The gateway listens on the node's network so the tunnel can reach it, which
  means every LUMI user can connect to it. It refuses any request without the
  access token in `~/.dd-web-token`, which `session.sbatch` creates readable by
  you only. The laptop's proxy adds the token; it never reaches the browser.
  Do not share it.
- The chat model listens on `127.0.0.1` only.
- The scoring services (ports 8082 to 8091) have no access control, so other
  LUMI users can reach them while the job runs.
- Keys for an external LLM go in `services/env.local.sh`, which git ignores.
