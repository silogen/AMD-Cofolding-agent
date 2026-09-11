# Reaching the AMD LLM Gateway from a Remote Server via SSH Tunnel

The remote server (e.g. `tw046`) cannot reach `llm-api.amd.com` directly, but
your **local machine** can. The workaround is an SSH reverse tunnel (`ssh -R` /
`RemoteForward`) from your laptop into the remote server, plus a small relay so
the Dockerized agent can reach that tunnel.

```
[container] --(llm-api.amd.com:9579 via extra_hosts host-gateway)-->
  host 172.17.0.1:9579 (relay) --> 127.0.0.1:9579 (ssh -R tunnel) -->
    [your laptop] --> llm-api.amd.com:443
```

Why the relay? SSH `RemoteForward` binds to the host **loopback** (`127.0.0.1`)
by default, and changing that needs `GatewayPorts` in the server's
`sshd_config` (admin-only). Docker compose containers run on a **bridge
network**, so their `127.0.0.1` is *not* the host's loopback. The relay bridges
the docker gateway IP to the loopback tunnel without touching `sshd_config`.

---

## How to SSH into the server?

When SSHing into the server, use an ad-hoc command:

```bash
ssh -R 127.0.0.1:9579:llm-api.amd.com:443 tw046
```

Notes:
- `9579` is an arbitrary free port; change it everywhere if it collides.
- The forward target `llm-api.amd.com:443` is dialed from your **laptop**, so
  your laptop must be on the network/VPN that can reach the gateway.
- The tunnel lives only as long as this SSH session stays open.

If you see `remote port forwarding failed for listen port 9579`, the port is
still held by a previous session. Find and clear it on the remote:

```bash
ss -ltnp 'sport = :9579'      # find the PID
# kill the stale sshd/PID, or pick a different port everywhere
```

---

## 0. One-time configuration for docker compose and environment

Once you move the files via `rsync`

Create the local environment file:

```bash
cd dd_agentic
cp services/env.sh services/env.local.sh
```

Edit `env.local.sh` with your values:


```bash
# Required: LLM API access
export LLM_GATEWAY_KEY="your-gateway-key-here"
export REINVENT4_AGENT_BASE_URL="https://llm-api.amd.com:9579/Anthropic" # or whichever port you have used
export REINVENT4_AGENT_NO_VERIFY_SSL="false"
```

The `docker-compose.yml` `reinvent4-mcp` service forwards `REINVENT4_AGENT_*`
into the container. The hostname mapping that reaches the tunnel lives in a
separate overlay, `docker-compose.tunnel.yml`, which adds
`extra_hosts: ["llm-api.amd.com:host-gateway"]` — every `docker compose`
command below therefore passes both files.

> Only use the overlay when the node really cannot reach the gateway. With it
> applied, `llm-api.amd.com` always resolves to the Docker host, so if the
> relay is not running every request fails with
> `[Errno 111] Connection refused` on port 443.

### Install docker compose plugin

1. Create the user-local CLI plugins dir

```bash
mkdir -p ~/.docker/cli-plugins
```

2. Detect arch and download the matching compose binary

```bash
ARCH=$(uname -m)   # x86_64 or aarch64
curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-${ARCH}" \
  -o ~/.docker/cli-plugins/docker-compose
```

3. Make it executable

```bash
chmod +x ~/.docker/cli-plugins/docker-compose
```

4. Verify

```bash
docker compose version
```

---

## 1. Slurm Allocation

```bash
salloc --reservation=MDL --partition=AIG_Models -w tw046 --gres=gpu:2 --cpus-per-task=8 --mem=64G --time=4:00:00
srun --jobid=$SLURM_JOB_ID --pty bash
```

---

## 2. Remote host: start the relay

`socat` is not installed and can't be (`apt` needs admin), so use the bundled
Python relay. Run it on the **same node where docker runs**.

```bash
ip addr show docker0                      # confirm bridge gateway, usually 172.17.0.1

python3 services/slurm/llm_tunnel_relay.py --bind 172.17.0.1 --port 9579 &
```

Expected output:

```
relay listening on 172.17.0.1:9579 -> 127.0.0.1:9579
```

Stop it later with `kill %1` (or `ps aux | grep llm_tunnel_relay`).

> If `socat` ever becomes available, the equivalent one-liner is:
> `socat TCP-LISTEN:9579,bind=172.17.0.1,fork,reuseaddr TCP:127.0.0.1:9579 &`

---

## 3. Run

Order matters: **SSH tunnel up → relay running → then compose.**

```bash
cd ~/dd_agentic
source services/env.local.sh
docker compose build
docker compose -f docker-compose.yml -f docker-compose.tunnel.yml \
    run --rm reinvent4-mcp
```

---

# Verification / Troubleshooting

## Verify

If the commands above are running the LLM gateway correctly and agent is responding, then the connection is established. If not, type `exit` to chat window and follow the commands:

On the host — both listeners should be present:

```bash
ss -ltnp 'sport = :9579'
# expect 172.17.0.1:9579 (relay) AND 127.0.0.1:9579 (ssh tunnel)
```

Inside the container:

```bash
docker compose -f docker-compose.yml -f docker-compose.tunnel.yml \
    run --rm --entrypoint bash reinvent4-mcp

getent hosts llm-api.amd.com        # should resolve to 172.17.0.1
                                    # (if not, restart the relay with --bind <that IP>)

curl -v https://llm-api.amd.com:9579/Anthropic/v1/models \
  -H "Ocp-Apim-Subscription-Key: $LLM_GATEWAY_KEY" \
  -H "user: levguner" \
  -H "anthropic-version: vertex-2023-10-16"
```

The AMD gateway requires all three headers (`Ocp-Apim-Subscription-Key`,
`user`, `anthropic-version`) — the agent sends them automatically. Any HTTP
response (even 404/401/405) means the tunnel + relay + TLS chain all work; the
agent's real endpoint is `POST /Anthropic/v1/messages`.

Full end-to-end check:

```bash
curl -v https://llm-api.amd.com:9579/Anthropic/v1/messages \
  -H "Ocp-Apim-Subscription-Key: $LLM_GATEWAY_KEY" \
  -H "user: NTID" \
  -H "anthropic-version: vertex-2023-10-16" \
  -H "content-type: application/json" \
  -d '{"model":"claude-sonnet-4.6","max_tokens":16,"messages":[{"role":"user","content":"ping"}]}'
```

---

## Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `remote port forwarding failed for listen port 9579` | Stale tunnel holds the port. `ss -ltnp 'sport = :9579'`, kill it, or use another port everywhere. |
| `curl: (60) ... certificate subject name ... '127.0.0.1'` | You connected by IP/`localhost`. Use the hostname `llm-api.amd.com` (matches the cert) — that's what `/etc/hosts` + `extra_hosts` are for. |
| Error still shows `https://llm-api.amd.com/Anthropic` (no `:9579`) | Env var not reaching the container. Ensure `env.local.sh` is sourced and `docker-compose.yml` forwards `REINVENT4_AGENT_BASE_URL`. |
| Connection refused / timeout from inside container | Relay not running, bound to the wrong IP, or tunnel down. Check `getent hosts llm-api.amd.com` matches the relay `--bind`, and that all listeners are up. |
| `Connection refused` on **port 443** while `curl` works on the host | The tunnel overlay is applied on a node that can reach the gateway directly, so the hostname resolves to the Docker host. Drop `-f docker-compose.tunnel.yml`. |
| Gateway rejects the `user` header | Container username isn't `levguner`. Check: `docker compose run --rm --entrypoint python reinvent4-mcp -c "import getpass; print(getpass.getuser())"`. |
| Works on login node but not compute node | Loopback is per-host. The SSH tunnel + relay must run on the **same node** that runs docker. Tunnel directly to that node if needed. |
