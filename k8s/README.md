# Kubernetes

This directory deploys the six Compose containers as one Kubernetes Deployment
plus an in-cluster AIM endpoint. Run the scripts from `examples/dd_agentic/`
(the Dockerfiles and gitignored build inputs live there).

```
k8s/
├── README.md                 # this file
├── apply.sh                  # kustomize overlay: namespace + nodeSelector
├── build-images.sh           # docker build of the six selected image tags
├── load-images.sh            # import those tags into kubelet containerd
├── rbac.yaml                 # developer grant for demo-lux on internal staging
├── manifests/                # kustomize resources (Deployment, AIMService, …)
└── scripts/
    ├── switch-aim-model.sh   # wrapper
    ├── switch-aim-model.py   # catalog lookup, AIMService recreate, ConfigMap
    └── uat-scoring.sh        # piped scoring prompts; checks tool score tables
```

The manifests in [`manifests/`](manifests) run the six Compose containers in one
Deployment. REINVENT and Boltz each request one `amd.com/gpu` (two GPUs on
that pod). AIM is a separate `AIMService` and can use **1 GPU** (the Qwen3-32B
example) or **8 GPUs** (a tensor-parallel catalog profile). The two workloads
do not have to share a node: pin the agent pod with `apply.sh --node`, and
leave AIM where `acceleratorCount` GPUs are free (often the other GPU node).
Images for the agent pod are imported onto **the agent node** — nothing is
pushed to a registry. AIM pulls its catalog image as usual.

The agent talks to AIM at `http://dd-agentic-aim/v1`. Set
`REINVENT4_AGENT_MODEL` to the served-model `id` from `/v1/models` (commonly
`Qwen/Qwen3-32B-FP8` when the AIM profile is fp8).

AIM is configured for **latency** and **fp8** on **1 GPU**. That matches
dd-agent: one interactive session, sequential Driver/Reviewer tool-call turns,
not a batched multi-tenant server. Do not switch the profile to throughput
unless several concurrent agent sessions share the same AIM replica. Keep
`precision: fp8` explicit — Qwen3-32B has latency and throughput profiles at
fp8/fp16/bf16; omitting precision can select a throughput or higher-precision
profile. `throughput` (or unset metric) is an environment-specific override,
not the default for this example.

### Configuration (not `env.local.sh`)

Do not `source services/env.local.sh` before `k8s/apply.sh`. That file is
only for Compose, Apptainer, and Slurm. The `dd-agent` container reads
`REINVENT4_AGENT_*` from a ConfigMap and Secret.

| `env.local.sh` | Kubernetes |
|----------------|------------|
| `REINVENT4_AGENT_BACKEND`, `MODEL`, `BASE_URL`, `MULTI_AGENT`, SSL, log dir | ConfigMap `dd-agentic-config` in [`manifests/configmap.yaml`](manifests/configmap.yaml) |
| `LLM_GATEWAY_KEY` / `REINVENT4_AGENT_API_KEY` | Secret `dd-agentic-llm` in [`manifests/llm-secret.yaml`](manifests/llm-secret.yaml) |
| Gateway URL (`https://llm-api.amd.com/...`) | In-cluster AIM: `http://dd-agentic-aim/v1` |

The example ConfigMap is:

```
REINVENT4_AGENT_BACKEND=openai
REINVENT4_AGENT_BASE_URL=http://dd-agentic-aim/v1
REINVENT4_AGENT_MODEL=Qwen/Qwen3-32B-FP8
```

`REINVENT4_AGENT_MODEL` must match the served `/v1/models` **id** (often
`Qwen/Qwen3-32B-FP8` when the AIM profile is fp8). The LLM is
`AIMService/dd-agentic-aim` ([`manifests/aim-service.yaml`](manifests/aim-service.yaml)),
not the AMD gateway. Do not set `LLM_GATEWAY_KEY` for AIM.

The in-cluster AIM endpoint typically needs no real key. The OpenAI client
still requires a non-empty value, so the Secret uses
`REINVENT4_AGENT_API_KEY: not-required`. Replace that only if AIM requires a
Bearer token.

### Prerequisites

- `kubectl` pointed at a cluster with AIM Engine (`aim.eai.amd.com/v1alpha2`).
- `docker` on the machine that will build images, and on the node that will
  import them (or another way to produce the same `*:local` tags there).
- An **existing** Kubernetes namespace. The manifests do not create one.
  On the internal staging environment, deploy into `demo-lux`. See
  [RBAC](#rbac) for the developer grant that covers that namespace.
- A Qwen3-32B `AIMClusterModel` (or another catalog model you will set in
  `manifests/aim-service.yaml`).
- A GPU node with at least two free `amd.com/gpu` for `deployment/dd-agentic`,
  with the six `*:local` images in that node's kubelet store. AIM needs
  `acceleratorCount` free GPUs on **one** node (1 for the Qwen3-32B example, 8
  for an 8-way profile). That can be a different node; see
  [Pin the agent pod and AIM](#pin-the-agent-pod-and-aim).
- Gitignored build inputs from
  [Restore the gitignored dependencies](../README.md#1-restore-the-gitignored-dependencies)
  (`REINVENT4/`, `boltz2/`, `models/admet/`, `models/retrodmta/`).

Pick the namespace and the GPU node that will run **the agent pod** (not
necessarily the AIM node):

```bash
cd examples/dd_agentic
export PROJECT="demo-lux"          # internal staging namespace; see RBAC
kubectl get namespace "${PROJECT}"
kubectl get nodes --output custom-columns=NAME:.metadata.name,GPU:.status.allocatable.amd\\.com/gpu
export NODE="<gpu-node-name>"   # Kubernetes node name for deployment/dd-agentic
```

### RBAC

[`rbac.yaml`](rbac.yaml) grants developers the permissions needed to deploy
the dd-agentic solution into the `demo-lux` namespace on the internal staging
environment. A cluster-admin applies it once. It is not listed in
`manifests/kustomization.yaml`, so `apply.sh` does not copy these objects
into the namespace you pass with `--namespace`:

```bash
kubectl apply --filename k8s/rbac.yaml
```

The file contains the Role, ClusterRole, RoleBinding, and ClusterRoleBinding.
Subjects are **Groups**, not Users: Keycloak group `demo-lux`, which the API
server maps to Kubernetes groups `oidc:demo-lux` and `oidcdemo-lux` (both are
bound). Add the developer to that Keycloak group. This grant does not use
Keycloak `account*` or `realm-management*` client roles, and it does not bind
a username.

Reuse the cluster's existing OIDC kubeconfig (the same `oidc-login` /
`kubelogin` exec plugin other project members use). After the apply, mint a
new token so the groups claim is current, then check:

```bash
kubectl auth can-i get nodes
kubectl auth can-i --namespace demo-lux create deployments
kubectl auth can-i --namespace demo-lux create aimservices.aim.eai.amd.com
```

All three should print `yes`. `load-images.sh` uses `ctr` on the GPU node
and is outside this RBAC.

The bindings grant:

- Role `dd-agentic-operator` in `demo-lux`: create, update, and delete
  ConfigMap, Secret, PVC, Service, and Deployment (including
  `deployments/scale` and rollout restart/status); Pods (exec, logs, wait);
  Endpoints; and AIMService. The Role exists only in `demo-lux`.
  `apply.sh --namespace` still accepts another namespace; this grant does not
  cover it.
- ClusterRole `dd-agentic-cluster`: `kubectl get namespace demo-lux`,
  `kubectl get nodes` (including GPU allocatable columns), `kubectl get
  aimclustermodel` / `aimclusterprofile`, StorageClass reads for the PVC, and
  cluster-wide Pod and Event list so `kubectl describe node` can show
  **Allocated resources**. Those APIs are cluster-scoped; a Role cannot grant
  them. AIM catalog types (`AIMClusterModel`, `AIMClusterProfile`) are
  cluster-scoped.

If an older `dd-agentic-catalog-reader`, `dd-agentic-namespace-extras`, or
ClusterRole `dd-agentic-operator` grant is still on the cluster, delete those
objects after this apply.

Set `spec.model.name` in `manifests/aim-service.yaml` to the **AIMClusterModel
object name** on that cluster (not a short alias):

```bash
kubectl get aimclustermodel | grep --ignore-case qwen3-32b
```

### Build locally, load onto the node, apply

Docker images are **not** visible to kubelet until they are imported into
the kubelet containerd store on the node that will run the pod (`k8s.io`
namespace). Run `load-images.sh` **on that node**. Do not import into Docker's
own containerd socket.

The example scripts default to the k3s/RKE2 kubelet socket
`/run/k3s/containerd/containerd.sock`. If your node uses a different address,
set `CONTAINERD_SOCK` before `load-images.sh`. Image tarballs are large; the
script writes them under `IMPORT_TMP` (default `$HOME/dd-agentic-import` —
override if that filesystem is too small).

`load-images.sh --node` must match this machine's `hostname`.

```bash
chmod +x k8s/build-images.sh k8s/load-images.sh k8s/apply.sh

./k8s/build-images.sh
./k8s/load-images.sh --namespace "${PROJECT}" --node "${NODE}"
./k8s/apply.sh --namespace "${PROJECT}" --node "${NODE}"
```

The default image tag is `local`. To use another tag, pass the same `--tag`
value to all three scripts.

`apply.sh` pins `deployment/dd-agentic` to `${NODE}` via
`nodeSelector[kubernetes.io/hostname]` (not `spec.nodeName`, which some
schedulers reject while scheduling gates are set) and
`imagePullPolicy: IfNotPresent`, so kubelet uses the imported `*:local` tags
and does not try a registry. Re-run `apply.sh` with a different `--node` to
move the agent pod. **AIM is not pinned** — see
[Pin the agent pod and AIM](#pin-the-agent-pod-and-aim).

Watch AIM separately (cache warm can take several minutes). Confirm the
`AIMService` object exists and the Service has endpoints before chatting:

```bash
kubectl --namespace "${PROJECT}" get aimservice dd-agentic-aim --watch
kubectl --namespace "${PROJECT}" get endpoints dd-agentic-aim
```

Confirm the served model id, then start the interactive agent:

```bash
kubectl --namespace "${PROJECT}" exec deployment/dd-agentic \
  --container dd-agent -- python -c \
  'import json,urllib.request; print(urllib.request.urlopen("http://dd-agentic-aim/v1/models").read().decode())'

kubectl --namespace "${PROJECT}" exec \
  --stdin --tty deployment/dd-agentic \
  --container dd-agent -- dd-agent
```

[`scripts/uat-scoring.sh`](scripts/uat-scoring.sh) sends two scoring prompts
through that same `dd-agent` command and checks the `score_molecules` tables:
aspirin QED `0.550` and molecular weight `180.159`, then ethanol and benzene
for QED, SA score, and SlogP. It omits `--tty` so the prompts can be piped,
and passes `--no-monitor` so a Reviewer cycle does not interleave. Defaults
are namespace `demo-lux`, deployment `dd-agentic`, container `dd-agent`, and
a 20-minute timeout. `--transcript` sets the log path; otherwise the script
writes a temp file and prints that path.

```bash
./k8s/scripts/uat-scoring.sh --namespace "${PROJECT}"
```

If `/v1/models` reports a different `id` than `REINVENT4_AGENT_MODEL` in
`manifests/configmap.yaml`, update the ConfigMap and re-run `apply.sh`, or use
[`scripts/switch-aim-model.sh`](scripts/switch-aim-model.sh) below.

### Pin the agent pod and AIM

`deployment/dd-agentic` and `AIMService/dd-agentic-aim` are scheduled
independently. The agent pod talks to AIM over the in-cluster Service
`http://dd-agentic-aim/v1` (any node). `AIMService` has **no** nodeSelector
in this API; the predictor lands where `spec.profile.selector.acceleratorCount`
GPUs are free on a **single** node.

| Workload | GPUs | How it is pinned |
|----------|------|------------------|
| `deployment/dd-agentic` | 2 (`dd-agent` + `boltz`) | `apply.sh --node` → `nodeSelector[kubernetes.io/hostname]` |
| `AIMService/dd-agentic-aim` | `acceleratorCount` (1 or 8 in the examples) | Scheduler only. Free that many GPUs on the node you want; do not leave that many free on the other GPU node |

**8-GPU AIM** (for example DeepSeek-V3.1-Terminus fp8 tp8) needs one node with
**8 free** `amd.com/gpu`. Put the agent pod on the **other** node, which needs
**2 free** GPUs and the `*:local` images. Two 8-GPU nodes can run this as 2+8
or 8+2; they cannot run 8-GPU AIM and the agent pod on the same node.

Check allocatable vs already requested:

```bash
kubectl get nodes --output custom-columns=NAME:.metadata.name,GPU:.status.allocatable.amd\\.com/gpu
kubectl describe node "${NODE}" | sed -n '/Allocated resources/,/Events/p'
```

Move **dd-agentic** to another node (images must already be in that node's
kubelet store; the `dd-agentic-data` PVC is `ReadWriteOnce`, so scale down
first and wait for Longhorn to detach):

```bash
export AGENT_NODE="<gpu-node-for-dd-agentic>"   # e.g. useocpm2m-silogen-007
kubectl --namespace "${PROJECT}" scale deployment/dd-agentic --replicas=0
kubectl --namespace "${PROJECT}" wait pod \
  --selector app.kubernetes.io/name=dd-agentic --for=delete --timeout=5m
./k8s/apply.sh --namespace "${PROJECT}" --node "${AGENT_NODE}"
```

`apply.sh` scales the rollout back up as it applies `replicas: 1` from the
manifest. Confirm:

```bash
kubectl --namespace "${PROJECT}" get pods --selector app.kubernetes.io/name=dd-agentic \
  --output wide
```

Place **AIM** by GPU packing, then create or switch the service. Scale the
agent pod onto `AGENT_NODE` first so the AIM node still has 8 free:

```bash
export AIM_NODE="<gpu-node-for-aim>"   # e.g. useocpm2m-silogen-017
# AIM_NODE must show 8 free amd.com/gpu (0 requested) for an 8-way profile.

./k8s/scripts/switch-aim-model.sh --namespace "${PROJECT}" \
    --model deepseek-ai/DeepSeek-V3.1-Terminus \
    --accelerator-count 8 --precision fp8
```

If AIM is already running on the wrong node, delete and recreate it after the
GPUs are packed (same as a model change; `switch-aim-model.sh` does that when
`spec.model` changes). Then:

```bash
kubectl --namespace "${PROJECT}" get pods --selector aim.eai.amd.com/service.name=dd-agentic-aim \
  --output wide
kubectl --namespace "${PROJECT}" get endpoints dd-agentic-aim
```

Do not `kubectl cordon` a GPU node unless you intend to drain every other
workload on it. Occupying GPUs with `dd-agentic` (2) or scaling unrelated
jobs down is enough for the 2+8 split.

### Rebuild the dd-agent image

`dd-agent:local` bakes `reinvent4_agent/` into the image. Docker and kubelet
do not share an image store. `load-images.sh` replaces an existing containerd
tag during import, and `apply.sh` restarts the Deployment after applying so
pods load the newly imported image. For a single-image rebuild, use the
focused workflow below because `load-images.sh` requires all six Compose
images to exist in Docker.

Either build **on the GPU node** (same host as `kubernetes.io/hostname` on
the Deployment) or build **elsewhere and copy the tar** onto that node.
Both paths end the same way: remove the old kubelet tag, import the tar,
restart the Deployment.

The tar is about 11 GiB. Write it on a filesystem with enough free space
that your user can create files on (not a tiny root `/tmp`). Confirm the
build log shows **not CACHED** for
`COPY reinvent4_agent/ /app/reinvent4_agent/` and the following
`pip install ... /app/reinvent4_agent`. A cache hit means the image still
has the previous agent. A BuildKit warning about undefined `$PYTHONPATH` on
the `ENV PYTHONPATH=...` line is harmless.

```bash
export PROJECT="demo-lux"
export NODE="<gpu-node-name>"
export CONTAINERD_SOCK="${CONTAINERD_SOCK:-/run/k3s/containerd/containerd.sock}"
export WORKDIR="${WORKDIR:-${HOME}/dd-agentic-import}"
```

#### On the GPU node

Build, save, and import on the same machine. `hostname` must match `NODE`.

```bash
cd examples/dd_agentic   # or the copy of this tree on that node
kubectl get namespace "${PROJECT}"
hostname

docker build --file services/dd-agent/Dockerfile --tag dd-agent:local .

sudo ctr --address "${CONTAINERD_SOCK}" --namespace k8s.io \
  images rm docker.io/library/dd-agent:local

mkdir --parents "${WORKDIR}"
docker save --output "${WORKDIR}/dd-agent.tar" dd-agent:local
sudo ctr --address "${CONTAINERD_SOCK}" --namespace k8s.io \
  images import "${WORKDIR}/dd-agent.tar"
rm --force "${WORKDIR}/dd-agent.tar"
```

If you already ran `docker build` and `ctr images rm`, start at `docker save`.
Import before the pod is rescheduled — kubelet has no `dd-agent:local` until
the import finishes.

Then [restart the Deployment](#restart-the-deployment-after-import).

#### From another machine (tar copy)

Build where Docker is convenient, copy only the tar to the GPU node, import
there. The build machine does not need `ctr` or kubelet.

On the build machine:

```bash
cd examples/dd_agentic
docker build --file services/dd-agent/Dockerfile --tag dd-agent:local .
mkdir --parents "${WORKDIR}"
docker save --output "${WORKDIR}/dd-agent.tar" dd-agent:local
```

Copy `${WORKDIR}/dd-agent.tar` onto the GPU node (scp, shared filesystem, or
similar) to a path with enough free space. Then on the GPU node:

```bash
hostname   # must match NODE
sudo ctr --address "${CONTAINERD_SOCK}" --namespace k8s.io \
  images rm docker.io/library/dd-agent:local
sudo ctr --address "${CONTAINERD_SOCK}" --namespace k8s.io \
  images import /path/to/dd-agent.tar
rm --force /path/to/dd-agent.tar
```

Then restart as below.

#### Restart the Deployment after import

If `spec.replicas` is `0`, scale back to `1` first — a restart of a scaled-down
Deployment schedules nothing:

```bash
kubectl --namespace "${PROJECT}" get deployment dd-agentic \
    --output jsonpath='{.spec.replicas}{"\n"}'
kubectl --namespace "${PROJECT}" scale deployment/dd-agentic --replicas=1
kubectl --namespace "${PROJECT}" rollout restart deployment/dd-agentic
kubectl --namespace "${PROJECT}" rollout status deployment/dd-agentic \
    --timeout=20m
```

### Switch the AIM model

Do not create a second AIM endpoint. Reuse `AIMService/dd-agentic-aim` so the
agent's `REINVENT4_AGENT_BASE_URL` stays valid.
[`scripts/switch-aim-model.sh`](scripts/switch-aim-model.sh) runs
[`scripts/switch-aim-model.py`](scripts/switch-aim-model.py) from the same directory:
it checks that an `AIMClusterProfile` exists for that object, GPU count, and
precision, updates `manifests/aim-service.yaml`, applies it, writes the served
`/v1/models` `id` into `REINVENT4_AGENT_MODEL`, and restarts
`deployment/dd-agentic`. Keep both helpers in `scripts/`; they patch files in `manifests/`.

`AIMService.spec.model` is immutable (`model selection is immutable after
creation`), so changing the model requires deleting and recreating the
`AIMService` — the script does that automatically. Its `cache-cleanup`
finalizer only removes **non-Ready** caches, so an already warm cache survives
and is reused. Changing only precision, GPU count, or metric on the same model
is patched in place instead, with a `force-rebind` annotation to break the
sticky profile binding. Pass `--no-recreate` to abort rather than delete.

`--model` may be an AIMID (`google/gemma-4-31B-it`) or the full `AIMClusterModel`
object name. If several versions share an AIMID, pass `--name` with the object
name from `kubectl get aimclustermodel`.

```bash
chmod +x k8s/scripts/switch-aim-model.sh k8s/scripts/switch-aim-model.py

# validate only
./k8s/scripts/switch-aim-model.sh --namespace "${PROJECT}" \
    --model google/gemma-4-31B-it \
    --accelerator-count 1 --precision bf16 --dry-run

./k8s/scripts/switch-aim-model.sh --namespace "${PROJECT}" \
    --model google/gemma-4-31B-it \
    --accelerator-count 1 --precision bf16

./k8s/scripts/switch-aim-model.sh --namespace "${PROJECT}" \
    --model openai/gpt-oss-120b \
    --accelerator-count 1 --precision fp4

./k8s/scripts/switch-aim-model.sh --namespace "${PROJECT}" \
    --name amdenterpriseai-aim-deepseek-ai-deepseek-v3-1-ter-0-11-1-bb879a \
    --model deepseek-ai/DeepSeek-V3.1-Terminus \
    --accelerator-count 8 --precision fp8   # needs 8 free GPUs on one node; see pin section
```

Precision must match a deployable profile for that model on **your** cluster.
The commands above are examples from one catalog (Gemma-4-31B **bf16**,
gpt-oss-120b **fp4**, DeepSeek-V3.1-Terminus **fp8** on 8 GPUs). On a mismatch
the script lists every profile the model actually publishes and exits without
touching the cluster. Cache warm can take a long time; `--timeout` defaults
to `45m`.

`--dry-run` still rewrites the local `manifests/aim-service.yaml` so you can inspect it,
then stops before apply, ConfigMap, and restart.

Health of the in-pod scorers:

```bash
kubectl --namespace "${PROJECT}" exec deployment/dd-agentic \
  --container dd-agent -- python -c \
  'import urllib.request; [print(url, urllib.request.urlopen(url).read().decode()) for url in ["http://127.0.0.1:8091/health", "http://127.0.0.1:8082/health", "http://127.0.0.1:8083/health", "http://127.0.0.1:8084/health", "http://127.0.0.1:8085/health"]]'
```

The `dd-agentic-data` PVC is `ReadWriteOnce`; the Deployment uses `Recreate`.

### Tool calls on vLLM-served models

vLLM decodes `function.arguments` from the JSON text OpenAI clients send into
a Python object before it renders the model's chat template. DeepSeek-V3.1's
own template concatenates that value onto a string, so a conversation that
contains one assistant tool call is rejected for the rest of the session:

```
HTTP 400 - can only concatenate str (not "dict") to str
```

The OpenAI backend detects that rejection and retries with the arguments
encoded as a JSON string, which decodes back to a string the template can
render, then keeps that encoding for the session. No configuration is needed,
and endpoints that do not re-decode the arguments are unaffected.
