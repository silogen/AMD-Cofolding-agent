# Kubernetes

This directory deploys the Compose stack as one Kubernetes Deployment per
container, plus the fleet control plane and an in-cluster AIM endpoint. Run
the scripts from `examples/dd_agentic/` (the Dockerfiles and gitignored build
inputs live there).

The manifests carry no architecture constraint and there is no per-arch
variant: a `*:local` tag resolves to whatever was imported into that node's
containerd, so the node's architecture is settled by where the images were
built. Building on the node itself -- the documented path below -- makes the
question moot. Building on an arm64 Mac does not: the base images
(`python:3.1x-slim`, `mambaorg/micromamba`) are multi-arch, so the build
silently produces arm64, containerd imports it without complaint, and the pods
then crash-loop with `exec format error`. See "Building off-node" below.

```
k8s/
├── README.md                 # this file
├── apply.sh                  # kustomize overlay: namespace + nodeSelector
├── build-images.sh           # docker build of the twelve image tags
├── load-images.sh            # import those tags into kubelet containerd
├── rbac.yaml                 # developer grant for demo-lux on internal staging
├── manifests/                # kustomize resources (Deployments, AIMService, …)
│   └── scorers/              # one Deployment + ClusterIP Service per scorer
└── scripts/
    ├── switch-aim-model.sh   # wrapper
    ├── switch-aim-model.py   # catalog lookup, AIMService recreate, ConfigMap
    ├── uat-scoring.sh        # piped scoring prompts; checks tool score tables
    ├── analyze-scoring-latency.py  # harvest scorer latency + cache hit rate
    ├── ab-boltz-kernels.py   # A/B the optimized Boltz kernels on one /score call
    ├── verify-artifact-handoff.py  # Boltz writes in pod A, the others read in pod B
    └── verify-boltz-placement.py   # concurrent batches land on idle replicas
```

The manifests in [`manifests/`](manifests) run each container as its own
Deployment:

| Deployment | Service | GPU | `/shared` |
|------------|---------|-----|-----------|
| `dd-agentic` | — (reached by `kubectl exec`) | no | yes |
| `dd-agentic-boltz` | `dd-agentic-boltz:8091` (headless) | 1 each | yes |
| `dd-agentic-boltz-proxy` | `dd-agentic-boltz-proxy:8091` (internal) | no | no |
| `dd-agentic-boltz-gateway` | `dd-agentic-boltz-gateway:8091` | no | yes |
| `dd-agentic-reinvent-runner` | `dd-agentic-reinvent-runner:8086` | 1 | yes |
| `dd-agentic-campaign-store` | `dd-agentic-campaign-store:8087` | no | yes |
| `dd-agentic-supervisor` | — (a client, nothing dials it) | no | yes |
| `dd-agentic-conductor` | — (reached by `kubectl exec`) | no | read-only |
| `dd-agentic-posebusters` | `dd-agentic-posebusters:8082` | no | yes |
| `dd-agentic-interaction-profiler` | `dd-agentic-interaction-profiler:8083` | no | yes |
| `dd-agentic-admet` | `dd-agentic-admet:8084` | no | no |
| `dd-agentic-retrodmta` | `dd-agentic-retrodmta:8085` | no | no |
| `dd-agentic-synthesis-planner` | `dd-agentic-synthesis-planner:8088` | no | no |
| `dd-agentic-structure-prep` | `dd-agentic-structure-prep:8089` | no | yes |
| `dd-agentic-annotation` | `dd-agentic-annotation:8090` | no | yes |

The last two are tool services rather than scorers: the agents call them
between campaigns, not the scoring pipeline during one, so they carry no
`service_url`. They are reached through `SP_SERVICE_URL` and
`ANNOTATION_SERVICE_URL` in the ConfigMap, which the agent, supervisor and
conductor all read. Unset, the structure-prep tools fall back to the Compose
hostname `http://structure-prep:8080`, which does not resolve here, and the
annotation fetchers go straight to UniProt instead of the cache -- both of
which look like a broken tool rather than an unwired one.

`dd-agentic-synthesis-planner` mounts the building-block catalog read-only
from the volume's `aizynth-stock` subPath rather than from the image, which is
why it is the one scorer with a volume but no `/shared`. The catalog is 1-10 GB
and is staged separately; without it the service still starts, and
availability checks answer `unknown` rather than failing.

The `service_url`s in [`manifests/configmap.yaml`](manifests/configmap.yaml)
are those Service names, unqualified so they resolve in whichever namespace
`apply.sh --namespace` deploys into. They were `http://127.0.0.1:<port>` while
everything shared one pod.

GPU demand on this node is **one per Boltz replica plus one for the runner**,
so the shipped `replicas: 4` on `dd-agentic-boltz` needs five free
`amd.com/gpu` here; the two AIM tiers cost one each on the *other* node. The
agent holds none: REINVENT4 moved to `dd-agentic-reinvent-runner`, which is what
lets agent count and RL capacity scale independently. No GPU workload may surge
during a rollout, since the extra pod would ask for another GPU and stay
`Pending`: the runner uses `strategy: Recreate`, and `dd-agentic-boltz` uses a
rolling update pinned to `maxSurge: 0`, which replaces one replica at a time and
keeps the others serving.

Scoring traffic goes to **`dd-agentic-boltz-gateway`**, not to the replicas.
See [Place batches on idle Boltz replicas](#place-batches-on-idle-boltz-replicas).

AIM is **two** `AIMService`s, one per role, because the conductor and the
supervisors want opposite things from an LLM:

| Tier | Model | Profile | Who dials it |
|------|-------|---------|--------------|
| `dd-agentic-aim-conductor` | `openai/gpt-oss-120b` | MI300X ×1, fp4, **latency** | `dd-agentic-conductor` |
| `dd-agentic-aim-supervisor` | `Qwen/Qwen3-32B` | MI300X ×1, fp8, **throughput** | `dd-agentic-supervisor`, `dd-agentic` |

The conductor takes one turn at a time with a person waiting on it; the
supervisors are N agents grinding through tool calls with nobody watching any
single response. Serving both from one latency-tuned endpoint is what this
README already warned against for concurrent sessions — and N supervisors are
exactly that.

**Both tiers are single-GPU profiles**, which is what makes the split
affordable. A tensor-parallel conductor model (8 GPUs) would consume a whole
node and leave nothing for Boltz. Each tier costs 1 GPU, so the split costs
one GPU more than the single endpoint did — budget it against §3.4, which
allocates 1 per tier against **5 for Boltz**, since Boltz is the measured
bottleneck (both replicas were observed at 100% while a third job waited).

AIM does not have to share a node with the rest: `apply.sh --node` pins every
Deployment, and AIM lands wherever `acceleratorCount` GPUs are free (often the
other GPU node). Images for the Deployments are imported onto **that node** —
nothing is pushed to a registry. AIM pulls its catalog image as usual.

#### Keeping AIM off the Boltz node — and why the order you apply in matters

The layout this deployment wants is AIM on one node and the GPU-hungry
scorers on the other:

| Node | Workloads | GPUs |
|------|-----------|------|
| `useocpm2m-silogen-017` | `aim-conductor`, `aim-supervisor` | 1 each |
| `useocpm2m-silogen-007` | `boltz` ×M, `reinvent-runner` | 1 each |

The second row is **enforced**. `apply.sh --node` stamps
`nodeSelector[kubernetes.io/hostname]` onto every Deployment, and those
workloads have to be together anyway because the `dd-agentic-data` claim is
`ReadWriteOnce` and attached to that node.

The first row is **not enforced, and cannot be.** An `AIMService` has no
placement field: its `template` takes only `name`/`allowUnoptimized` and its
`overrides` only `hardware`/`metric`/`precision`. Neither
`AIMClusterRuntimeConfig` nor `AIMClusterServiceTemplate` has one either. And
a `nodeSelector` patched onto the predictor Deployment that AIM generates is
**stripped by the operator within about 20 seconds**. Where both nodes are the
same accelerator model, the profile's hardware requirement cannot separate
them either.

What places AIM is therefore the scheduler, which means **capacity at the
moment the AIMService is created**. So when migrating to this layout, create
the new tiers *before* deleting the old AIMService:

```bash
# 1. With the Boltz node full, a new AIMService cannot fit there and must
#    land on the other node.
kubectl --namespace "${PROJECT}" apply -f manifests/aim-service.yaml
kubectl --namespace "${PROJECT}" get aimservice --watch   # ~10 min each

# 2. Only now release the old endpoint's GPU.
kubectl --namespace "${PROJECT}" delete aimservice dd-agentic-aim
kubectl --namespace "${PROJECT}" delete service dd-agentic-aim

# 3. The freed GPU goes to the bottleneck.
kubectl --namespace "${PROJECT}" scale deployment/dd-agentic-boltz --replicas=3
```

Done in the other order, deleting first leaves a free GPU on the Boltz node
and a new tier may land there — taking a GPU that Boltz wants and putting the
LLM where it was not meant to be.

**The consequence to remember:** because nothing enforces it, an AIM tier
recreated while the Boltz node has a spare GPU may settle on the wrong node,
and the only symptom is Boltz quietly having one fewer replica available to
scale into. This is not hypothetical — raising the conductor's context
(below) recreated its predictor at a moment when this node had exactly one
GPU free, and it landed here. Check with:

```bash
kubectl --namespace "${PROJECT}" get pods \
  --selector app.kubernetes.io/component=inference \
  --output custom-columns='NAME:.metadata.name,NODE:.spec.nodeName'
```

Making it durable needs cluster-admin: either a Kyverno mutation injecting the
`nodeSelector` into AIM predictor pods, or a taint on the Boltz node that the
pinned Deployments tolerate and AIM does not.

#### Moving an AIM tier that landed on the wrong node

`kubectl rollout restart` on the generated predictor Deployment does **not**
work, even though its `maxUnavailable` resolves to 0 and the surge pod would
have had to go elsewhere. The operator strips the `restartedAt` annotation the
same way it strips `nodeSelector`, and no new pod is ever created. The
AIMService is the only lever.

Deleting and recreating it frees the GPU, so something must be **waiting** to
claim it or the new tier lands right back here. A pending Boltz replica is the
right claimant: it carries the `nodeSelector` for this node, so it can go
nowhere else and is first in line the moment the GPU frees.

```bash
# 1. Queue the claim. This pod stays Pending with
#    "Insufficient amd.com/gpu" -- that is the point.
kubectl --namespace "${PROJECT}" scale deployment/dd-agentic-boltz --replicas=4

# 2. Release the GPU. The pending replica takes it within seconds.
kubectl --namespace "${PROJECT}" delete aimservice dd-agentic-aim-conductor

# 3. Confirm this node is full again *before* recreating the tier.
kubectl --namespace "${PROJECT}" get pods --all-namespaces \
  --field-selector spec.nodeName=useocpm2m-silogen-007

# 4. With nowhere else to go, the tier lands on the AIM node (~3-4 min).
kubectl --namespace "${PROJECT}" apply -f manifests/aim-service.yaml
```

Keep the stable `Service` out of step 2: it selects on labels, so it picks up
the new predictor pod on its own and the agents' base URL never changes. The
conductor is down from step 2 until the new predictor reports `Running`.

Each tier has its own ConfigMap keys, so one `kubectl get configmap` shows
which model each is on: `CONDUCTOR_AGENT_MODEL`/`CONDUCTOR_AGENT_BASE_URL` and
`REINVENT4_AGENT_MODEL`/`REINVENT4_AGENT_BASE_URL`. Set the model to the
served-model `id` from that tier's `/v1/models` (commonly `Qwen/Qwen3-32B-FP8`
when the profile is fp8) — `switch-aim-model.sh --tier` does this for you.

AIM is configured for **latency** and **fp8** on **1 GPU**. That matches
dd-agent: one interactive session, sequential Driver/Reviewer tool-call turns,
not a batched multi-tenant server. Do not switch the profile to throughput
unless several concurrent agent sessions share the same AIM replica. Keep
`precision: fp8` explicit — Qwen3-32B has latency and throughput profiles at
fp8/fp16/bf16; omitting precision can select a throughput or higher-precision
profile. `throughput` (or unset metric) is an environment-specific override,
not the default for this example.

### Context size

A discovered profile's `max-model-len` is a **serving choice, not the model's
limit**, and both tiers override it:

| tier | model's ceiling | profile ships | deployed | how |
|------|-----------------|---------------|----------|-----|
| conductor | 131072 | 32768 | **131072** | `max-model-len` alone |
| supervisor | 32768 | 32768 | **65536** | `max-model-len` + YaRN `hf-overrides` |

The asymmetry is the point. gpt-oss-120b declares
`max_position_embeddings: 131072` and carries its own YaRN scaling in
`config.json` (factor 32 over an original 4096), so raising the arg is all it
takes. Qwen3-32B is *natively* 32768, so exceeding that needs RoPE scaling
supplied alongside — which is why only the supervisor tier has
`hf-overrides`. Adding them to the conductor would fight the shipped weights
rather than extend them.

The conductor's 32768 was not enough in practice: a Pareto front with the
molecules behind each point sent ~42k tokens of tool results, and since vLLM
derives `max_tokens` as `max-model-len` minus the prompt, the request failed
with `max_tokens must be at least 1, got -72257`. Those payloads were
narrowed to ~8k, so the bump is headroom rather than the fix.

**It costs concurrency, not memory.** vLLM pre-allocates the KV pool at
`gpu-memory-utilization` (0.95 here, so the GPU reads ~196 of 206 GB at any
context length); raising the limit only changes how much of that pool one
sequence may take. For gpt-oss-120b that is 36 KiB/token — low because 18 of
its 36 layers use a 128-token sliding window, leaving 18 full-attention
layers — so 1.21 GB per sequence at 32768 against 4.83 GB at 131072. A single
interactive conductor can spend batch slots on context; **do not copy the
bump to the supervisor tier**, where N agents need the slots.

Check what is actually served, which is the only number that matters:

```bash
kubectl --namespace "${PROJECT}" exec deployment/dd-agentic-conductor \
  --container dd-conductor -- python -c \
  'import json,urllib.request; print(json.load(urllib.request.urlopen(
   "http://dd-agentic-aim-conductor/v1/models"))["data"][0]["max_model_len"])'
```

**What cannot be changed this way.** `profileOverrides.engineArgs`
shallow-merges and can only replace *matching* keys — there is no way to
express "drop this one". The conductor profile disables prefix caching with
the negative flag `no-enable-prefix-caching`, so adding `enable-prefix-caching`
beside it just renders both, and the args are emitted alphabetically, which
puts the negative last and wins. Enabling it needs a forked profile, and
`aimclusterprofiles` is cluster-scoped and admin-only. Worth asking an admin
for: with caching off, every REPL turn re-prefills the whole transcript, which
a longer context makes more expensive, and an interactive conductor's prefix
is stable between turns.

### Configuration (not `env.local.sh`)

Do not `source services/env.local.sh` before `k8s/apply.sh`. That file is
only for Compose, Apptainer, and Slurm. The `dd-agent` container reads
`REINVENT4_AGENT_*` from a ConfigMap and Secret.

| `env.local.sh` | Kubernetes |
|----------------|------------|
| `REINVENT4_AGENT_BACKEND`, `MODEL`, `BASE_URL`, `MULTI_AGENT`, SSL, log dir | ConfigMap `dd-agentic-config` in [`manifests/configmap.yaml`](manifests/configmap.yaml) |
| `LLM_GATEWAY_KEY` / `REINVENT4_AGENT_API_KEY` | Secret `dd-agentic-llm` in [`manifests/llm-secret.yaml`](manifests/llm-secret.yaml) |
| Gateway URL (`https://llm-api.amd.com/...`) | In-cluster AIM: `http://dd-agentic-aim-supervisor/v1` (or `-conductor`) |

The example ConfigMap is:

```
REINVENT4_AGENT_BACKEND=openai
REINVENT4_AGENT_BASE_URL=http://dd-agentic-aim-supervisor/v1
REINVENT4_AGENT_MODEL=Qwen/Qwen3-32B-FP8
```

`REINVENT4_AGENT_MODEL` must match the served `/v1/models` **id** (often
`Qwen/Qwen3-32B-FP8` when the AIM profile is fp8). The LLM is
the two `AIMService`s ([`manifests/aim-service.yaml`](manifests/aim-service.yaml)),
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
- A GPU node with enough free `amd.com/gpu` for the GPU workloads: one per
  `dd-agentic-boltz` replica (4 by default) plus one for
  `dd-agentic-reinvent-runner`, so **five** as shipped. The agent itself needs
  none any more. All the `*:local` images must be in that node's kubelet store.
  AIM needs `acceleratorCount` free GPUs on **one** node (1 for the Qwen3-32B
  example, 8 for an 8-way profile). That can be a different node; see
  [Pin the Deployments and AIM](#pin-the-deployments-and-aim).
- Gitignored build inputs from
  [Restore the gitignored dependencies](../README.md#1-restore-the-gitignored-dependencies)
  (`REINVENT4/`, `boltz2/`, `models/admet/`, `models/retrodmta/`).

Pick the namespace and the GPU node that will run **the agent and scorer
pods** (not necessarily the AIM node):

```bash
cd examples/dd_agentic
export PROJECT="demo-lux"          # internal staging namespace; see RBAC
kubectl get namespace "${PROJECT}"
kubectl get nodes --output custom-columns=NAME:.metadata.name,GPU:.status.allocatable.amd\\.com/gpu
export NODE="<gpu-node-name>"   # Kubernetes node name for the dd-agentic pods
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

### Building off-node

`build-images.sh` assumes it runs where the images will run, so it sets no
`--platform`. On an arm64 Mac that produces arm64 images, which import cleanly
and then fail at runtime with `exec format error` -- there is no manifest-level
guard against this, because the tag is resolved by containerd, not by the
manifest. Two further things bite on an AMD laptop: Zscaler's TLS interception
breaks the `conda.anaconda.org` fetches that `structure-prep` and
`synthesis-planner` depend on, and `reinvent-runner` needs the ROCm base image.

Build on the GPU node where possible. Where it is not, `build-images.local.sh`
(gitignored, since it encodes one machine's proxy) forces
`--platform linux/amd64`, injects the corporate CA into the Dockerfiles that
need it, and verifies every tag is `amd64` before you import. Check the
architecture of what you are about to ship either way:

```bash
docker image inspect --format '{{.Architecture}}  {{.RepoTags}}' \
    $(docker images --filter 'reference=*:local' --format '{{.Repository}}:{{.Tag}}')
```

`apply.sh` pins **every** dd-agentic Deployment to `${NODE}` via
`nodeSelector[kubernetes.io/hostname]` (not `spec.nodeName`, which some
schedulers reject while scheduling gates are set) and
`imagePullPolicy: IfNotPresent`, so kubelet uses the imported `*:local` tags
and does not try a registry. It then restarts and waits for all of them by the
`app.kubernetes.io/part-of=dd-agentic` label, so a new Deployment is picked up
without editing the script. Re-run `apply.sh` with a different `--node` to move
the pods. **AIM is not pinned** — see
[Pin the Deployments and AIM](#pin-the-deployments-and-aim).

Two separate things make that pin necessary, and only one of them goes away on
a multi-node cluster:

- **Images are node-local.** `load-images.sh` imports the `*:local` tags
  into one node's kubelet store, so every pod using them must land there.
- **`/shared` is node-local.** Boltz writes structures to `/shared` and returns
  their absolute paths; PoseBusters and the interaction profiler then read
  those paths. On a single-node cluster `dd-agentic-data` stays
  `ReadWriteOnce`, which is per *node*, not per pod — any number of pods on the
  node may mount it read-write, but a pod elsewhere could not. Kyverno mutates
  the claim to `ReadWriteMany` on multi-node clusters, where those three may
  spread. Never switch the claim to `ReadWriteOncePod`: that one really is
  single-pod and would admit only one of them.

Watch AIM separately (cache warm can take several minutes). Confirm the
`AIMService` object exists and the Service has endpoints before chatting:

```bash
kubectl --namespace "${PROJECT}" get aimservice --watch   # both tiers
kubectl --namespace "${PROJECT}" get endpoints dd-agentic-aim-conductor dd-agentic-aim-supervisor
```

Confirm the served model id, then start the interactive agent:

```bash
kubectl --namespace "${PROJECT}" exec deployment/dd-agentic \
  --container dd-agent -- python -c \
  'import json,urllib.request; print(urllib.request.urlopen("http://dd-agentic-aim-supervisor/v1/models").read().decode())'

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

### Measure scorer latency and cache hit rate

`ExternalRESTScorer` appends one JSON record per external call to
`/data/reinvent4_mcp/logs/external_scoring.jsonl` on the PVC, carrying
`elapsed_seconds`, the service name, the status code, and the request — so the
SMILES list makes batch size recoverable per call. Any campaign that has
already run therefore *measured* how long a Boltz batch takes and how often
molecules repeat, and no GPU benchmark is needed to read it back.

Do not assume the file is there. On `demo-lux` as of 2026-10-06 it did not
exist: `/data/reinvent4_mcp/logs/` was empty, `/shared` held no MSA cache, and
Boltz's weight cache was unpopulated — no campaign had ever driven the external
scorers on that PVC. Check before planning around the harvest, and fall back to
a measured call when it is missing.

[`scripts/analyze-scoring-latency.py`](scripts/analyze-scoring-latency.py)
reads that log — a path, a directory, or `-` for stdin — and reports, per
service, the latency distribution per batch and per molecule, the breakdown by
batch size, and the share of molecule-level demand a result cache would serve.
It excludes records with `error` set or a non-2xx `status_code`: a prediction
is killed at `PREDICT_TIMEOUT_SECONDS` and those capped values would skew p95.

```bash
kubectl --namespace "${PROJECT}" exec deployment/dd-agentic \
  --container dd-agent -- cat /data/reinvent4_mcp/logs/external_scoring.jsonl \
  | ./k8s/scripts/analyze-scoring-latency.py -
```

From the two harvested numbers it derives the replica counts that scale off
them: Boltz replicas needed for `N` concurrent agents at a target step time,
agents supportable at a given replica count, and how many RL steps fit a
campaign window. `--supervisors`, `--replicas`, `--step-minutes`, and
`--campaign-hours` set those inputs; `--batch-size` reads the per-batch time at
a specific batch size instead of the most common one; `--json` emits the same
report for a script to consume.

The exit code is 2 when the log holds no usable Boltz call, which is the signal
to fall back to a single measured call (`ab-boltz-kernels.py --baseline-only`)
rather than to trust a placeholder.

### A/B the optimized Boltz kernels

`services/boltz2/app.py` passes `--no_kernels` to `boltz predict` by default,
which disables Boltz's optimized kernels. `BOLTZ_NO_KERNELS=0` turns them on.

**Measured on MI300 (`useocpm2m-silogen-007`): the kernels do not work and the
flag is not a lever.** Enabling them aborts after ~81 s, partway through the
pairformer, with `ModuleNotFoundError: No module named 'cuequivariance_torch'`
— Boltz's kernel path depends on NVIDIA's cuEquivariance, which is CUDA-only
and absent from this image. Keep `BOLTZ_NO_KERNELS=1` on ROCm. The measured
per-batch time with kernels off is **386 s (6.4 min) for 16 molecules**, 24 s
per molecule, against BRD4 BD1 at `recycling_steps: 3` and
`diffusion_samples_affinity: 5`.

[`scripts/ab-boltz-kernels.py`](scripts/ab-boltz-kernels.py) is what produced
that result and is how to re-check it on a CUDA node, or after an image ships
cuEquivariance. It times one `/score` call each way against BRD4 BD1, using the
precomputed `brd4_bd1.a3m` on the shared volume so neither arm pays an
MSA-server round trip, at the production batch size of 16:

```bash
kubectl --namespace "${PROJECT}" port-forward deployment/dd-agentic-boltz 8091:8091 &
./k8s/scripts/ab-boltz-kernels.py --namespace "${PROJECT}"
```

The script flips `BOLTZ_NO_KERNELS` on `deployment/dd-agentic-boltz` between
arms and waits for the container to come back healthy, so both arms hit the
same replica with the same weights cache. It reports the speedup and what it
does to the replica count. `--baseline-only` runs the default arm alone, which
is also the fallback when the scoring log has no usable history.

### Place batches on idle Boltz replicas

A Boltz replica runs **one prediction at a time** (`BOLTZ_MAX_CONCURRENCY=1`),
and a prediction takes minutes — 386 s for a production batch of 16. Replicas
are therefore not interchangeable mid-flight, and a plain ClusterIP Service is
the wrong shape for them: kube-proxy round-robins without knowing who is busy,
so it will hand a six-minute batch to a replica that is already predicting
while another sits idle, costing a whole RL step.

`dd-agentic-boltz-proxy` is HAProxy in front of the replicas with:

- **`balance leastconn`** — because per-replica concurrency is 1,
  least-connections *is* least-busy.
- **`maxconn 1` per replica** — what turns that from a hint into placement. A
  surplus batch waits in the gateway's queue instead of being stacked onto a
  busy replica.
- **timeouts above `PREDICT_TIMEOUT_SECONDS`** (`client`, `server`, and `queue`
  all 2100 s) — the proxy must never be what aborts a live prediction, and a
  batch that queued for an hour and then succeeded is correct, not a failure.
- **`retries 2` + `option redispatch`** — only connection failures retry, so a
  replica dying before it answers costs a redispatch rather than a NaN batch.

`dd-agentic-boltz` is a **headless** Service (`clusterIP: None`) so the proxy
resolves one address per ready pod and can track them individually; a ClusterIP
would collapse them behind one virtual address. The proxy runs exactly **one**
replica, because least-connections is only least-busy if a single process owns
the whole view of who is busy.

Scale Boltz with `replicas` on `scorers/boltz.yaml` (or `kubectl scale`); the
config has 12 server slots, so nothing in the proxy changes:

```bash
kubectl --namespace "${PROJECT}" scale deployment/dd-agentic-boltz --replicas=3
kubectl --namespace "${PROJECT}" rollout status deployment/dd-agentic-boltz
```

A new replica takes **about 30 s after it is Ready** to start receiving work:
the proxy caches DNS for 10 s (`hold valid`) and then needs two passing
health checks 10 s apart (`rise 2 inter 10s`) before the slot leaves
`MAINT (resolution)`. Checking the stats page immediately after a scale-up
shows the old count and looks like a failure — wait for the slot to turn `UP`.

Queue depth, in-flight work per replica, and which replicas are up are on the
proxy's stats endpoint. Each request also logs `%Tw` (time spent waiting for
a free replica) and `queue=%sq/%bq`, which is what distinguishes a slow
prediction from a queued one, and every response carries an `X-Boltz-Replica`
header naming the replica that served it:

```bash
kubectl --namespace "${PROJECT}" port-forward \
  deployment/dd-agentic-boltz-proxy 8091:8091 8404:8404 &
curl -s http://localhost:8404/stats            # HTML dashboard
curl -s "http://localhost:8404/stats;csv"      # same numbers, parseable
curl -s http://localhost:8404/metrics | head   # Prometheus exposition
kubectl --namespace "${PROJECT}" logs deployment/dd-agentic-boltz-proxy | tail
```

[`scripts/verify-boltz-placement.py`](scripts/verify-boltz-placement.py) proves
the behaviour rather than assuming it. It fires one more concurrent `/score`
call than there are replicas, at reduced fidelity, and checks that each is
placed on its own replica, that no replica is ever given two at once, and that
the surplus queues and still succeeds:

```bash
./k8s/scripts/verify-boltz-placement.py --namespace "${PROJECT}"
```

**Measured on `demo-lux` with M=2**: three concurrent batches used both
replicas, peak in-flight was 1 per replica, and the third waited **197.6 s**
in the queue (`%Tw`) before running — 784.9 s of work in 372.0 s of wall clock,
a 2.11x speedup over serial. A 503 from this script would mean `timeout queue`
is below the prediction time, which downstream reads as a component failure.

> **One-time step when upgrading an existing namespace.** `spec.clusterIP` is
> immutable, so turning `dd-agentic-boltz` headless fails with
> `may not change once set` until the old Service is removed. Delete it once,
> then apply as usual; nothing dials it directly any more:
>
> ```bash
> kubectl --namespace "${PROJECT}" delete service dd-agentic-boltz
> ./k8s/apply.sh --namespace "${PROJECT}" --node "${NODE}"
> ```
>
> The HAProxy workload was also renamed from `dd-agentic-boltz-gateway` to
> `dd-agentic-boltz-proxy`, and that name now belongs to the caching gateway
> below. A Deployment cannot change `strategy` from the defaulted
> `RollingUpdate` to `Recreate` in place, so delete the old one first:
> `kubectl -n "${PROJECT}" delete deployment dd-agentic-boltz-gateway` and
> `kubectl -n "${PROJECT}" delete configmap dd-agentic-boltz-gateway`.

The proxy image, `haproxy:3.2-alpine`, is the **one image not built from this
repo**: kubelet pulls it from Docker Hub once and `IfNotPresent` reuses it. On a
cluster without registry access, import it the same way as the local tags
(`docker pull`, `docker save`, then `ctr -n k8s.io images import`).

### Cache results and coalesce batches

Placement alone does not reduce the amount of prediction being asked for.
`dd-agentic-boltz-gateway` sits in front of the proxy and owns the two things a
proxy cannot do, because they need to understand the payload. It speaks the same
`{smiles, params}` contract as a bare replica, so the ConfigMap's `service_url`
is what it was.

**A per-molecule result cache** under `/shared/boltz_results_cache`, keyed on
`sha256(canonical_smiles ‖ protein_sequence ‖ msa_digest ‖ predict_params)`.
Per *molecule* rather than per batch is the point: a batch of 16 where 10 were
seen before costs 6 predictions. Canonicalising is load-bearing rather than
tidy, because the RL config sets `randomize_smiles: true` — the same molecule
arrives spelled differently on every step, and a literal-string cache would
almost never hit. The MSA is keyed by **content**, so two paths holding one
alignment share entries and editing an `.a3m` invalidates them.

Exploration-tier requests read and write it; `fidelity_tier: validation`
bypasses the read and always re-predicts, so a reported result is never a stale
sample — but still populates the cache. `cache_policy: none` opts out entirely.
A cached entry that names a structure file which has since been deleted is
**dropped rather than served**, since downstream scorers re-open those paths and
a hit pointing at a missing file is worse than a miss.

**Batch coalescing.** `boltz predict <dir>` folds a directory of YAMLs into one
process, so the per-invocation cost is paid once regardless of batch size — and
at least 124 s of the measured 386 s batch is exactly that fixed cost. Requests
waiting on the same `(protein_sequence, params)` are merged into one upstream
call. Concurrent upstream calls are capped at the number of healthy replicas, so
every GPU stays busy while anything beyond that accumulates into a larger batch
instead of queueing a one-molecule call behind another.

**In-flight de-duplication.** If a molecule is already being predicted, a second
request for it waits on that prediction rather than starting another. With
several supervisors exploring neighbouring weight vectors over one target, this
is where the cross-supervisor saving actually lands — caching alone cannot help
when neither request has finished yet.

**Measured on `demo-lux`**: the same three-molecule handoff verification took
**3 m 35 s cold and 12 s warm**, served the identical affinity values and the
same structure paths, and PoseBusters and the interaction profiler re-opened
those cached poses and scored identically. Note the warm run also hit on the
molecule Boltz *skipped*: a deterministic skip is cached, a failed prediction
never is.

`GET /stats` is what `N` should be throttled against, and reports the cache hit
rate that every sizing calculation so far has had to assume:

```bash
kubectl --namespace "${PROJECT}" port-forward \
  deployment/dd-agentic-boltz-gateway 8091:8091 &
curl -s http://localhost:8091/stats | python -m json.tool
```

**Submit and poll instead of holding a connection open.** `POST /score?mode=async`
answers `202 {"task_id": ...}`; `GET /result/{task_id}` answers `202` while it is
queued or predicting, then `200` with the result once — after which it is freed,
so a long campaign cannot grow the registry without bound. `ExternalRESTScorer`
uses it with `async_poll: true`, and its timeout is now configurable
(`timeout` in the component params, or `EXTERNAL_SCORER_TIMEOUT_SECONDS`)
instead of hardcoded at the same 1800 s the service itself enforces. That
combination is what stops a deep queue from reading as a failure and filling an
RL batch with NaN.

The knobs are env on the Deployment: `COALESCE_WINDOW_SECONDS` (how long a batch
is held open), `COALESCE_MAX_MOLECULES` (raise it to merge whole batches from
several supervisors, at the cost of a later first result),
`GATEWAY_MAX_INFLIGHT` (`0` = healthy replicas), and `GATEWAY_CACHE_ENABLED`.

### Move RL off the agent: reinvent-runner

`dd-agentic-reinvent-runner` owns REINVENT4 — the RL and transfer-learning job
lifecycle, the in-process `Scorer` behind `score_molecules`, and the
component/model/transform registries — behind REST. The agent becomes an HTTP
client, so it needs neither REINVENT4 nor a GPU. That is what lets agent count
and RL capacity scale independently: an RL step is mostly spent waiting on the
scoring call, so one GPU hosts several jobs (`jobs.max_concurrent_jobs`),
whereas one-GPU-per-agent would cap the fleet at the number of RL GPUs.

Turn it on in the ConfigMap; it is **off by default**, so an unconfigured
deployment keeps running REINVENT4 in the MCP process exactly as before:

```yaml
runner:
  enabled: true
  base_url: "http://dd-agentic-reinvent-runner:8086"
```

Nothing above the seam changes. `get_job_manager()` returns a `RemoteJobManager`
speaking the same seven methods over HTTP, so no tool knows where REINVENT4
runs. Application outcomes keep their shape too: a refusal like "Maximum
concurrent jobs" or an unknown job id comes back as an `{"error": ...}` dict,
the same as the local manager, and so does an unreachable runner — every caller
treats a dict with `error` as the failure case and would crash on a raise.

**TTPs stay with the agent.** `run_optimization` already resolves a TTP into a
plain `scorer_config` before the job manager sees it, so only that mapped config
crosses the wire and the runner needs no TTP store — note the deliberate absence
of a `ttps` mount in its manifest. `/score` takes the TTP inline for the same
reason.

Two things are deliberate and easy to get wrong. The runner runs **one uvicorn
worker**: the job table and the concurrency slots live in process memory, so a
second worker would keep its own copy and double the effective
`max_concurrent_jobs`. And it sets **`REINVENT_RUNNER_SELF=1`**, because it reads
the same config file as the agent and would otherwise honour `runner.enabled`
and forward every job to itself; the job endpoints additionally hold their own
`JobManager` rather than going through the seam, so they cannot loop even if
that env were lost.

**Do not build a second `JobManager` inside this container.** Because
`REINVENT_RUNNER_SELF=1` makes `runner_enabled()` false here, any extra process
— a `kubectl exec ... python -c "...get_job_manager()..."` diagnostic, most
obviously — constructs a real manager, and restoring the job table used to
mark every job it found running as `failed` on disk. The assumption was that a
job "is dead with the process that ran it", which is false: the worker is a
separate `_run_worker.py` process whose pid is recorded in the same metadata.
One diagnostic command buried three healthy jobs mid-Boltz-call, their logs
still advancing, and the supervisors watching them abandoned the tasks and the
GPUs they held.

Restore now checks whether that pid is still running this job before declaring
it dead, reading `/proc/<pid>/cmdline` for the job's own directory so a
recycled pid is not mistaken for the worker — a real hazard here, since a
restarted runner starts pids from 1 again. A live worker is adopted and keeps
its concurrency slot; a dead one is still failed, which is what frees the slot.
Diagnostics that only need job *state* should go through the conductor's tools
(`get_optimization_status`, `list_jobs`), which cross the seam over HTTP and
touch no manager, or read `job_metadata.json` with plain `sh`.

**Verified on `demo-lux`**: `reinvent_importable: true`, 61 scoring components,
9 priors and 8 transforms served over HTTP; a real 2-step RL job created through
`POST /jobs` ran to completion; and driving it through the client seam gave the
same result, with the third concurrent job refused and naming the two to stop.

```bash
kubectl --namespace "${PROJECT}" exec deployment/dd-agentic --container dd-agent \
  -- python -c "import json,urllib.request; print(urllib.request.urlopen('http://dd-agentic-reinvent-runner:8086/health').read().decode())"
```

The image split is done: `reinvent-runner:local` owns REINVENT4 and the priors
and is built from its own ROCm base, and `dd-agent:local` is **164 MB** of agent
plus MCP server on `python:3.12-slim` — down from 40.8 GB unpacked. The runner
still *imports* `reinvent4_mcp` rather than owning a copy of the job lifecycle,
which is deliberate: one source of truth for the code that spawns RL
subprocesses.

Because the agent image has no REINVENT4 at all, `runner.enabled` is **not
optional** in this deployment — both the ConfigMap and the baked
`reinvent4_mcp_config.yaml` set it, and the in-process code path only exists on
the runner now.

### The fleet's ledger: campaign-store

`dd-agentic-campaign-store` is the control plane for a conductor-led fleet: the
queue of weight vectors still to try, who is holding which task and until when,
what each supervisor achieved, and the Pareto front over those results. It is a
FastAPI service over a SQLite file on `/shared`, so a conductor that restarts
rebuilds its loop by reading the ledger rather than from memory.

A word on vocabulary, because the agent code already uses it differently: in
the store a **campaign** is the whole sweep (one goal, N supervisors, a
wall-clock budget) and a **task** is one supervisor run — which is what
`reinvent4_agent/coordinator.py` calls a campaign.

| Endpoint | Purpose |
|----------|---------|
| `POST /campaigns` | Open a sweep: goal, base TTP, the objectives the front is computed over, dispatch and grace windows |
| `POST /campaigns/{id}/tasks` | Enqueue one weight vector |
| `POST /tasks/claim` | A supervisor takes the oldest queued task on a lease; `204` when idle |
| `POST /tasks/{id}/heartbeat` | Extend the lease |
| `POST /tasks/{id}/result` | Post the achieved objective profile and candidates |
| `POST /tasks/{id}/fail` | Report a failure, requeued unless `requeue: false` |
| `GET /campaigns/{id}/pareto` | The non-dominated front, plus results excluded for missing an objective |
| `GET /campaigns/{id}/events?since=N` | The conductor's completion feed, polled by last event id |
| `GET /campaigns/{id}/status` | The aggregate the REPL prints: task counts, in-flight supervisors with their weight vectors, the front, and time left in each window |
| `POST /campaigns/{id}/close` / `finalize` | Enter the drain window; then close the ledger |

Three properties are load-bearing and worth knowing before changing anything:

- **One replica, and not provisionally.** The atomic claim relies on SQLite's
  write lock. A second replica on another node would be a second process
  writing the same file over a network filesystem, which is exactly where that
  locking cannot be relied on — and two supervisors would be handed the same
  weight vector and burn two GPUs on it. Hence `replicas: 1` and
  `strategy: Recreate`.
- **Claims take the write lock before they read.** `BEGIN IMMEDIATE`, not
  sqlite3's default deferred begin. Without it, concurrent claims read the same
  row as queued and all update it; the regression test for this measured 8
  concurrent claims being handed only 2 distinct tasks.
- **Leases, not assignments.** `DEFAULT_LEASE_SECONDS` is 3600, sized to
  outlive a whole RL step even if the holder never heartbeats, since the
  gateway will hold one scoring call for up to its 2100 s upstream timeout. A
  supervisor that is evicted mid-task therefore gives it back without anyone
  having to notice the crash, and `TASK_MAX_ATTEMPTS: 2` stops a task that
  fails for its own reasons cycling through the whole pool.

```bash
kubectl --namespace "${PROJECT}" exec deployment/dd-agentic --container dd-agent \
  -- python -c "import urllib.request; print(urllib.request.urlopen('http://dd-agentic-campaign-store:8087/health').read().decode())"
```

### The worker pool: dd-supervisor

`dd-agentic-supervisor` is the headless half of the agent: it claims a task
from the ledger, runs the same Driver + Reviewer pair the REPL runs, posts what
it achieved, and claims the next one. It is **the `dd-agent` image with a
different entrypoint** (`python -m reinvent4_agent.supervisor`) — a supervisor
*is* the agent, so there is no supervisor image to build and no second copy of
the agent to keep in step.

It ships at **`replicas: 0`**, so applying the manifests does not start burning
GPU against an empty queue. A campaign's concurrency `N` is the replica count:

```bash
kubectl --namespace "${PROJECT}" scale deployment/dd-agentic-supervisor --replicas=4
```

That is also how the conductor will drive it, through the `deployments/scale`
permission `rbac.yaml` already grants, and it ports to Compose unchanged as
`--scale dd-supervisor=N`.

Two decisions are worth knowing because they are easy to undo by accident:

- **The weight vector is applied in code, not asked for in prose.** A task
  exists to try specific objective weights, so the supervisor reads the base
  TTP, overrides the named weights and transforms, and persists the result as
  a variant TTP (`<base>-<task_id>`) that the Driver is told to run *by id*.
  Asking the Driver to "use weight 1.4 on affinity" invites it to round,
  reinterpret or forget, and the campaign would then be comparing points it
  cannot name. An override naming an objective the TTP does not have fails the
  task rather than being ignored — otherwise the task would report a profile
  for a weight vector it never ran. The variants are kept: they are the record
  of what each task actually ran, which is why this pod mounts the same
  `ttps` directory as the agent.
- **The objective profile is the best *single molecule's* scores.** Not the
  best value seen per objective, which would mix molecules and describe a
  compound nobody made — useless on a front whose whole job is to show
  achievable trade-offs.
- **A task's `intent` is prose handed to the agent verbatim.** Weights,
  transforms, fidelity, step count and a deadline are all the structured
  fields can say, and some of what a run is *for* does not fit them. Holding a
  chemotype is the standard example: that is a choice of generative prior, not
  a weight, and the agent is better placed to make it than a seed field would
  be — it can look the reference compound up with `fetch_compound` and pick a
  model. Without a channel for it the agent generated de novo not because it
  could not do better but because nothing asked it to. State the experiment,
  not the tool calls.

  A task carrying an `intent` also **turns off the de novo fallback**. When an
  agent finishes without starting a run the supervisor normally starts one
  itself, which is a faithful substitute only while every task is de novo. An
  intent may well have asked for a different prior, and a substituted run
  would post into the same Pareto front while measuring something else, so the
  task fails and is retried instead. That guard is what caught an agent
  calling `run_optimization` on two *other* tasks' TTPs: the TTP store is
  shared, so every agent can see all N sibling variants, and nothing scopes it
  to its own.

If a task has a deadline, the supervisor stops the job when it is reached
rather than letting it run: `stop_optimization` asks REINVENT4 to checkpoint,
so the GPU comes back and the work can be resumed instead of discarded. The
partial candidates are still posted, with the deadline noted in the summary.

A supervisor heartbeats its lease every 60 s against the store's 3600 s
default, so a couple of missed beats cost nothing, and a pod that is evicted
mid-task has its task requeued when the lease lapses. On SIGTERM it stops
claiming and lets the task in hand run out the 120 s grace; anything still held
goes back on the queue the same way.

**Presence is a second, separate beat, and the separation is the point.** A
supervisor used to touch the ledger only when it *claimed* something, so an
idle one — the only kind you can actually dispatch to — was indistinguishable
from one that never started. Each supervisor now reports into a `supervisors`
table every 45 s against a 120 s TTL, including on the passes that find the
queue empty, and leaves the pool deliberately on shutdown so a drained pool
does not read as N crashed pods for two minutes. Dead presences are swept on
read, exactly like expired leases. That table is what `list_supervisors` and
`get_pool_capacity` report, and `last_seen_seconds_ago` is what says whether
to believe the state beside it.

The beat is owned by the supervisor's lifetime, **not** by the task's lease
heartbeat. Renewing presence inside the lease heartbeat ties the two together,
and a supervisor whose task is taken from it — by a cancel, or by its campaign
being finalized — keeps working on the job it already started while vanishing
from the pool. Two did exactly that: alive, holding a GPU, invisible. That is
the inverse of what presence exists to prevent, so if you ever move the
renewal, keep it out of anything that can raise `LeaseLost`.

One consequence worth internalising: presence is self-reported, so a pod that
is running but wedged shows as **absent** rather than idle. That is the right
answer for "can I dispatch to it", and it will disagree with `kubectl get
pods` in exactly that case.

### The conductor

`dd-agentic-conductor` opens a campaign, keeps `N` supervisors busy, reviews
the Pareto front to decide what to try next, and closes the books when the
clock runs out. It is the agent image again, kept interactive with `sleep
infinity` so the REPL is reachable by `kubectl exec`, with the loop started per
campaign:

```bash
# open a campaign and conduct it in one command
kubectl --namespace "${PROJECT}" exec -it deployment/dd-agentic-conductor -- \
  python -m reinvent4_agent.conductor --start \
    --goal "improve BRD4 BD1 affinity without losing drug-likeness" \
    --base-ttp brd4-bd1-lead-opt --concurrency 4 --hours 4 --grace-hours 1 \
    --task-budget-hours 0.5 --max-steps 60

# or attach to a campaign that is already open
kubectl --namespace "${PROJECT}" exec -it deployment/dd-agentic-conductor -- \
  python -m reinvent4_agent.conductor --task-budget-hours 0.5 --max-steps 60
```

The campaign id is optional in both directions. With `--start` a new one is
opened and its id printed; without it, the loop attaches to **the only live
campaign** and refuses to guess when several are running — conducting the wrong
one would spend GPU against the wrong goal. Name an id explicitly when you do
have several.

`--objectives` is optional too, and defaulting it is worth knowing about:
omitted, it uses every objective the base TTP declares. Typing them by hand is
the easiest way to get a campaign wrong, because an objective the TTP does not
have fails every task that references it — and the mistake surfaces minutes
later as a task failure rather than as a typo.

It also exposes thirteen MCP tools so a conductor agent can do the same
conversationally: `start_campaign`, `get_campaign_state`, `list_campaigns`,
`enqueue_task`, `list_supervisors`, `get_supervisor_result`,
`get_pareto_front`, `get_campaign_files`, `set_concurrency`,
`cancel_supervisor`, `get_pool_capacity`, `close_campaign` and
`finalize_campaign`.

`get_campaign_files` is the one that is not about steering: it joins the
ledger's task-to-job mapping to the runner's working directories, so "where
are the results" is answerable without knowing the layout. It reports tasks
newest first and carries no scores, so position in that list says nothing
about which run was any good — `get_pareto_front` ranks tasks and
`get_top_candidates` ranks molecules within one job.

**Its TTP mount is writable, and the tool restriction beside it is half of
one decision.** Defining the base TTP is the first step of a campaign on a new
target, and the conductor is who opens campaigns — mounted read-only it could
only ever conduct campaigns against profiles somebody else had made, and
`create_ttp` failed with `EROFS`. What is actually worth preventing is
*rewriting* a base TTP that tasks are already being measured against, which
would silently change what earlier and later results mean. So the mount is
writable and the Deployment sets
`REINVENT4_AGENT_DENY_TOOLS: "update_ttp,delete_ttp"`, which the CLI turns
into an allow-list computed from the live tool list — a deny-list, so a tool
added later is available by default and only the exclusions are maintained.
Relax either half alone and the guarantee is gone, which is why
[`tests/test_k8s_manifests.py`](../tests/test_k8s_manifests.py) asserts them
together. Job output and `/shared` stay read-only: the conductor reviews what
the fleet produced, and the supervisors are what write it.

The restriction lands on the agent, not the filesystem — the tool function
still works, so an operator exec'ing in with `python` can still fix a TTP by
hand. It is the LLM's authority that is withheld.

**It needs its own ServiceAccount, and this is a correction to the original
plan.** Scaling the supervisor pool is how concurrency is expressed, which
means calling the Kubernetes API. The `deployments/scale` grant in
[`k8s/rbac.yaml`](rbac.yaml) is bound to the **`oidc:demo-lux` developer
group**, so it authorises a human's `kubectl` and not a pod — there was no
ServiceAccount anywhere in `k8s/` before this. The manifest therefore ships a
ServiceAccount, Role and RoleBinding, with the Role scoped by `resourceNames`
to `dd-agentic-supervisor` alone and granting only `get` and `patch`, so a
conductor cannot scale Boltz or the runner. (`list` and `watch` are withheld
because they cannot be restricted by name, and granting them would widen the
scope to every Deployment.)

Three things about how the loop behaves:

- **It holds no campaign state.** Every decision comes from a fresh read of
  the ledger, including which weight vectors have already been tried, so a
  conductor restarted mid-campaign picks up where the ledger says it is rather
  than re-dispatching the sweep. `tick()` is idempotent: running it twice
  changes nothing.
- **The soft stop is the task deadline, not a signal.** Each task is enqueued
  with `deadline = min(now + task_budget, hard_stop_at)`, and the supervisor
  turns that into a checkpointing stop of its own RL job. In the normal case
  every in-flight task has therefore wound itself up *before* the hard stop,
  with a checkpoint and a posted result.
- **The hard stop also stops orphaned jobs.** What is left at the hard stop is
  tasks that overran or whose supervisor died. Abandoning a task in the ledger
  does **not** stop its RL job — the job lives in the runner and holds a GPU
  until something tells it to stop, and the first live wind-down left two
  running for exactly this reason. The conductor derives each task's TTP id
  from the task id, finds the job running it, and stops it. The ledger is
  closed first so a supervisor cannot post against a finished campaign, and
  only then are the jobs stopped; freeing the GPUs is best-effort, so a runner
  that cannot be reached does not block the wind-down.
- **It asks before it dispatches.** Boltz is the binding constraint, not the
  agent count, so the loop reads the gateway's queue depth and holds back when
  it is above `max_queue_pressure` batches per healthy replica. A gateway that
  does not answer is treated as *unknown*, not idle — but unknown does not
  block dispatch, or an unreachable gateway would stall the campaign
  indefinitely.

  Queue pressure counts **in-flight batches as well as queued ones**, which is
  not an embellishment: the gateway caps its own upstream calls at the healthy
  replica count, so work beyond capacity waits on a semaphore *inside* the
  gateway rather than queueing at the proxy. Measured on queued alone it reads
  ~0 with every replica saturated, which is no signal at all. One batch per
  replica is a scorer that is busy and keeping up.

  `get_pool_capacity` answers the same question for a conductor agent and
  reports **both** constraints — idle supervisors and scorer room — with
  `binding_limit` naming whichever said no. Either number alone misleads: an
  idle supervisor handed work while Boltz is saturated only moves the wait,
  and a quiet scorer is no use with nothing alive to dispatch to.

The weight vectors come from a two-phase proposer. With no results it seeds the
extremes — a balanced vector plus one per objective emphasised — because
bounding the front early is what makes later refinement meaningful; refining
around a single lucky point first would just explore one corner. With a front,
it perturbs front members one objective at a time, since the front is by
definition the set of trade-offs nothing else beats. It never re-proposes a
vector already in the ledger, and an exhausted sweep dispatches nothing rather
than paying twice for the same point.

`replicas: 1`, and that one matters: two conductors would each top the pool up
to `N` and the campaign would quietly run at `2N`. That is the one way to
overcommit the GPUs the ledger cannot catch, because each individual dispatch
is legitimate.

### A pod stuck in `ContainerCreating` with a Multi-Attach error

```
Warning  FailedAttachVolume  Multi-Attach error for volume "pvc-..."
         Volume is already used by pod(s) dd-agentic-boltz-..., dd-agentic-...
```

The pod landed on a node that cannot attach `dd-agentic-data`, because the
claim is `ReadWriteOnce` and some other node already has it. It will never
start — this is not a transient pull.

It means the Deployment is missing its node pin. `apply.sh` stamps
`nodeSelector[kubernetes.io/hostname]` onto **every** Deployment for exactly
this reason, so the usual cause is a manifest applied directly with
`kubectl apply` instead of through `apply.sh --node`. It bites the supervisor
pool first and most visibly, because that is the only workload that scales past
one replica, so it is the only one with a second pod for the scheduler to place
somewhere else.

Fix it in place without disturbing the rest of the fleet:

```bash
NODE=$(kubectl --namespace "${PROJECT}" get deployment dd-agentic-boltz \
  --output jsonpath='{.spec.template.spec.nodeSelector.kubernetes\.io/hostname}')

for d in dd-agentic-campaign-store dd-agentic-supervisor dd-agentic-conductor; do
  kubectl --namespace "${PROJECT}" patch deployment "$d" \
    --patch "{\"spec\":{\"template\":{\"spec\":{\"nodeSelector\":{\"kubernetes.io/hostname\":\"${NODE}\"}}}}}"
done
```

### Start and stop the fleet

```bash
k8s/scripts/fleet-up.sh --namespace "${PROJECT}" --supervisors 2
k8s/scripts/fleet-up.sh --namespace "${PROJECT}" --down
```

Deployments are independent — scaling the supervisor pool does not start the
scorers it needs, and `dd-agent` only starts a process in a pod that already
exists. Nothing here has a dependency graph, so this script is it: it scales
each tier in order and waits on `kubectl rollout status` between them, which is
gated on the readiness probes and therefore on the services' own `/health`.

Two reasons the order is load-bearing. A supervisor that starts before the
runner is ready claims a task, cannot reach the runner, and burns one of its
two attempts. And a **missing scorer is worse than a missing runner, because it
is quiet**: the scoring component turns a connection error into NaN for every
molecule, which is right for a flaky service mid-campaign, but with one down
for a whole run you get a completed job full of NaN and a geometric mean built
on them — a failure that produces results rather than an error.

For that reason every CPU scorer is started regardless of which objectives the
campaign uses. They hold no GPU, so starting one that is not needed costs
nothing, while skipping one the TTP references costs a silently NaN-scored
campaign.

`--down` takes the runner with it, and RL jobs live in the runner's pod, so
anything still running is killed rather than checkpointed — stop the jobs first
if the work matters. AIM is left alone either way: it is reconciled by the
`dd-agentic-aim-*` AIMServices, so scaling their Deployments would simply be undone.

### Watch the fleet

```bash
k8s/scripts/campaign-status.sh --namespace "${PROJECT}"            # once
k8s/scripts/campaign-status.sh --namespace "${PROJECT}" --watch 30 # every 30s
```

It prints both halves, because **capacity and activity are different
questions**: `kubectl get deploy` says how many supervisors *exist* and is what
`kubectl scale` changes, while the ledger says how many are actually holding a
task. An idle supervisor is a running pod with nothing claimed, and in
`kubectl get pods` it looks exactly like a busy one.

A healthy campaign has those two numbers equal. The two ways they diverge are
worth recognising:

- **`claimed` below the pod count** — supervisors are idle. Either the queue is
  empty, or the conductor is holding dispatch back because the Boltz queue is
  deep (it logs `holding dispatch:` when it does).
- **`queued` above zero with every pod busy** — the campaign wants more
  concurrency than the pool has pods. That is what happens when a campaign's
  `concurrency` and `kubectl scale --replicas` were set to different numbers;
  see below.

To follow what the supervisors are actually doing, all of them at once:

```bash
kubectl --namespace "${PROJECT}" logs --selector app.kubernetes.io/name=dd-agentic-supervisor \
  --prefix --tail=20 --follow
```

### Setting the concurrency N

There are **two numbers**, and they have to agree:

| Number | Where | What it does |
|--------|-------|--------------|
| `concurrency` | on the campaign, in the ledger | What the conductor *targets*: it tops the queue up until `queued + claimed == N` |
| `replicas` | the `dd-agentic-supervisor` Deployment | How many supervisors actually exist to claim that work |

Normally you set only the first and the conductor reconciles the second on its
next tick. **Until the conductor's RBAC is applied that reconciliation gets a
`403`, so set both by hand:**

```bash
# 1. the target, when the campaign is opened (or later via the set_concurrency tool)
#    start_campaign(..., concurrency=4)
# 2. the pods
kubectl --namespace "${PROJECT}" scale deployment/dd-agentic-supervisor --replicas=4
```

If they disagree the campaign still works, it just runs at the smaller of the
two with the difference sitting in `queued`. Leave the manifest at
`replicas: 0` rather than baking `N` in, so applying the manifests never starts
agents against an empty queue.

### Prove the structure handoff across pods

Boltz returns the *absolute path* of each structure it writes to `/shared`, and
PoseBusters and the interaction profiler re-open those paths — no structure is
ever carried in a payload. That was free while all six containers shared one
pod. Now they are separate Deployments, and the handoff holds only because
every `/shared`-mounting pod is pinned to one node (`ReadWriteOnce` is per
node), or because Kyverno mutated the claim to `ReadWriteMany`. Both are
assumptions about the storage class, so this is the one thing in the split
worth a run rather than an argument.

[`scripts/verify-artifact-handoff.py`](scripts/verify-artifact-handoff.py)
drives the production path: one Boltz `/score` against BRD4 BD1, then the paths
it returns relayed to PoseBusters and the profiler in exactly the shape
`ExternalRESTScorer` injects them — the full index-aligned `structure_paths`
plus `protein_pdb_path`. Run it in-cluster with no port-forward at all:

```bash
kubectl --namespace "${PROJECT}" exec --stdin deployment/dd-agentic \
  --container dd-agent -- python - --in-cluster \
  < k8s/scripts/verify-artifact-handoff.py
```

Or from a workstation, where `--namespace` also reports which pod and node
answered and stats the handed-over files from inside them — which is what makes
a green result *cross-pod* rather than merely green:

```bash
kubectl --namespace "${PROJECT}" port-forward deployment/dd-agentic-boltz-gateway 8091:8091 &
kubectl --namespace "${PROJECT}" port-forward deployment/dd-agentic-posebusters 8082:8082 &
kubectl --namespace "${PROJECT}" port-forward deployment/dd-agentic-interaction-profiler 8083:8083 &
./k8s/scripts/verify-artifact-handoff.py --namespace "${PROJECT}"
```

The batch deliberately includes a 60-heavy-atom molecule, past the wrapper's
limit of 56, so Boltz skips it and leaves a `None` hole mid-list.
`ExternalRESTScorer` forwards those holes instead of compacting them, and the
consumers pair `structure_paths[i]` with `smiles[i]`, so a batch without a hole
would never exercise the alignment every partial RL batch depends on.

Exit status is 0 proven, 1 disproven, 2 inconclusive, and the four outcomes are
deliberately not collapsed:

| Outcome | Means |
|---------|-------|
| `ok` | The consumer scored real values off the producer's paths |
| `blind` | 200 with nothing but nulls: the path was accepted and no file behind it could be opened. **This is the storage failure**; the fallback is object storage with URIs instead of paths |
| `rejected` | 4xx: the consumer refused the payload shape. A config or schema problem — the file was never looked for |
| same pod | Producer and consumer answered from one pod, so the run re-proved the in-pod handoff and exits 2 |

Two extra checks run when `--namespace` is given. PoseBusters extracts the
ligand to `<structure>_lig.sdf` *beside* the PDB it was handed, i.e. into the
producer's own output directory, so the script looks for that file from inside
the Boltz pod — if it is missing while PoseBusters scored, the two pods
resolved one path on two filesystems. And if the pods turn out to sit on
different nodes, the claim must already be `ReadWriteMany`, which the report
says out loud.

Re-run it after anything that moves a pod or changes the storage class.
`--structure`/`--smiles` replays against a pose already on the volume, which
proves the same thing without spending another prediction:

```bash
./k8s/scripts/verify-artifact-handoff.py --namespace "${PROJECT}" \
  --structure /shared/predictions/out_<id>/boltz_results_batch_<id>/predictions/mol_0/mol_0_model_0.pdb \
  --smiles 'CC1=C(C)C2=C(S1)N1C(C)=NN=C1[C@H](CC(=O)OC(C)(C)C)N=C2c1ccc(Cl)cc1'
```

### Pin the Deployments and AIM

The dd-agentic Deployments and `AIMService/dd-agentic-aim` are scheduled
independently. The agent pod talks to AIM over the in-cluster Service
`http://dd-agentic-aim-{conductor,supervisor}/v1` (any node). `AIMService` has **no** nodeSelector
in this API; the predictor lands where `spec.profile.selector.acceleratorCount`
GPUs are free on a **single** node.

| Workload | GPUs | How it is pinned |
|----------|------|------------------|
| Every dd-agentic Deployment | 2 (`dd-agentic` + `dd-agentic-boltz`) | `apply.sh --node` → `nodeSelector[kubernetes.io/hostname]` on each |
| `AIMService/dd-agentic-aim` | `acceleratorCount` (1 or 8 in the examples) | Scheduler only. Free that many GPUs on the node you want; do not leave that many free on the other GPU node |

**8-GPU AIM** (for example DeepSeek-V3.1-Terminus fp8 tp8) needs one node with
**8 free** `amd.com/gpu`. Put the dd-agentic pods on the **other** node, which
needs **2 free** GPUs and the `*:local` images. Two 8-GPU nodes can run this as
2+8 or 8+2; they cannot run 8-GPU AIM and the dd-agentic pods on the same node.

Check allocatable vs already requested:

```bash
kubectl get nodes --output custom-columns=NAME:.metadata.name,GPU:.status.allocatable.amd\\.com/gpu
kubectl describe node "${NODE}" | sed -n '/Allocated resources/,/Events/p'
```

Move **dd-agentic** to another node (images must already be in that node's
kubelet store). On a single-node storage class `dd-agentic-data` is
`ReadWriteOnce`, so every pod that mounts it has to leave the old node before
Longhorn will attach the volume to the new one — scale all of them down and
wait for the detach:

```bash
export AGENT_NODE="<gpu-node-for-dd-agentic>"   # e.g. useocpm2m-silogen-007
kubectl --namespace "${PROJECT}" scale deployment --replicas=0 \
  --selector app.kubernetes.io/part-of=dd-agentic
kubectl --namespace "${PROJECT}" wait pod \
  --selector app.kubernetes.io/part-of=dd-agentic --for=delete --timeout=5m
./k8s/apply.sh --namespace "${PROJECT}" --node "${AGENT_NODE}"
```

`apply.sh` scales the rollouts back up as it applies `replicas` from the
manifests. Confirm:

```bash
kubectl --namespace "${PROJECT}" get pods \
  --selector app.kubernetes.io/part-of=dd-agentic --output wide
```

Place **AIM** by GPU packing, then create or switch the service. Scale the
dd-agentic pods onto `AGENT_NODE` first so the AIM node still has 8 free:

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
kubectl --namespace "${PROJECT}" get endpoints dd-agentic-aim-conductor dd-agentic-aim-supervisor
```

Do not `kubectl cordon` a GPU node unless you intend to drain every other
workload on it. Occupying GPUs with `dd-agentic` (2) or scaling unrelated
jobs down is enough for the 2+8 split.

### Rebuild the dd-agent image

`dd-agent:local` carries `reinvent4_agent/` and `reinvent4_mcp/` and nothing
else: **164 MB**, CPU-only, built on `python:3.12-slim`. It used to extend the
REINVENT4 ROCm image with REINVENT4 and the Zenodo priors baked in — 40.8 GB
unpacked, ~11 GiB as a tar — which every agent replica had to pull while also
pinning a GPU it mostly did not use. All of that now lives in
`reinvent-runner:local`, so **that** is the slow image to rebuild and ship, and
the agent is a quick one.

The agent build asserts REINVENT4 and torch are *absent*, so a dependency
pulling them back in fails the build rather than quietly restoring a 40 GB
image. The runner build keeps the opposite check: `import reinvent` works and
`externalrestscorer` is registered.

Docker and kubelet do not share an image store. `load-images.sh` replaces an
existing containerd tag during import, and `apply.sh` restarts the Deployments
after applying so pods load the newly imported image. For a single-image
rebuild, use the focused workflow below because `load-images.sh` requires all
Compose images to exist in Docker.

Either build **on the GPU node** (same host as `kubernetes.io/hostname` on
the Deployment) or build **elsewhere and copy the tar** onto that node.
Both paths end the same way: remove the old kubelet tag, import the tar,
restart the Deployment.

Substitute `reinvent-runner` for `dd-agent` below when it is the runner you are
rebuilding — that tar is the big one, so write it on a filesystem with room (not
a tiny root `/tmp`). Confirm the build log shows **not CACHED** for the `COPY`
of whichever package you changed; a cache hit means the image still has the old
code. A BuildKit warning about undefined `$PYTHONPATH` on the runner's
`ENV PYTHONPATH=...` line is harmless.

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

### Check scorer health

Dial the scorers over cluster DNS from the agent pod. These are the same URLs
the ConfigMap gives `ExternalRESTScorer`, so this also proves the split
services resolve:

```bash
kubectl --namespace "${PROJECT}" exec deployment/dd-agentic \
  --container dd-agent -- python -c \
  'import urllib.request; [print(url, urllib.request.urlopen(url).read().decode()) for url in ["http://dd-agentic-boltz-gateway:8091/health", "http://dd-agentic-posebusters:8082/health", "http://dd-agentic-interaction-profiler:8083/health", "http://dd-agentic-admet:8084/health", "http://dd-agentic-retrodmta:8085/health"]]'
```

Boltz answers through its gateway here, which is the path
`ExternalRESTScorer` takes. The headless `dd-agentic-boltz` still resolves to
the pods themselves if you want to reach one replica directly.

`dd-agentic-admet` should have three ready addresses, the other CPU scorers
one each, and `dd-agentic-boltz` one per replica (`v1 Endpoints` is deprecated
from Kubernetes 1.33, so this reads EndpointSlices):

```bash
kubectl --namespace "${PROJECT}" get endpointslices \
  --selector 'app.kubernetes.io/component in (scorer,scorer-gateway)'
```

Boltz also reports `no_kernels` on `/health`, but only once
`boltz2-wrapper:local` has been rebuilt since that field was added — an older
image returns `{"status": "ok"}` alone, and `ab-boltz-kernels.py` falls back to
trusting the arm it asked for.

**ADMET is three replicas at 32Gi because it is the one scorer a TTP calls
more than once per step** — once per ADMET objective. At six concurrent jobs
and three endpoints that is eighteen calls per step round, and a single
replica at 8Gi was OOMKilled eight times (`exitCode 137`, one of them after
eight minutes alive). It is also the most expensive call per unit of answer:
`/score` predicts *every* endpoint for the whole batch and returns the one
that was asked for, so three objectives cost twenty-one endpoint-predictions
where three would do. The service is stateless with no `/shared` mount, so
replicas need no coordination and a plain ClusterIP round-robin is enough —
unlike Boltz, which needs the proxy to place batches.

That OOM is worth recognising by its downstream shape, because none of the
symptoms point at memory:

- **Objectives go NaN unevenly.** One run had `Caco2` at 50% NaN,
  `Solubility` at 99% and `logD` at 99%, while every non-ADMET component sat
  at 0%. A failure confined to the objectives sharing one service is a
  capacity problem, not a scoring one.
- **The Pareto front comes back empty with everything ineligible.** A result
  needs every declared objective to reach the front, so a dead scorer takes
  all six results off it at once. `get_pareto_front`'s `ineligible` list names
  the missing objectives — read that before concluding the run found nothing.
- **The results CSV goes ragged and readers drop rows silently.** The header
  is written from the first batch; a scorer that fails during step 1 reports
  no metadata, then reports some later, and from that point every row is
  wider than the header. 32 of 48 molecules were discarded without a word.
  `ExternalRESTScorer` now fixes its metadata schema on the first batch
  whether or not that batch succeeded, and `read_results` reports
  `rows_skipped` with a warning rather than summarising what survived.

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
