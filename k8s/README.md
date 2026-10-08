# Kubernetes

This directory deploys the six Compose containers as six Kubernetes
Deployments plus an in-cluster AIM endpoint. Run the scripts from
`examples/dd_agentic/` (the Dockerfiles and gitignored build inputs live
there).

```
k8s/
├── README.md                 # this file
├── apply.sh                  # kustomize overlay: namespace + nodeSelector
├── build-images.sh           # docker build of the six selected image tags
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

The manifests in [`manifests/`](manifests) run each of the six Compose
containers as its own Deployment:

| Deployment | Service | GPU | `/shared` |
|------------|---------|-----|-----------|
| `dd-agentic` | — (reached by `kubectl exec`) | 1 | yes |
| `dd-agentic-boltz` | `dd-agentic-boltz:8091` (headless) | 1 each | yes |
| `dd-agentic-boltz-proxy` | `dd-agentic-boltz-proxy:8091` (internal) | no | no |
| `dd-agentic-boltz-gateway` | `dd-agentic-boltz-gateway:8091` | no | yes |
| `dd-agentic-posebusters` | `dd-agentic-posebusters:8082` | no | yes |
| `dd-agentic-interaction-profiler` | `dd-agentic-interaction-profiler:8083` | no | yes |
| `dd-agentic-admet` | `dd-agentic-admet:8084` | no | no |
| `dd-agentic-retrodmta` | `dd-agentic-retrodmta:8085` | no | no |

The `service_url`s in [`manifests/configmap.yaml`](manifests/configmap.yaml)
are those Service names, unqualified so they resolve in whichever namespace
`apply.sh --namespace` deploys into. They were `http://127.0.0.1:<port>` while
everything shared one pod.

GPU demand is **one for REINVENT in `dd-agentic` plus one per Boltz replica**,
so the default `replicas: 2` on `dd-agentic-boltz` needs three free
`amd.com/gpu`. Neither GPU workload may surge during a rollout, since the extra
pod would ask for another GPU and stay `Pending`: `dd-agentic` uses
`strategy: Recreate`, and `dd-agentic-boltz` uses a rolling update pinned to
`maxSurge: 0`, which replaces one replica at a time and keeps the others
serving.

Scoring traffic goes to **`dd-agentic-boltz-gateway`**, not to the replicas.
See [Place batches on idle Boltz replicas](#place-batches-on-idle-boltz-replicas).

AIM is a separate `AIMService` and can use **1 GPU** (the Qwen3-32B example) or
**8 GPUs** (a tensor-parallel catalog profile). It does not have to share a
node with the rest: `apply.sh --node` pins every Deployment, and AIM lands
wherever `acceleratorCount` GPUs are free (often the other GPU node). Images
for the Deployments are imported onto **that node** — nothing is pushed to a
registry. AIM pulls its catalog image as usual.

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
- A GPU node with at least two free `amd.com/gpu` for the dd-agentic
  Deployments, with the six `*:local` images in that node's kubelet store. AIM
  needs `acceleratorCount` free GPUs on **one** node (1 for the Qwen3-32B
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

- **Images are node-local.** `load-images.sh` imports the six `*:local` tags
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
`http://dd-agentic-aim/v1` (any node). `AIMService` has **no** nodeSelector
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
kubectl --namespace "${PROJECT}" get endpoints dd-agentic-aim
```

Do not `kubectl cordon` a GPU node unless you intend to drain every other
workload on it. Occupying GPUs with `dd-agentic` (2) or scaling unrelated
jobs down is enough for the 2+8 split.

### Rebuild the dd-agent image

`dd-agent:local` bakes `reinvent4_agent/` into the image, and after the pod
split it is used only by `deployment/dd-agentic`. Docker and kubelet do not
share an image store. `load-images.sh` replaces an existing containerd tag
during import, and `apply.sh` restarts the Deployments after applying so pods
load the newly imported image. For a single-image rebuild, use the focused
workflow below because `load-images.sh` requires all six Compose images to
exist in Docker.

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

The CPU scorers should each have exactly one ready address, and
`dd-agentic-boltz` one per replica (`v1 Endpoints` is deprecated from
Kubernetes 1.33, so this reads EndpointSlices):

```bash
kubectl --namespace "${PROJECT}" get endpointslices \
  --selector 'app.kubernetes.io/component in (scorer,scorer-gateway)'
```

Boltz also reports `no_kernels` on `/health`, but only once
`boltz2-wrapper:local` has been rebuilt since that field was added — an older
image returns `{"status": "ok"}` alone, and `ab-boltz-kernels.py` falls back to
trusting the arm it asked for.

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
