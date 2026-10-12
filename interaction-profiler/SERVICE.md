# Interaction Profiler — REST service

A FastAPI wrapper around the profiler so the **agent calls it over HTTP** instead of importing
the library. The single FastAPI app lives in `services/interaction-profiler/app.py`
(boltz2/posebusters convention): `GET /health`, `GET /schema`, `POST /analyze`, `POST /profile`,
structures exchanged through a shared volume, self-describing JSON out.

> **Out-of-band analysis service — NOT a REINVENT RL scorer.** It deliberately does not
> implement the `POST /score` contract (`{smiles, params}` → `{scores, metadata, artifacts}`)
> that `ExternalRESTScorer` expects, and is not registered in `config.py` / `mapper.py` /
> `docker-compose` scoring. The agent consumes `/analyze` and `/profile` directly. A `/score`
> route + a defensible scalar objective could be added later if it should join the RL loop.

Normal use is via `docker compose` (Section 2); the local path below is for development only.

---

## 1. Local run + test (dev only, no Docker)

Dependencies are bundled in the Docker image; for local dev install the package with service
extras, then run the deployment app by path (it is no longer a package module):

```bash
cd interaction-profiler
pip install -e ".[service]"

# run the single service app (services/interaction-profiler/app.py) via --app-dir:
uvicorn app:app --app-dir ../services/interaction-profiler --host 0.0.0.0 --port 8080

# throughput (CPU-bound -> scale with processes):  add --workers 4
# point at a structure dir (relative `structure` paths resolve here):
IP_SHARED_DIR="$PWD/examples" uvicorn app:app --app-dir ../services/interaction-profiler --port 8080
```

### Test

In another terminal:

```bash
# 1) liveness + tool versions
curl -s localhost:8080/health | python3 -m json.tool

# 2) SMILES only (medchem taxonomy; no structure file needed)
curl -s localhost:8080/profile -H 'content-type: application/json' -d '{
  "molecule": {"name": "aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O"}
}' | python3 -m json.tool

# 3) a structure (start the server with IP_SHARED_DIR="$PWD/examples", as above)
curl -s localhost:8080/analyze -H 'content-type: application/json' -d '{
  "molecules": [{"name": "1OIU", "structure": "1OIU_boltz.pdb", "ligand_resname": "LIG",
                 "smiles": "c1cc(cc(c1)S(=O)(=O)N)Nc2nc3c(c(n2)OCC4CCCCC4)nc[nH]3"}]
}' | python3 -m json.tool
```

Interactive API docs (Swagger UI) are served at
<http://localhost:8080/docs> while the service is running.

You can also run the bundled smoke + speed harness:

```bash
python scripts/service_benchmark.py --batch examples/example_batch.json \
    --concurrency 8 --out /tmp/service_benchmark.json
```

---

## 2. Docker build + run + test

The image bases on `micromamba` and pulls the scientific stack
(RDKit / OpenMM / MDAnalysis) from conda-forge.

> **Cluster note:** do not build images on the head/login node. Submit the
> build through Slurm so it lands on an allocated compute node, e.g.:
> ```bash
> srun --partition=<partition> --cpus-per-task=8 --mem=32G --time=1:00:00 \
>   docker build -t interaction-profiler:latest .
> ```

### Build

```bash
cd interaction-profiler
docker build -t interaction-profiler:latest .
```

### Run

Mount the directory that holds the structures you want profiled as `/shared`
(replace the placeholder path with a real one):

```bash
docker run --rm -p 8080:8080 \
  -v "$PWD/examples:/shared" \
  interaction-profiler:latest
```

Or with `docker compose` (edit the `./shared` volume in `docker-compose.yml`):

```bash
docker compose up --build
```

### Test

Same requests as the local case (the container listens on `:8080`):

```bash
curl -s localhost:8080/health | python3 -m json.tool

curl -s localhost:8080/analyze -H 'content-type: application/json' -d '{
  "molecules": [{"name": "1OIU", "structure": "1OIU_boltz.pdb", "ligand_resname": "LIG",
                 "smiles": "c1cc(cc(c1)S(=O)(=O)N)Nc2nc3c(c(n2)OCC4CCCCC4)nc[nH]3"}]
}' | python3 -m json.tool
```

Because the volume is mounted at `/shared` (the default `IP_SHARED_DIR`), the
relative `structure` path `1OIU_boltz.pdb` resolves to `/shared/1OIU_boltz.pdb`
inside the container.

---

## Input modes & ligand matching

**SMILES validation.** Any supplied `smiles` is parsed up front
(`MolFromSmiles` + InChIKey in one pass); an unparseable SMILES is rejected
immediately with a per-molecule error (`invalid SMILES: ...`) rather than
silently degrading to geometry-perceived bond orders.

**SMILES advisories.** A *valid but suspicious* template adds a
`preparation.warnings` note (it still profiles): multi-component SMILES
(salts/counterions), wildcard/dummy atoms (`*`), or no-carbon (inorganic/ion).

Every request is classified into an **input mode**, echoed back on
`metadata.input_mode`:

| Mode                | Inputs                | Behavior                                                         |
|---------------------|-----------------------|------------------------------------------------------------------|
| `smiles_only`       | `smiles`              | Medchem taxonomy only (no structure).                           |
| `structure_only`    | `structure`           | Ligand bond orders **perceived from geometry** — less reliable. A warning is added; supply a SMILES for correct chemistry. |
| `structure+smiles`  | `structure` + `smiles`| The SMILES is the bond-order template **and** identifies the ligand. Recommended. |

**Ligand selection by formula.** When a `smiles` is given and the structure
has several hetero groups, the profiler prefers the ligand copy/type whose
heavy-atom formula matches the SMILES — so you get the intended molecule
without always needing `ligand_resname`.

**InChIKey verification.** Every result reports, in `preparation`:

- `template_inchikey` — InChIKey of the input SMILES,
- `ligand_inchikey` — InChIKey of the prepared ligand,
- `inchikey_match` — whether their connectivity skeletons (first block) agree
  (robust to protonation/stereo). `null` when no SMILES was supplied.
- `inchikey_exact_match` — whether the full InChIKeys are identical (stereo +
  charge sensitive). `inchikey_match=true` with `inchikey_exact_match=false`
  means connectivity agrees but stereochemistry/protonation differs.

The same `strict_ligand_match` behavior is available in the CLI
(`interaction-profile … --strict-ligand-match`) and the library
(`profile_batch(..., strict_ligand_match=True)`), and an unparseable SMILES is
rejected everywhere, not just in the service.

A mismatch adds a warning. Set the option `strict_ligand_match: true` to turn a
mismatch into a per-molecule **error** instead (the interactions/triples are
cleared) — useful as a guardrail in an automated scorer:

```bash
curl -s localhost:8080/analyze -H 'content-type: application/json' -d '{
  "molecules": [{"name": "1OIU", "structure": "1OIU_boltz.pdb", "ligand_resname": "LIG",
                 "smiles": "c1cc(cc(c1)S(=O)(=O)N)Nc2nc3c(c(n2)OCC4CCCCC4)nc[nH]3"}],
  "options": {"strict_ligand_match": true}
}'
```

## Endpoints

| Method | Path       | Purpose                                                        |
|--------|------------|----------------------------------------------------------------|
| GET    | `/health`  | Liveness, shared-dir status, tool versions (rdkit/openmm/…).   |
| GET    | `/schema`  | Published JSON schema of the `BatchResult` response.           |
| POST   | `/analyze` | Profile a **list** of molecules → one `BatchResult`.           |
| POST   | `/profile` | **Single** molecule → per-ligand result list (scorer-friendly).|
| GET    | `/docs`    | Interactive Swagger UI.                                        |

### `POST /analyze` request

```json
{
  "batch_id": "round7",
  "molecules": [
    {"name": "1OIU", "structure": "1OIU_boltz.pdb", "ligand_resname": "LIG",
     "smiles": "c1cc(cc(c1)S(=O)(=O)N)Nc2nc3c(c(n2)OCC4CCCCC4)nc[nH]3"},
    {"name": "cand_042", "structure": "boltz/cand_042.cif", "smiles": "Cc1ccc(cc1)S(N)(=O)=O"},
    {"name": "aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O"}
  ],
  "options": {"ph": 7.4, "radius": 8.0, "drop_uncertain": true}
}
```

**Molecule fields** (at least one of `structure` / `smiles` required):

| Field                  | Meaning                                                             |
|------------------------|---------------------------------------------------------------------|
| `name`                 | Identifier echoed on the result (required).                         |
| `smiles`               | Ligand SMILES. Alone → taxonomy only; with a structure → bond-order template. |
| `structure`            | PDB/mmCIF path. Relative paths resolve under `IP_SHARED_DIR`.       |
| `ligand_resname`       | Pin the ligand by residue name.                                     |
| `ligand_chain`         | Pin the ligand by chain id.                                         |
| `ligand_keyword`       | Case-insensitive substring match on the ligand residue id.          |
| `peptide_ligand_chain` | Treat a whole chain (e.g. a cyclic peptide) as the ligand.          |
| `source`               | Free-form provenance tag, e.g. `crystal` / `cofold`.               |
| `ph`                   | Per-molecule pH override.                                           |

**Options** (run-level):

| Option             | Default | Meaning                                                        |
|--------------------|---------|----------------------------------------------------------------|
| `ph`               | `7.4`   | Protonation pH for ligand + receptor.                          |
| `radius`           | `8.0`   | Pocket crop radius (Å), 1–25.                                  |
| `max_interactions` | `null`  | Cap per input. `null` = adaptive (~1.2× heavy atoms). `0` = no cap. |
| `drop_uncertain`   | `true`  | Keep only confident interactions (dropped after the cap).      |
| `all_ligands`      | `false` | Profile every qualifying ligand in each structure.             |
| `use_gpu`          | `false` | Use Torch `cdist` for distances if available.                  |
| `strict_ligand_match` | `false` | Fail a `structure+smiles` molecule when its ligand InChIKey doesn't match the template. |

### Response

`/analyze` returns a `BatchResult`: `batch_id`, `n_molecules`, `provenance`,
and `molecules[]`, each with `metadata` (name, `input_mode`, ligand SMILES,
receptor type, counts, timing), `composition` (fragments + coverage),
`interactions`, `triples`, and a `preparation` report (bond assignment,
`template_inchikey` / `ligand_inchikey` / `inchikey_match`, warnings).
`/profile` returns `{ "provenance": {...}, "results": [ <MoleculeResult>, ... ] }`.

### Python client

```python
import httpx

resp = httpx.post("http://localhost:8080/profile", json={
    "molecule": {"name": "cand_042", "structure": "boltz/cand_042.cif",
                 "smiles": "Cc1ccc(cc1)S(N)(=O)=O"},
    "options": {"ph": 7.4, "radius": 8.0},
}, timeout=300)
result = resp.json()["results"][0]
n_interactions = result["metadata"]["n_interactions"]
triples = result["triples"]   # (fragment, residue, interaction) rows
```

---

## Configuration (environment variables)

| Variable        | Default   | Meaning                                             |
|-----------------|-----------|-----------------------------------------------------|
| `IP_HOST`       | `0.0.0.0` | Bind address.                                       |
| `IP_PORT`       | `8080`    | Port.                                               |
| `IP_WORKERS`    | `1`       | Uvicorn worker **processes**. Scale here for throughput. |
| `IP_SHARED_DIR` | `/shared` | Base dir for resolving relative `structure` paths. Paths are confined to it (no `..` traversal / outside-absolute). |
| `IP_CACHE_DIR`  | temp dir  | Where prepared receptors are cached (hash-keyed, atomic writes). Kept **out** of `/shared` so the artifact volume can be read-only and concurrent requests don't race. |

---

## Performance notes

- The profiler is **CPU-bound Python**. Client concurrency against a single
  worker does **not** speed things up (the GIL serializes the work) — scale
  with worker **processes** (`IP_WORKERS`), not client threads.
- Prepared receptors are cached (hash-keyed under `IP_CACHE_DIR`, atomic
  writes). The first request for a new receptor pays a few seconds of
  PDBFixer/OpenMM prep; every later ligand against that target is fast
  (~0.1 s). The cache lives outside `/shared`, so the artifact volume can be
  mounted read-only and concurrent workers won't collide.
- Benchmark (60 crystal complexes, warm cache, 1 GPU / 8 CPU compute node):
  ~471 complexes/min single-worker serial; ~934/min at 4 workers with 8-way
  client concurrency (1.93×). See `scripts/service_benchmark.py`.
