# admet-predictor — REST service

HTTP service for consensus ADMET prediction with selectable `drug` (17 endpoints)
and `agro` (25 endpoints) model sets. Models and domain references load **once**
at startup and stay resident; each request is pure compute. Single FastAPI app
in `services/admet-predictor/app.py` (boltz2/posebusters convention). See
[README.md](README.md) for the library/CLI.

**Role:** REINVENT scorer (`POST /score`, one objective per endpoint) *and* agent analysis
service (`POST /predict`). Normal deployment is via `docker compose` (below); the local path is
for development only.

## 1. Local run + test (dev only, no Docker)

```bash
# from dd_agentic/ (deps only needed for local dev; the image bundles them)
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu && pip install ./admet-predictor
MODELS_DIR=$PWD/models/admet \
  uvicorn app:app --app-dir services/admet-predictor --host 0.0.0.0 --port 8084

curl -s http://localhost:8084/health
curl -s http://localhost:8084/score -H 'Content-Type: application/json' \
     -d '{"smiles":["CC(=O)Oc1ccccc1C(=O)O"],"params":{"endpoint":"logp","model_set":"drug"}}'
```

## 2. Docker build + run + test

### Build (context = dd_agentic/)
```bash
docker build -f services/admet-predictor/Dockerfile -t admet-predictor:latest .
```

### Run
```bash
docker run -d --name admetp -p 8084:8084 admet-predictor:latest
# GPU (needs a CUDA/ROCm torch image variant + --gpus all): -e PRED_DEVICE=cuda
```

### Test
```bash
curl -s http://localhost:8084/health          # {"status":"ok","device":"cpu",...}
curl -s http://localhost:8084/predict -H 'Content-Type: application/json' \
     -d '{"smiles":["CCO","c1ccccc1O"],"model_set":"drug","format":"compact"}'
printf "smiles\nCCO\nc1ccccc1O\n" > /tmp/t.csv
curl -s -F "file=@/tmp/t.csv" -F "format=csv" http://localhost:8084/predict_csv
docker rm -f admetp
```

## Endpoints

| method | path | purpose |
|---|---|---|
| GET | `/health` | liveness, endpoint list, active device, batch size |
| POST | `/score` | REINVENT scorer contract (`{smiles, params:{endpoint}}` -> `{scores, metadata}`) |
| POST | `/predict` | `{"smiles":[...], "format":"records|compact|table", "batch_size":N}` |
| POST | `/predict_csv` | multipart CSV upload; form `smiles_col`, `batch_size`, `format` (`json`\|`csv`) |
| GET | `/docs` | Swagger UI |

### `POST /score` (REINVENT scorer contract)

One TTP objective per endpoint. Returns **one scalar per input SMILES** (the ensemble mean of
the selected endpoint), `null` on failure, strict index alignment. The raw predicted value is
returned; the TTP `transform`/`weight` own the direction (no baked "more = better" objective).

```jsonc
// request
{"smiles": ["CCO", "c1ccccc1O"],
 "params": {"endpoint": "logp", "model_set": "drug"}}
// response
{"scores": [-0.17, 1.53],
 "metadata": {"model_set": "drug", "endpoint": "logp",
              "unit": "logP (dimensionless)",
              "std": [0.21, 0.04], "ad_pass": [true, true],
              "response_ad_pass": [true, true],
              "out_of_training_range": [false, false],
              "excluded_models": [{}, {}]},
 "artifacts": {}}
```

Bad endpoint -> HTTP 400. Example TTP objective:

```json
{"name": "LogD", "type": "external", "service": "admet",
 "service_params": {"endpoint": "logd"}, "weight": 0.3,
 "transform": {"type": "reverse_sigmoid", "low": 1, "high": 4, "k": 0.5}}
```

### `POST /predict` response (records, per molecule)
```json
{
  "input_smiles": "CC(=O)Oc1ccccc1C(=O)O",
  "standardized_smiles": "CC(=O)Oc1ccccc1C(=O)[O-]",
  "ok": true,
  "model_set": "drug",
  "predictions": {
    "logp": {
      "mean": 1.14, "std": 0.14, "n_models": 2, "n_models_total": 2,
      "unit": "logP (dimensionless)", "condition": "...",
      "per_model": {"XGBoost": 1.23, "Chemprop-MTL": 1.05},
      "excluded_models": {},
      "physical_bounds": {"lower": -8.0, "upper": 14.0},
      "response_domain": {
        "method": "training_p01_p99", "available": true,
        "lower": -1.0, "upper": 6.2, "n_train": 58388, "pass": true
      },
      "out_of_training_range": false
    }
  },
  "applicability_domain": {
    "AD Tanimoto": {"score": 0.82, "threshold": 0.4358, "pass": true},
    "AD Pass": true
  }
}
```

### Output formats
- `records` (default): list of the above, one per input (order preserved; failures → `ok:false`).
- `compact`: `{endpoints, units, conditions, molecules:[{input_smiles, ok, mean[], std[], response_ad_pass[], ad_pass}]}`.
- `table`: `{columns, rows}` (one row per molecule) — direct to a DataFrame.

### Python client
```python
import requests
r = requests.post("http://localhost:8084/predict",
                  json={"smiles": smiles_list, "model_set": "drug",
                        "format": "compact"})
res = r.json()
```

## Models & training

| model | representation | notes |
|---|---|---|
| XGBoost | Avalon-2048 plus the serialized descriptor contract | one model per endpoint |
| Chemprop-MTL | learned D-MPNN | 17-task drug or 25-task agro checkpoint |

Consensus is the mean of physically valid available members. Chemprop v2 embeds
its inverse target scaler in the checkpoint; inference therefore uses checkpoint
outputs directly and does not apply `mean + std × prediction` again.

## Applicability domain
Three distinct checks are reported:

1. **Chemical-space AD:** nearest-neighbor Avalon Tanimoto against training structures.
2. **Physical member validation:** a model member outside absolute endpoint bounds
   is excluded from consensus and retained under `excluded_models` for audit.
3. **Response-domain AD:** the consensus is flagged when outside the endpoint's
   observed training p01–p99 interval. This is a warning only; the value remains
   available.

## Configuration (environment variables)

| var | default | meaning |
|---|---|---|
| `MODELS_DIR` | `/app/models` (image; central `dd_agentic/models/admet` baked in) | model files |
| `PRED_BATCH` | 1000 | molecules per frozen pass |
| `PRED_DEVICE` | `auto` | `cpu` \| `cuda` \| `auto` |
| `PRED_WORKERS` | min(32, ncpu) | featurization/standardization processes |
| `PRED_LOADER_WORKERS` | 8 | chemprop dataloader workers |

## Performance notes
CPU, models resident: ~**158 mol/s** at batch 1000 (full ensemble + AD). ~4.75 s one-time
startup (model load); small-batch latency ~0.4 s is per-request overhead. Stateless — scale
horizontally (multiple containers / `--workers N`, each loads its own models).

## Limitations
Regression estimates, not measurements; harder endpoints (kinetic solubility, permeability,
efflux) have modest R². Trained at pH 7.4. Chemical-space and response-domain
flags measure different kinds of extrapolation and should both be reported.
