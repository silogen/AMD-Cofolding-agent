# admet-predictor — REST service

HTTP service for consensus ADMET prediction (7 endpoints) with a double applicability domain.
Models load **once** at startup and stay resident; each request is pure compute. Single FastAPI
app in `services/admet-predictor/app.py` (boltz2/posebusters convention). See
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
     -d '{"smiles":["CC(=O)Oc1ccccc1C(=O)O"],"params":{"endpoint":"logp"}}'
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
     -d '{"smiles":["CCO","c1ccccc1O"],"format":"compact"}'
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
{"smiles": ["CCO", "c1ccccc1O"], "params": {"endpoint": "logp"}}
// response
{"scores": [-0.17, 1.53],
 "metadata": {"endpoint": "logp", "unit": "logP (dimensionless)",
              "std": [0.21, 0.04], "ad_pass": [true, true]},
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
  "predictions": {
    "logp": {"mean": 1.14, "std": 0.14, "n_models": 3, "unit": "logP (dimensionless)",
             "condition": "...", "per_model": {"XGBoost": .., "Chemprop-MTL": .., "CheMeleon-MTL": ..}}
  },
  "applicability_domain": {
    "AD Tanimoto": {"score": .., "top5_mean": .., "threshold": 0.4358, "pass": true},
    "AD Property profile FFN": {"score": .., "top5_mean": .., "threshold": 0.7334, "pass": true},
    "AD Pass": true
  }
}
```

### Output formats
- `records` (default): list of the above, one per input (order preserved; failures → `ok:false`).
- `compact`: `{endpoints, units, conditions, models, molecules:[{input_smiles, ok, mean[], std[], ad_pass}]}`.
- `table`: `{columns, rows}` (one row per molecule) — direct to a DataFrame.

### Python client
```python
import requests
r = requests.post("http://localhost:8084/predict",
                  json={"smiles": smiles_list, "format": "compact"})
res = r.json()
```

## Models & training

| model | representation | notes |
|---|---|---|
| XGBoost | RDKit physchem (~210) + Avalon-2048 | 500 trees; 5-seed mean |
| Chemprop-MTL | learned D-MPNN | 7-task multitask, from scratch |
| CheMeleon-MTL | foundation-initialized D-MPNN | 7-task multitask |

Trained on 156,797 curated molecules; evaluated by a shared **Butina 5-fold scaffold-cluster
CV** (generic Bemis-Murcko scaffolds, cutoff 0.4, whole clusters held out). Consensus = mean
of the three. Multitask objective = MAE on standardized targets + per-head inverse-std weights.

### Performance (out-of-fold consensus)

| endpoint | R² | RMSE | MAE |
|---|---:|---:|---:|
| logp | 0.826 | 0.744 | 0.509 |
| logd | 0.740 | 0.709 | 0.507 |
| aqueous_solubility | 0.704 | 0.859 | 0.630 |
| kinetic_solubility | 0.475 | 0.444 | 0.328 |
| caco2_permeability | 0.414 | 0.467 | 0.342 |
| efflux_mdck | 0.438 | 0.474 | 0.350 |
| efflux_caco2 | 0.521 | 0.287 | 0.200 |

## Applicability domain
Two orthogonal nearest-neighbor signals to the 156k training set: **AD Tanimoto** (Avalon
structural) and **AD Property profile FFN** (Chemprop latent cosine). Thresholds = 5th
percentile of a 1M-compound scan (~95% of a screening library passes each axis). `AD Pass` =
`true` only if both pass; `false` ⇒ extrapolation, treat cautiously.

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
efflux) have modest R². Trained at pH 7.4. `AD Pass: false` = out of the training region.
