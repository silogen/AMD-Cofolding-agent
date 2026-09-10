# retrodmta-predictor — REST service

HTTP service for per-target, per-temporal-window RetroDMTA project-endpoint prediction. An
agent fixes a `(target, window)` for a campaign; that model loads once and stays resident
(LRU cache), so subsequent requests are pure compute. Single FastAPI app in
`services/retrodmta-predictor/app.py` (boltz2/posebusters convention). See [README.md](README.md)
for the library/CLI.

**Role:** REINVENT scorer (`POST /score`, one objective per target endpoint) *and* agent
analysis service (`POST /predict`). Normal deployment is via `docker compose` (below); the local
path is for development only.

## 1. Local run + test (dev only, no Docker)

```bash
# from dd_agentic/ (deps only needed for local dev; the image bundles them)
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu && pip install ./retrodmta-predictor
MODELS_DIR=$PWD/models/retrodmta \
  uvicorn app:app --app-dir services/retrodmta-predictor --host 0.0.0.0 --port 8085

curl -s http://localhost:8085/targets
curl -s http://localhost:8085/score -H 'Content-Type: application/json' \
     -d '{"smiles":["CCO"],"params":{"target":"FXA","endpoint":"FXA_pKi"}}'
```

## 2. Docker build + run + test

### Build (context = dd_agentic/)
```bash
docker build -f services/retrodmta-predictor/Dockerfile -t retrodmta-predictor:latest .
```

### Run
```bash
docker run -d --name retrop -p 8086:8085 retrodmta-predictor:latest
```

### Test
```bash
curl -s http://localhost:8086/health           # {"status":"ok","device":"cpu","targets":[...]}
curl -s http://localhost:8086/predict -H 'Content-Type: application/json' \
     -d '{"target":"FXA","window":4,"method":"ensemble","smiles":["CCO"],"format":"compact"}'
printf "smiles\nCCO\nc1ccccc1O\n" > /tmp/t.csv
curl -s -F "file=@/tmp/t.csv" -F "target=FXA" -F "window=4" -F "format=csv" \
     http://localhost:8086/predict_csv
docker rm -f retrop
```

## Endpoints

| method | path | purpose |
|---|---|---|
| GET | `/health` | liveness, device, targets, cache size |
| GET | `/targets` | per target: endpoints, available windows, cut dates, methods, n_train |
| POST | `/score` | REINVENT scorer contract (`{smiles, params:{target,endpoint,window,method}}`) |
| POST | `/predict` | `{"target","window","method","smiles":[...],"format","batch_size"}` |
| POST | `/predict_csv` | multipart CSV upload; form `target`, `window`, `method`, `smiles_col`, `format` |
| GET | `/docs` | Swagger UI |

- `window`: an index or `"latest"` (default). `method`: `ensemble` (default) \| `xgboost` \| `chemprop` \| `chemeleon`.
- Unknown target / unavailable window → **HTTP 400** with the valid options.

### `POST /score` (REINVENT scorer contract)

One TTP objective per target endpoint. Returns **one scalar per input SMILES** (ensemble mean of
the selected endpoint), `null` on failure, strict index alignment. `endpoint` defaults to the
target's sole endpoint; for FXA (pKi/logD/logS) it is required.

```jsonc
// request
{"smiles": ["CCO"], "params": {"target": "FXA", "endpoint": "FXA_pKi", "window": "latest", "method": "ensemble"}}
// response
{"scores": [3.43],
 "metadata": {"target": "FXA", "endpoint": "FXA_pKi", "window": 11, "cut_date": "...", "std": [2.16]},
 "artifacts": {}}
```

Example TTP objective:

```json
{"name": "FXA potency", "type": "external", "service": "retrodmta",
 "service_params": {"target": "FXA", "endpoint": "FXA_pKi", "window": "latest"},
 "weight": 0.5, "transform": {"type": "sigmoid", "low": 5, "high": 9, "k": 0.5}}
```

### `POST /predict` response (records, per molecule)
```json
{
  "input_smiles": "CCO", "standardized_smiles": "CCO", "ok": true,
  "target": "FXA", "window": 4, "cut_date": "2004-01-31",
  "predictions": {
    "FXA_pKi":  {"mean": 8.70, "std": 0.49, "n_models": 3, "unit": "pKi (-log10 Ki[M])",
                 "per_model": {"XGBoost": .., "Chemprop-MTL": .., "CheMeleon-MTL": ..}},
    "FXA_logD": {...}, "FXA_logS": {...}
  }
}
```

### Output formats
- `records` (default): list of the above.
- `compact`: `{target, window, cut_date, endpoints, units, molecules:[{input_smiles, ok, mean[], std[]}]}`.
- `table`: `{target, window, cut_date, columns, rows}`.

### Python client
```python
import requests
r = requests.post("http://localhost:8086/predict",
                  json={"target":"FXA","window":4,"method":"ensemble",
                        "smiles": smiles_list, "format":"compact"})
```

## Models

Each `(target, window)` snapshot = expanding-window training on data up to that cut:
- **Chemprop-MTL** / **CheMeleon-MTL**: multitask (7 general ADME tasks + the target endpoints);
  one model returns all of a target's endpoints.
- **XGBoost**: one single-task model per target endpoint (target data only).

`ensemble` = mean ± std across the available methods per endpoint. 27 snapshots total (FXA 11,
MMP8 5, PPARD 7, RENIN 4). These are the prospective temporal-validation models; deploy the
`latest` window per target for production, or select any window for a temporal study.

## Configuration (environment variables)

| var | default | meaning |
|---|---|---|
| `MODELS_DIR` | `/app/models` (image; central `dd_agentic/models/retrodmta` baked in) | model bundle + `manifest.json` |
| `PRED_BATCH` | 1000 | molecules per frozen pass |
| `PRED_DEVICE` | `auto` | `cpu` \| `cuda` \| `auto` |
| `PRED_WORKERS` | min(32, ncpu) | featurization/standardization processes |
| `PRED_LOADER_WORKERS` | 8 | chemprop dataloader workers |
| `PRED_CACHE_MAX` | 6 | resident `(target, window, method)` snapshots (LRU) |

## Performance notes
CPU, models resident: ensemble ~**205 mol/s**, single method (e.g. chemprop) ~**600 mol/s** at
batch 1000. First call for a `(target, window)` loads its 3 models (~4.75 s), then warm. Fix a
`(target, window)` per campaign so the load happens once. Stateless — scale horizontally.

## Limitations
Prospective single-target potency (pKi/pIC50/pEC50) is hard: modest R², high cross-model
spread; use `std` and ranking, not just absolute values. Trained at pH 7.4. No applicability
domain for project models (by design).
