# retrodmta-predictor

Per-target, per-temporal-window **project-endpoint** prediction (RetroDMTA). Companion to
`admet-predictor` (general ADMET): use the general models for ADMET, these for project
(target) endpoints. See [SERVICE.md](SERVICE.md) for deployment, endpoints, and details.

You pick a **target** + **window**; each `(target, window)` snapshot bundles three models
trained on data up to that time cut:

- **Chemprop-MTL** and **CheMeleon-MTL** (multitask: 7 general ADME tasks + the target's endpoints)
- one single-task **XGBoost** per target endpoint

`method` = `ensemble` (mean ± std of the three, default) | `xgboost` | `chemprop` | `chemeleon`.

| target | endpoints (one multitask model covers all) | windows |
|---|---|---|
| FXA | pKi, logD, logS | 11 |
| MMP8 | pIC50 | 5 |
| PPARD | pEC50 | 7 |
| RENIN | pIC50 | 4 |

All 27 windows are baked in and selectable by index; `window` defaults to `latest` (most data).
Models load once per `(target, window)` and stay resident (LRU cache). No applicability domain.

**Role:** both a **REINVENT scoring component** (`POST /score`, one objective per target
endpoint — see [SERVICE.md](SERVICE.md)) and an **agent analysis service** (`POST /predict`).
Single FastAPI app in [../services/retrodmta-predictor/app.py](../services/retrodmta-predictor/app.py)
(boltz2/posebusters convention).

> Layout + `/score` scorer contract follow the PR #121 review; per-comment original-vs-new
> corrections are in [../interaction-profiler/REVIEW_RESPONSE.md](../interaction-profiler/REVIEW_RESPONSE.md).

## Install

Model weights are **not** in git. First restore the central model dir from the SharePoint zip
(so `dd_agentic/models/retrodmta/` exists), then build (models baked in from `dd_agentic/models/retrodmta`):

```bash
# from dd_agentic/
unzip dd_agentic_models.zip          # provides ./models/
docker build -f services/retrodmta-predictor/Dockerfile -t retrodmta-predictor:latest .
```

<details><summary>Local dev only (no Docker)</summary>

```bash
# from dd_agentic/
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
pip install ./retrodmta-predictor
export MODELS_DIR=$PWD/models/retrodmta   # central model dir (restore from SharePoint zip)
```
</details>

## Quickstart

```bash
# CLI
retrodmta-predict --target FXA --window 4 --method ensemble --smiles "CCO"
retrodmta-predict --target FXA --csv input.csv --format table --out fxa.json   # window=latest
```

```python
from retrodmta_predictor import core
recs = core.predict("FXA", ["CCO", "c1ccccc1O"], window=4, method="ensemble")
compact = core.predict("FXA", smiles_list, window="latest", fmt="compact")
```

## Tests

```bash
# 1) build self-test (loads a target/window + predicts)
docker build -f services/retrodmta-predictor/Dockerfile -t retrodmta-predictor:latest .  # from dd_agentic/

# 2) CLI smoke test
docker run --rm retrodmta-predictor:latest retrodmta-predict \
    --target FXA --window 4 --method ensemble --smiles "CCO" --format table

# 3) server test
docker run -d --name retrop -p 8086:8085 retrodmta-predictor:latest
curl -s http://localhost:8086/health
curl -s http://localhost:8086/targets
curl -s http://localhost:8086/predict -H 'Content-Type: application/json' \
     -d '{"target":"FXA","window":4,"method":"ensemble","smiles":["CCO"]}'
docker rm -f retrop
```

Sanity: an FXA-like inhibitor at window 4 → `FXA_pKi` ≈ **8.7**; a bad target/window → HTTP 400
with the valid options; identical input → identical output.

## CLI

| flag | default | meaning |
|---|---|---|
| `--target` | (required) | FXA \| MMP8 \| PPARD \| RENIN |
| `--window` | `latest` | window index or `latest` |
| `--method` | `ensemble` | ensemble \| xgboost \| chemprop \| chemeleon |
| `--smiles` / `--csv` / `--smiles-col` | — | inputs |
| `--out` | stdout | write JSON here |
| `--format` | `records` | records \| compact \| table |

## Options (environment variables)

| var | default | meaning |
|---|---|---|
| `MODELS_DIR` | `models` | baked model bundle |
| `PRED_BATCH` | 1000 | molecules per frozen pass |
| `PRED_DEVICE` | `auto` | `cpu` \| `cuda` \| `auto` |
| `PRED_WORKERS` | min(32, ncpu) | featurization/standardization processes |
| `PRED_LOADER_WORKERS` | 8 | chemprop dataloader workers |
| `PRED_CACHE_MAX` | 6 | resident `(target, window, method)` snapshots (LRU) |

## Output

Same `format` options as `admet-predictor` (`records` | `compact` | `table`); the endpoints are
the target's, each with consensus `mean`/`std` + `per_model`. No AD block. Order preserved;
failed SMILES → `"ok": false`.

## License

MIT.
