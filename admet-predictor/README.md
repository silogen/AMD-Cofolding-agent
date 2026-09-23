# admet-predictor

Manifest-driven **ADMET** prediction with two selectable model stacks:

- `drug` (default): 17 pharmaceutical endpoints
- `agro`: 25 endpoints (the drug set plus 8 agrochemical endpoints)

Each endpoint is a consensus mean ± std over available **XGBoost** and
**Chemprop-MTL** members. `pharma` is accepted as an input alias for `drug`
but is never emitted. Endpoint names, units, conditions, and model assets are
read from `models/admet/<model_set>/manifest.json`.

Chemprop v2 checkpoints return predictions in endpoint units; the runtime does
not apply the exported training scaler a second time. Before consensus, every
member is checked against endpoint-specific physical bounds. Impossible members
are reported under `excluded_models` and omitted from the mean. The retained
consensus is separately flagged against the endpoint's training-data p01–p99
response range; this response-domain warning does not suppress the prediction.

Input SMILES are standardized with the **same pipeline used for training** (desalt →
stereo-preserving re-aromatization → Normalizer → canonical tautomer → dimorphite pH 7.4
dominant species → N-charge check). See [SERVICE.md](SERVICE.md) for deployment, endpoints,
model details, and performance.

**Role:** this is both a **REINVENT scoring component** (`POST /score`, one objective per
endpoint — see [SERVICE.md](SERVICE.md)) and an **agent analysis service** (`POST /predict`
for the full per-endpoint records). It follows the boltz2/posebusters convention: a single
FastAPI app in [../services/admet-predictor/app.py](../services/admet-predictor/app.py).

> Layout + `/score` scorer contract follow the PR #121 review; per-comment original-vs-new
> corrections are in [../interaction-profiler/REVIEW_RESPONSE.md](../interaction-profiler/REVIEW_RESPONSE.md).

## Install

Model weights are **not** in git. First restore the central model dir from the SharePoint zip:

```bash
# from dd_agentic/ : unzip so that dd_agentic/models/admet/ and dd_agentic/models/retrodmta/ exist
unzip dd_agentic_models.zip          # provides ./models/
```

Then build (models are baked into the image from `dd_agentic/models/admet`):

```bash
# from dd_agentic/
docker build -f services/admet-predictor/Dockerfile -t admet-predictor:latest .
```

<details><summary>Local dev only (no Docker)</summary>

```bash
# from dd_agentic/
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
pip install ./admet-predictor
export MODELS_DIR=$PWD/models/admet   # central model dir (restore from SharePoint zip)
```
</details>

## Quickstart

```bash
# CLI (installed as a console script)
admet-predict --smiles "CC(=O)Oc1ccccc1C(=O)O" "CCN(CC)CC"

# from a CSV
admet-predict --csv input.csv --smiles-col smiles --out preds.json

# condensed output for many molecules
admet-predict --csv input.csv --format compact --out preds.json
```

```python
from admet_predictor import core
recs = core.predict(["CC(=O)Oc1ccccc1C(=O)O", "CCN(CC)CC"])          # records (default)
compact = core.predict(smiles_list, fmt="compact")                    # metadata once + arrays
```

## Tests

```bash
# 1) build self-test (predicts a molecule; build fails if broken)
docker build -f services/admet-predictor/Dockerfile -t admet-predictor:latest .   # from dd_agentic/

# 2) CLI smoke test
docker run --rm admet-predictor:latest admet-predict --smiles "CC(=O)Oc1ccccc1C(=O)O" --format compact

# 3) server test
docker run -d --name admetp -p 8084:8084 admet-predictor:latest
curl -s http://localhost:8084/health
curl -s http://localhost:8084/predict -H 'Content-Type: application/json' \
     -d '{"smiles":["CCO","c1ccccc1O"],"format":"compact"}'
docker rm -f admetp
```

Sanity: aspirin `logp` ≈ **1.2**, `AD Pass: true`; identical input → identical output
(deterministic); a failed SMILES returns `"ok": false` (never dropped, order preserved).

## CLI

| flag | default | meaning |
|---|---|---|
| `--smiles S [S ...]` | — | one or more SMILES |
| `--csv PATH` | — | CSV input (combined with `--smiles`) |
| `--smiles-col` | `smiles` | SMILES column in the CSV |
| `--out PATH` | stdout | write JSON here |
| `--batch-size` | 1000 | molecules per frozen pass |
| `--format` | `records` | `records` \| `compact` \| `table` |

## Options (environment variables)

| var | default | meaning |
|---|---|---|
| `MODELS_DIR` | `models` | location of the baked model files |
| `PRED_BATCH` | 1000 | molecules per frozen pass (raise if headroom, lower if tight) |
| `PRED_DEVICE` | `auto` | `cpu` \| `cuda` \| `auto` for the GNN forward |
| `PRED_WORKERS` | min(32, ncpu) | CPU processes for standardization + featurization |
| `PRED_LOADER_WORKERS` | 8 | chemprop graph-featurization workers |

## Output

Three shapes via `format` (see [SERVICE.md](SERVICE.md) for the full schema):

- **`records`** (default) — list, one verbose object per molecule (full `per_model`, units, AD).
- **`compact`** — metadata (`endpoints`/`units`/`conditions`/`models`) declared once + per
  molecule `mean[]`/`std[]`/`response_ad_pass[]` arrays + `ad_pass`. ~5–10×
  smaller for large N.
- **`table`** — `columns` + `rows` (drop straight into a DataFrame). CSV via `POST /predict_csv`.

Every result carries the consensus `mean` (use this), cross-model `std` (uncertainty proxy),
chemical-space `applicability_domain` (Avalon Tanimoto), physical-member validation,
and a per-endpoint `response_domain` derived from the training p01–p99 range.

## License

MIT.
