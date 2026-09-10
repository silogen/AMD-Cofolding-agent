# Review response — PR #121 (REFM-872: interaction-profiler service)

One entry per review comment: the concern, the **original** code, the **new** code, and why.

Scope note: the layout/convention fixes below were applied to **all three** services that
shared the interaction-profiler layout — `interaction-profiler`, `admet-predictor`,
`retrodmta-predictor`. The two predictor services additionally implement the `POST /score`
scorer contract and are wired into `config.py` / `mapper.py` / `docker-compose.yml`.
The **interaction-profiler stays an out-of-band analysis service by design** (documented), so
its scorer-related comments are resolved by that declaration rather than by adding `/score`.

---

## 1. Reviewer (Guner) — integration: `/score` contract + wiring

> "it can't be used as a scorer … the scorer calls `POST /score` … `{smiles, params}` →
> `{scores, metadata, artifacts}` … not wired into compose/config/mapper."

**interaction-profiler:** kept out-of-band; documented explicitly (see comment 5).

**admet-predictor / retrodmta-predictor:** added the `/score` contract + wiring.

Original: services exposed only `/predict` (not the scorer contract), and nothing registered them.

New — `services/admet-predictor/app.py`:
```python
class ScoreRequest(BaseModel):
    smiles: list[str]
    params: ScoreParams = Field(default_factory=ScoreParams)

class ScoreResponse(BaseModel):
    scores: list[Optional[float]]
    metadata: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)

@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    ep = req.params.endpoint
    if ep not in P.ENDPOINTS:
        raise HTTPException(status_code=400, detail=f"unknown endpoint {ep!r}; have {P.ENDPOINTS}")
    recs = P.predict(req.smiles, batch_size=req.params.batch_size or P.BATCH_SIZE)
    scores = [(r["predictions"].get(ep, {}).get("mean") if r["ok"] else None) for r in recs]
    return ScoreResponse(scores=scores, metadata={...})
```

New — wiring (`reinvent4_mcp/config.py`):
```python
class AdmetConfig(BaseModel):
    enabled: bool = Field(default=True, ...)
    service_url: str = Field(default="http://admet-predictor:8084/score", ...)
# ... added to ScoringConfig and enabled_external_services (keys "admet", "retrodmta")
```
`reinvent4_mcp/ttp/mapper.py` gained `_get_retrodmta_defaults()` + a branch in
`_map_external_objective`; `docker-compose.yml` adds both to `reinvent4-mcp.depends_on`.

Why: matches the boltz2/posebusters contract so REINVENT's `ExternalRESTScorer` can drive them
as scoring components; one TTP objective per endpoint.

---

## 2. Reviewer — "two different app.py where one imports the other"

> "keep it simple and have everything under `services/<name>/app.py`. boltz2 and posebusters
> are good examples."

Original — `services/interaction-profiler/app.py` was a thin re-export:
```python
from interaction_profiler.service.app import app, create_app, main  # noqa: F401
```

New: the FastAPI app now lives **only** in `services/interaction-profiler/app.py` (single app,
importing library non-app modules `interaction_profiler.core/io/schemas/...`). The package
`service/` submodule was deleted. Same change for admet/retrodmta (`services/<name>/app.py` is
the sole app; `*/src/*/service/` removed; the `*-serve` console scripts dropped from
`pyproject.toml`).

Why: one app per service (boltz2/posebusters convention); no double indirection.

---

## 3. Reviewer — request schema would 422 a REINVENT payload

> "Request schema is `{batch_id, molecules, options}` with `extra="forbid"`, a REINVENT payload
> wouldn't degrade gracefully, it'd 422. If a `/score` route is added, it needs its own request
> model accepting `smiles` + `params`."

Original (`AnalyzeRequest`):
```python
class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    batch_id: str = ...
    molecules: list[MoleculeSpec] = ...
    options: AnalyzeOptions = ...
```

New: the analysis endpoints keep their schema (out-of-band). The **scorer** services define a
dedicated `ScoreRequest{smiles, params}` (see comment 1). interaction-profiler is documented as
not a scorer, so a REINVENT payload is never sent to it.

Why: the scorer contract needs `{smiles, params}`; conflating it with the analysis schema is
what caused the 422 concern.

---

## 4. Reviewer — `profile_molecule_all` is not 1:1 with input

> "can emit multiple `MoleculeResult`s per input (e.g. `all_ligands=True`) … a scorer must
> return exactly one score per input SMILES in order, with None for failures."

interaction-profiler: not a scorer, so no 1:1 requirement (documented).

admet/retrodmta `/score`: **one score per input, index-aligned**, because `core.predict`
returns exactly one record per input SMILES in order:
```python
recs = P.predict(req.smiles, ...)                     # len(recs) == len(req.smiles)
scores = [(r["predictions"].get(ep, {}).get("mean") if r["ok"] else None) for r in recs]
```

Why: strict alignment + `None` on failure is the scorer contract (mirrors boltz2/posebusters
`valid_indices` bookkeeping; here 1:1 holds by construction).

---

## 5. Reviewer — add `/score` or state out-of-band; and scalar-objective bias

> "add a `/score` route … or state explicitly that this service is out-of-band" and
> "`more interactions = higher score` can bias RL toward larger/greasier molecules; the scoring
> function is a science decision."

interaction-profiler — **documented out-of-band** (README + SERVICE.md + module docstring):
```text
Role: out-of-band analysis service — NOT a REINVENT RL scorer. It deliberately does not
implement the POST /score contract ... The agent consumes /analyze and /profile out-of-band.
A /score route + a defensible scalar objective could be added later.
```

admet/retrodmta — avoid a baked objective by returning the **raw predicted value** for the
selected endpoint; the TTP `transform`/`weight` own the direction:
```python
# /score returns raw endpoint mean; no "more = better" heuristic in the service
scores = [(r["predictions"].get(ep, {}).get("mean") if r["ok"] else None) for r in recs]
```

Why: the scalar objective / bias is a science decision; for the predictors it is deferred to the
TTP; for the profiler it is out of scope (out-of-band).

---

## 6. Reviewer (README) — `cd ~/interaction-profiler`

> suggested change: `cd ~/interaction-profiler` → `cd interaction-profiler`

Original:
```bash
cd ~/interaction-profiler
```
New:
```bash
cd interaction-profiler
```
Why: relative path; the home-dir path was machine-specific.

---

## 7. Reviewer (README) — "should not need pip installs since deps in docker compose"

Original (Install led with pip):
```bash
pip install -e ".[dev]"
pip install -e ".[service]"
```
New: Install now leads with Docker/compose (deps are in the image); pip is demoted to a
collapsed "Local dev only (no Docker)" block. Same reframing in admet/retrodmta READMEs.

Why: normal use is via `docker compose`; manual pip installs shouldn't be the headline path.

---

## 8. Copilot (HIGH) — receptor cache reuses wrong pH when `IP_CACHE_DIR` unset

> "In that mode the output path is always `<src>_prepared.pdb` and `fresh` only checks mtime, so
> calling `prepare_receptor(..., ph=6.5)` after a prior `ph=7.4` run will silently reuse the
> wrong protonation state."

Original — `prep/receptor.py`:
```python
if not cache_dir:
    return src.with_name(f"{src.stem}_prepared.pdb"), False
```
New:
```python
if not cache_dir:
    return src.with_name(f"{src.stem}_prepared_ph{ph:g}.pdb"), False
```
Why: the pH is now in the filename, so different pH → different file; a run at another pH cannot
silently reuse the wrong protonation state via the mtime-only freshness check.

---

## 9. Copilot (MEDIUM) — hashing loads the whole receptor into memory

> "`src.read_bytes()` loads the entire structure into memory … streaming the file in chunks
> avoids that."

Original:
```python
h.update(src.read_bytes())
```
New:
```python
with src.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
        h.update(chunk)
```
Why: bounded memory for large receptors; identical content-addressed cache key.

---

## Already-resolved / out of scope

- The Copilot items marked **"Outdated"** (`schemas.py`, `vocabulary.py`, `io/structure.py`, and
  earlier `service/app.py` findings) were already fixed by the "Potential fix for pull request
  finding" commits; no further action.
- interaction-profiler is intentionally **not** wired into `config.py` / `mapper.py` /
  `docker-compose` scoring (out-of-band); the reviewer noted he will handle its pipeline wiring
  post-merge if it ever becomes a scorer.
