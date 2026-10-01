# Review response — PR #3 (01-data-structure-foundation)

One entry per review comment: concern, original, new, why.  All changes are additive
robustness fixes; `tests/test_chembl.py` passes (59 passed, +5 new) and the RCSB suite is
unchanged by these edits (its pre-existing fixture-dependent failures are not touched).

---

## `reinvent4_mcp/tools/chembl.py`

### 1. `Retry-After` can crash the retry loop (HTTP-date form)

> "Per RFC 7231, Retry-After may be an HTTP-date … float() on a date string raises
> ValueError, which is not caught … a well-behaved server sending a date header would
> blow up the whole call."

Original:
```python
retry_after = float(resp.headers.get("Retry-After", 2 ** attempt))
```
New — `_parse_retry_after()` accepts delta-seconds or an HTTP-date and falls back to
backoff on anything unparseable:
```python
def _parse_retry_after(raw, fallback):
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        ...
        dt = parsedate_to_datetime(raw)   # HTTP-date → seconds
        ...
    return fallback
...
wait = _parse_retry_after(resp.headers.get("Retry-After"), 2 ** attempt)
```

### 2. Typed error lost on the final retry

> "On the last attempt, a 429/503/etc. skips both retry branches … falls through to
> `if not resp.ok:` → raises a generic ChEMBLError. So last_exc (the specific
> ChEMBLRateLimitError/ChEMBLUnavailableError) is discarded."

New — the transient-status block now raises the typed `last_exc` once retries are
exhausted, instead of falling through to the generic error:
```python
if resp.status_code in _RETRY_STATUSES:
    last_exc = ChEMBLRateLimitError(...) if resp.status_code == 429 \
               else ChEMBLUnavailableError(...)
    if attempt < _MAX_RETRIES:
        time.sleep(wait); continue
    raise last_exc          # typed: rate_limit / service_unavailable
```

### 3. `ConnectionError` isn't retried

> "Timeouts retry, but a transient ConnectionError (DNS blip, dropped socket) raises
> immediately. Those are often just as transient as a 503."

Original: `except requests.ConnectionError: raise ChEMBLUnavailableError(...)`.
New: a `ConnectionError` sets `last_exc` and retries with backoff (like `Timeout`),
raising the typed error only after retries are exhausted.

### 4. Bulk SMILES fetch can exceed URL length limits

> "`_fetch_smiles_bulk` puts up to 1000 IDs into `molecule_chembl_id__in` … ~13 KB of URL
> — over the common 8 KB cap … a failed batch is swallowed … you could silently get
> mostly None SMILES."

Original `BATCH = 1000` → New `BATCH = 200` (~2.6 KB/query, safely under the limit).

### 5. Negative limit not clamped

> "`_paginate` does min(int(limit), _MAX_LIMIT) — a negative limit slips through … Add
> max(1, …)."

New: `params["limit"] = max(1, min(int(limit), _MAX_LIMIT))`.

### 6. `fetch_all_activities` over-fetches past the cap

> "you extend a full page, then check offset >= cap, so you can exceed max_records by up
> to page_size-1."

New: after the loop, `all_records = all_records[:cap]` so `max_records` is a true ceiling.

### 7. Dead code

> "`_MAX_PAGES`, `urlencode`, the `_validate_chembl_id` helper, and `math` inside
> `fetch_all_activities` are all defined/imported but never used."

New: removed `_MAX_PAGES`, the `urlencode` import, `_validate_chembl_id`, and the unused
`math` import.  (The runaway-pagination guard is `max_records`, now a hard cap — see #6.)

---

## `reinvent4_mcp/tools/rcsb.py`

> "Same points as for chembl … robustness check especially for input edge cases, dead
> code, correctness bugs that can fail silently."

The RCSB client already retries `ConnectionError` and raises typed errors on exhaustion
(`_classify_error`), and transient-status exhaustion already raised typed errors — so the
chembl #2/#3 issues did not apply here.  The one matching gap was the `Retry-After`
header: it caught the date form but silently fell back to a fixed 1 s.

New — `_parse_retry_after` now parses the HTTP-date form too, matching chembl:
```python
try:
    return max(0.0, float(header))
except (ValueError, TypeError):
    dt = parsedate_to_datetime(header)   # HTTP-date → seconds
    ...
```

---

## `services/dd-agent/Dockerfile` — prior names can change (resolved by design)

> "we saw in REINVENT zenodo directory that over the years the names of the priors can
> change. Maybe good to add some robustness checks for the nearest applicable file."

Resolved by the existing design rather than a heuristic: the build pins to the **immutable**
Zenodo record 15641297 and verifies each prior's published MD5.  A renamed/missing/
truncated file therefore fails the build loudly (wget error or `md5sum -c` mismatch)
instead of silently grabbing a wrong or incompatible prior at runtime — which is safer
than a "nearest applicable file" fallback that could pick a mismatched model.

---

## `services/structure-prep/*` — error handling (verified present)

> curation.py / additives.py / models.py: "make sure error handling is done properly
> somewhere in pipeline"; "json missing error handling".

Verified: `AdditivesRegistry._load` already guards the registry JSON — a missing file logs
a warning and yields an empty registry, and a malformed file is caught with
`logger.exception` (never crashes the pipeline).  No change required.
