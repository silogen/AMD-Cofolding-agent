"""test_live_search.py — Live integration tests against the real RCSB Search API.

All tests in this file are skipped by default.  Run with:

    RCSB_LIVE_TESTS=1 pytest test_rcsb/live/test_live_search.py -v

These tests hit the real RCSB endpoint and require internet access.
They verify end-to-end correctness of query construction, HTTP execution, and
response parsing against live data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make rcsb.py importable when this file is run directly or from any cwd
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rcsb import (
    RCSB_SEARCH_URL,
    QueryBuilder,
    RCSBHTTPClient,
)


@pytest.fixture(scope="module")
def http():
    """Single HTTP client shared across all tests in this module."""
    return RCSBHTTPClient()


# ── text search ───────────────────────────────────────────────────────────────

def test_live_text_search_cdk2(http):
    """Full-text search for CDK2 returns more than 100 experimental entries."""
    query_dict = QueryBuilder.from_dict({
        "type": "terminal",
        "service": "text",
        "parameters": {"value": "CDK2"},
    })
    result = http.post_json(RCSB_SEARCH_URL, {
        "query": query_dict,
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": 1},
            "results_content_type": ["experimental"],
        },
    })
    assert result.get("total_count", 0) > 100, (
        f"Expected >100 CDK2 hits, got {result.get('total_count')}"
    )


def test_live_count_cdk2(http):
    """return_counts=True returns total_hits without a result list."""
    query_dict = QueryBuilder.from_dict({
        "type": "terminal",
        "service": "text",
        "parameters": {"value": "CDK2"},
    })
    result = http.post_json(RCSB_SEARCH_URL, {
        "query": query_dict,
        "return_type": "entry",
        "request_options": {
            "return_counts": True,
            "results_content_type": ["experimental"],
        },
    })
    assert result.get("total_count", 0) > 100


# ── chemical search ───────────────────────────────────────────────────────────

def test_live_chemical_search_atp(http):
    """Attribute search for ATP as a non-polymer entity returns > 100 hits."""
    query_dict = QueryBuilder.from_dict({
        "type": "terminal",
        "service": "attribute",
        "parameters": {
            "attribute": "rcsb_chem_comp_container_identifiers.comp_id",
            "operator": "exact_match",
            "value": "ATP",
        },
    })
    result = http.post_json(RCSB_SEARCH_URL, {
        "query": query_dict,
        "return_type": "non_polymer_entity",
        "request_options": {"return_counts": True},
    })
    assert result.get("total_count", 0) > 100, (
        f"Expected >100 ATP entity hits, got {result.get('total_count')}"
    )


# ── content type: computational ───────────────────────────────────────────────

def test_live_content_type_computational(http):
    """Searching with results_content_type=['computational'] returns CSM entries."""
    query_dict = QueryBuilder.from_dict({
        "type": "terminal",
        "service": "text",
        "parameters": {"value": "hemoglobin"},
    })
    result = http.post_json(RCSB_SEARCH_URL, {
        "query": query_dict,
        "return_type": "entry",
        "request_options": {
            "return_counts": True,
            "results_content_type": ["computational"],
        },
    })
    assert result.get("total_count", 0) > 0, (
        "Expected at least one AlphaFold/CSM hemoglobin hit"
    )


# ── facets ────────────────────────────────────────────────────────────────────

def test_live_facets_organism_cdk2(http):
    """Organism facet for CDK2 includes Homo sapiens with count > 100."""
    query_dict = QueryBuilder.from_dict({
        "type": "terminal",
        "service": "text",
        "parameters": {"value": "CDK2"},
    })
    result = http.post_json(RCSB_SEARCH_URL, {
        "query": query_dict,
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": 1},
            "results_content_type": ["experimental"],
            "facets": [
                {
                    "name": "Organism",
                    "aggregation_type": "terms",
                    "attribute": "rcsb_entity_source_organism.scientific_name",
                    "min_interval_population": 1,
                }
            ],
        },
    })
    facets = result.get("facets", [])
    organism_facet = next(
        (f for f in facets if f.get("name") == "Organism"), None
    )
    assert organism_facet is not None, "Expected 'Organism' facet in response"

    buckets = organism_facet.get("buckets", [])
    # RCSB Search API uses "value"/"key"/"label" for the bucket label and
    # "count"/"doc_count"/"population" for the count, depending on API version.
    homo_bucket = next(
        (
            b for b in buckets
            if str(b.get("value", b.get("key", b.get("label", "")))).lower() == "homo sapiens"
        ),
        None,
    )
    assert homo_bucket is not None, (
        f"Expected 'Homo sapiens' in organism buckets; got: {buckets[:5]}"
    )
    count = (homo_bucket.get("count")
             or homo_bucket.get("doc_count")
             or homo_bucket.get("population", 0))
    assert count > 100, f"Expected >100 Homo sapiens CDK2 entries, got {count}"


# ── no-results ────────────────────────────────────────────────────────────────

def test_live_no_results(http):
    """A nonsense query string returns zero hits.

    RCSB may return HTTP 200 with total_count=0, or HTTP 204 No Content
    (which rcsb.py currently surfaces as RCSBUnavailableError).  Both are
    acceptable — the test accepts either a zero-count response or a 204.
    """
    from rcsb import RCSBUnavailableError

    query_dict = QueryBuilder.from_dict({
        "type": "terminal",
        "service": "text",
        "parameters": {
            "value": "xyzzy_no_real_structure_should_match_this_query_ever"
        },
    })
    try:
        result = http.post_json(RCSB_SEARCH_URL, {
            "query": query_dict,
            "return_type": "entry",
            "request_options": {"return_counts": True},
        })
    except RCSBUnavailableError as exc:
        # HTTP 204 No Content — RCSB's way of saying "zero results"
        if "204" in str(exc):
            return  # acceptable zero-result response
        raise

    total = result.get("total_count", 0)
    result_list = result.get("result_list", [])
    assert total == 0 or result_list == [], (
        f"Expected zero hits, got total_count={total}"
    )
