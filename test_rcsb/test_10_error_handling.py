"""test_10_error_handling.py — Tests for RCSBHTTPClient error classification.

Each test patches requests.Session.post (or .get) at the session level to
return a synthetic response with the desired status code, then asserts that
the client raises the correct typed RCSBError subclass.

Retry behaviour is also tested: 5xx codes trigger up to MAX_RETRIES retries,
while 4xx codes are not retried.  time.sleep is patched throughout so tests
run instantly.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from rcsb import (
    MAX_RETRIES,
    RCSB_SEARCH_URL,
    RCSBHTTPClient,
    RCSBLocalError,
    RCSBMalformedResponse,
    RCSBNoResultsError,
    RCSBQueryError,
    RCSBRateLimitError,
    RCSBUnavailableError,
)


# ── helper ────────────────────────────────────────────────────────────────────

def make_mock_response(
    status_code: int,
    json_body: dict | None = None,
    text: str = "error body",
    headers: dict | None = None,
) -> MagicMock:
    """Build a minimal mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.headers = {"Content-Type": "application/json", **(headers or {})}
    if json_body is not None:
        resp.json.return_value = json_body
    else:
        resp.json.side_effect = ValueError("no JSON body")
    return resp


@pytest.fixture()
def client() -> RCSBHTTPClient:
    return RCSBHTTPClient()


# ── 4xx → RCSBQueryError (no retries) ────────────────────────────────────────

def test_400_raises_query_error(client):
    with patch.object(client._session, "post",
                      return_value=make_mock_response(400)):
        with pytest.raises(RCSBQueryError):
            client.post_json(RCSB_SEARCH_URL, {})


def test_401_raises_query_error(client):
    with patch.object(client._session, "post",
                      return_value=make_mock_response(401)):
        with pytest.raises(RCSBQueryError):
            client.post_json(RCSB_SEARCH_URL, {})


def test_404_raises_query_error(client):
    with patch.object(client._session, "post",
                      return_value=make_mock_response(404)):
        with pytest.raises(RCSBQueryError):
            client.post_json(RCSB_SEARCH_URL, {})


# ── 429 → RCSBRateLimitError ──────────────────────────────────────────────────

def test_429_raises_rate_limit_error(client):
    with patch.object(client._session, "post",
                      return_value=make_mock_response(429,
                                                      headers={"Retry-After": "5"})), \
         patch("rcsb.time.sleep"):
        with pytest.raises(RCSBRateLimitError):
            client.post_json(RCSB_SEARCH_URL, {})


def test_429_retry_after_parsed(client):
    """Retry-After header '5' → exc.retry_after == 5.0."""
    with patch.object(client._session, "post",
                      return_value=make_mock_response(429,
                                                      headers={"Retry-After": "5"})), \
         patch("rcsb.time.sleep"):
        with pytest.raises(RCSBRateLimitError) as exc_info:
            client.post_json(RCSB_SEARCH_URL, {})
    assert exc_info.value.retry_after == 5.0


# ── 5xx → RCSBUnavailableError ────────────────────────────────────────────────

def test_503_raises_unavailable(client):
    """503 after all retries → RCSBUnavailableError."""
    with patch.object(client._session, "post",
                      return_value=make_mock_response(503)), \
         patch("rcsb.time.sleep"):
        with pytest.raises(RCSBUnavailableError):
            client.post_json(RCSB_SEARCH_URL, {})


def test_502_raises_unavailable(client):
    with patch.object(client._session, "post",
                      return_value=make_mock_response(502)), \
         patch("rcsb.time.sleep"):
        with pytest.raises(RCSBUnavailableError):
            client.post_json(RCSB_SEARCH_URL, {})


# ── malformed / missing response body ─────────────────────────────────────────

def test_malformed_json_raises(client):
    """200 with non-JSON Content-Type and non-JSON body → RCSBMalformedResponse."""
    resp = MagicMock()
    resp.status_code = 200
    resp.text = "not json at all"
    resp.headers = {"Content-Type": "text/html"}
    resp.json.side_effect = ValueError("no JSON")

    with patch.object(client._session, "post", return_value=resp):
        with pytest.raises(RCSBMalformedResponse):
            client.post_json(RCSB_SEARCH_URL, {})


def test_missing_total_count(client):
    """200 with valid JSON but no total_count — HTTP layer returns raw dict (no error)."""
    body = {"result_list": [], "query_id": "abc"}
    resp = MagicMock()
    resp.status_code = 200
    resp.text = '{"result_list": [], "query_id": "abc"}'
    resp.headers = {"Content-Type": "application/json"}
    resp.json.return_value = body

    with patch.object(client._session, "post", return_value=resp):
        result = client.post_json(RCSB_SEARCH_URL, {})
    assert result == body


# ── retry counts ──────────────────────────────────────────────────────────────

def test_no_retries_on_400(client):
    """400 is a permanent error: called exactly once, no retries."""
    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return make_mock_response(400)

    with patch.object(client._session, "post", side_effect=side_effect):
        with pytest.raises(RCSBQueryError):
            client.post_json(RCSB_SEARCH_URL, {})
    assert call_count == 1


def test_no_retries_on_401(client):
    """401 is a permanent error: called exactly once, no retries."""
    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return make_mock_response(401)

    with patch.object(client._session, "post", side_effect=side_effect):
        with pytest.raises(RCSBQueryError):
            client.post_json(RCSB_SEARCH_URL, {})
    assert call_count == 1


def test_retries_on_503(client):
    """503 is retried MAX_RETRIES times before giving up (MAX_RETRIES + 1 calls)."""
    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return make_mock_response(503, text="unavailable")

    with patch.object(client._session, "post", side_effect=side_effect), \
         patch("rcsb.time.sleep"):
        with pytest.raises(RCSBUnavailableError):
            client.post_json(RCSB_SEARCH_URL, {})
    assert call_count == MAX_RETRIES + 1


def test_connection_error_raises_unavailable(client):
    """ConnectionError (e.g. DNS failure, refused) → RCSBUnavailableError."""
    with patch.object(
        client._session,
        "post",
        side_effect=requests.exceptions.ConnectionError("Connection refused"),
    ), patch("rcsb.time.sleep"):
        with pytest.raises(RCSBUnavailableError):
            client.post_json(RCSB_SEARCH_URL, {})


# ── backoff ───────────────────────────────────────────────────────────────────

def test_backoff_sleep_called_on_503(client):
    """time.sleep is called at least once while retrying 503 responses."""
    with patch("rcsb.time.sleep") as mock_sleep, \
         patch.object(client._session, "post",
                      return_value=make_mock_response(503)):
        with pytest.raises(RCSBUnavailableError):
            client.post_json(RCSB_SEARCH_URL, {})
    assert mock_sleep.call_count >= 1, (
        f"Expected at least one sleep call, got {mock_sleep.call_count}"
    )
