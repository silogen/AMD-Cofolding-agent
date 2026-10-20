"""Tests for reinvent4_mcp/tools/pubchem.py -- resolving a name to a real SMILES.

The agent must never recall a SMILES from memory, so this lookup failing is not
a minor inconvenience: it is the agent either stopping to ask a human or, worse,
inventing a structure. PubChem throttles under load rather than refusing, which
makes riding that out the difference between the two.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest
import requests

from reinvent4_mcp.tools import pubchem


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    """Backoff is asserted on through the recorded waits, not by waiting."""

    waits: list[float] = []
    monkeypatch.setattr(pubchem.time, "sleep", waits.append)
    return waits


class _Response:
    def __init__(
        self,
        status_code: int = 200,
        body: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        self.status_code = status_code
        self._body = body or {}
        self.headers = headers or {}

    def json(self) -> dict[str, Any]:
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Server Error")


def _found(smiles: str = "CCO", **extra: Any) -> _Response:
    props = {"CID": 702, "SMILES": smiles, "MolecularWeight": "46.07", **extra}
    return _Response(body={"PropertyTable": {"Properties": [props]}})


def _busy(retry_after: Optional[str] = None) -> _Response:
    # What PubChem actually sends under load: PUGREST.ServerBusy.
    headers = {"Retry-After": retry_after} if retry_after else None
    return _Response(status_code=503, headers=headers)


def _responses(monkeypatch, *queue: _Response) -> list[str]:
    """Serve the given responses in order, recording the URLs requested."""

    urls: list[str] = []
    pending = list(queue)

    def _get(url, timeout=None):
        urls.append(url)
        item = pending.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(pubchem.requests, "get", _get)
    return urls


class TestThrottling:
    """503 and 429 are "come back shortly", not "no such compound"."""

    def test_a_busy_server_is_retried_until_it_answers(self, monkeypatch):
        _responses(monkeypatch, _busy(), _busy(), _found("CCO"))

        result = pubchem.fetch_compound("JQ1")

        assert result["smiles"] == "CCO"
        assert "error" not in result

    def test_the_wait_grows_between_attempts(self, monkeypatch, no_sleeping):
        _responses(monkeypatch, _busy(), _busy(), _found())

        pubchem.fetch_compound("JQ1")

        assert no_sleeping == [1.0, 3.0]

    def test_it_gives_up_rather_than_retrying_forever(self, monkeypatch):
        """An endpoint that is genuinely down must not hold the turn open."""
        _responses(monkeypatch, _busy(), _busy(), _busy())

        result = pubchem.fetch_compound("JQ1")

        assert "error" in result
        assert "NOT retrieved" in result["error"]

    def test_rate_limiting_is_retried_too(self, monkeypatch):
        _responses(
            monkeypatch, _Response(status_code=429), _found("CCO")
        )

        assert pubchem.fetch_compound("JQ1")["smiles"] == "CCO"

    def test_the_servers_own_retry_after_is_honoured(self, monkeypatch, no_sleeping):
        _responses(monkeypatch, _busy(retry_after="2"), _found())

        pubchem.fetch_compound("JQ1")

        assert no_sleeping == [2.0]

    def test_an_absurd_retry_after_is_capped(self, monkeypatch, no_sleeping):
        """A half-hour hint would strand an interactive turn."""
        _responses(monkeypatch, _busy(retry_after="1800"), _found())

        pubchem.fetch_compound("JQ1")

        assert no_sleeping == [pubchem._MAX_RETRY_WAIT]

    def test_a_nonsense_retry_after_falls_back_to_the_schedule(
        self, monkeypatch, no_sleeping
    ):
        _responses(monkeypatch, _busy(retry_after="whenever"), _found())

        pubchem.fetch_compound("JQ1")

        assert no_sleeping == [1.0]


class TestNotRetried:

    def test_a_missing_compound_is_answered_at_once(self, monkeypatch, no_sleeping):
        """404 is a real answer; retrying it just delays saying so."""
        urls = _responses(monkeypatch, _Response(status_code=404))

        result = pubchem.fetch_compound("notacompound")

        assert "No PubChem compound found" in result["error"]
        assert len(urls) == 1
        assert no_sleeping == []

    def test_an_empty_name_never_reaches_the_network(self, monkeypatch):
        urls = _responses(monkeypatch)

        assert "error" in pubchem.fetch_compound("  ")
        assert urls == []


class TestTransportFailures:

    def test_a_connection_error_is_retried(self, monkeypatch):
        """A reset mid-handshake is as transient as a 503."""
        _responses(
            monkeypatch, requests.ConnectionError("reset"), _found("CCO")
        )

        assert pubchem.fetch_compound("JQ1")["smiles"] == "CCO"

    def test_an_unreachable_endpoint_is_reported_not_invented(self, monkeypatch):
        """The whole point: no structure is ever fabricated."""
        _responses(
            monkeypatch,
            requests.ConnectionError("refused"),
            requests.ConnectionError("refused"),
            requests.ConnectionError("refused"),
        )

        result = pubchem.fetch_compound("JQ1")

        assert "Could not reach PubChem" in result["error"]
        assert "smiles" not in result


class TestSuccessfulLookup:

    def test_the_isomeric_smiles_is_preferred(self, monkeypatch):
        """Stereochemistry matters downstream for cofolding."""
        _responses(monkeypatch, _found("C[C@H](N)O", ConnectivitySMILES="CC(N)O"))

        result = pubchem.fetch_compound("JQ1")

        assert result["smiles"] == "C[C@H](N)O"
        assert result["connectivity_smiles"] == "CC(N)O"

    def test_a_flat_smiles_is_used_when_there_is_no_isomeric_one(self, monkeypatch):
        props = {"CID": 1, "ConnectivitySMILES": "CC(N)O"}
        _responses(
            monkeypatch,
            _Response(body={"PropertyTable": {"Properties": [props]}}),
        )

        assert pubchem.fetch_compound("x")["smiles"] == "CC(N)O"

    def test_the_name_is_url_quoted(self, monkeypatch):
        urls = _responses(monkeypatch, _found())

        pubchem.fetch_compound("(+)-JQ1")

        assert "%28%2B%29-JQ1" in urls[0]
