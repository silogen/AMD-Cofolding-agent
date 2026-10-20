"""Tests for reinvent4_mcp/tools/campaign.py -- the conductor's own view.

These are the calls an interactive conductor makes to answer "what is the
fleet doing" and "where did it put the output". They read through the ledger,
so what matters here is the shaping on top of it: a dispatch verdict that
names its binding limit, and paths a person can actually open.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from reinvent4_agent.campaign_client import StoreUnavailable
from reinvent4_mcp.tools import campaign as tools


class FakeStore:
    """Enough of CampaignStoreClient for the tools to read through."""

    def __init__(
        self,
        pool: dict[str, Any] | None = None,
        tasks: list[dict[str, Any]] | None = None,
        results: list[dict[str, Any]] | None = None,
        status: dict[str, Any] | None = None,
    ) -> None:
        self._pool = pool or {
            "supervisors": [], "n_alive": 0, "idle": 0, "busy": 0, "by_state": {},
        }
        self._tasks = tasks or []
        self._results = results or []
        self._status = status or {
            "campaign": {"concurrency": 2}, "in_flight": [],
        }

    def supervisors(self, campaign_id=None):
        return self._pool

    def tasks(self, campaign_id, state=None):
        return self._tasks

    def results(self, campaign_id, full=False):
        return self._results

    def status(self, campaign_id):
        return self._status


@pytest.fixture
def store(monkeypatch):
    """Bind the tools to a fake ledger, and return a setter for its contents."""

    holder: dict[str, FakeStore] = {"store": FakeStore()}
    monkeypatch.setattr(tools, "_store", lambda: holder["store"])

    def _use(**kwargs):
        holder["store"] = FakeStore(**kwargs)
        return holder["store"]

    return _use


def _pool(idle=0, busy=0, supervisors=None):
    return {
        "supervisors": supervisors or [],
        "n_alive": idle + busy,
        "idle": idle,
        "busy": busy,
        "by_state": {"idle": idle, "busy": busy},
    }


# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------


class TestListSupervisors:

    def test_the_whole_fleet_needs_no_campaign(self, store):
        """A conductor asks "how many do I have" before it has a campaign."""
        store(pool=_pool(idle=2, busy=1))

        pool = tools.list_supervisors()

        assert (pool["idle"], pool["busy"], pool["n_alive"]) == (2, 1, 3)

    def test_one_campaign_also_gets_its_in_flight_tasks(self, store):
        store(
            pool=_pool(idle=1, busy=1),
            status={
                "campaign": {"concurrency": 3},
                "in_flight": [{"task_id": "t1", "owner": "s1"}],
            },
        )

        pool = tools.list_supervisors("feed0001")

        assert pool["n_in_flight"] == 1
        assert pool["concurrency"] == 3
        assert pool["idle"] == 1

    def test_an_unreachable_store_is_an_error_not_a_zero(self, store, monkeypatch):
        """Reporting "no supervisors" for an outage would read as a drained pool."""
        class Dead(FakeStore):
            def supervisors(self, campaign_id=None):
                raise StoreUnavailable("connection refused")

        monkeypatch.setattr(tools, "_store", Dead)

        assert "error" in tools.list_supervisors()


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------


class TestPoolCapacity:
    """Whether to dispatch, and which of the two limits said no.

    Either number alone misleads: an idle supervisor handed work while the
    scorer is saturated only moves the wait, and a quiet scorer is no use
    with nothing alive to dispatch to.
    """

    @staticmethod
    def _pressure(value):
        return patch("reinvent4_agent.conductor.queue_pressure", return_value=value)

    def test_a_free_supervisor_and_a_quiet_scorer_is_a_yes(self, store):
        store(pool=_pool(idle=1))

        with self._pressure(0.5):
            capacity = tools.get_pool_capacity()

        assert capacity["can_dispatch"] == "yes"
        assert capacity["binding_limit"] is None

    def test_a_busy_pool_is_a_no_even_when_the_scorer_is_quiet(self, store):
        store(pool=_pool(idle=0, busy=3))

        with self._pressure(0.0):
            capacity = tools.get_pool_capacity()

        assert capacity["can_dispatch"] == "no"
        assert capacity["binding_limit"] == "every supervisor is busy"

    def test_a_saturated_scorer_is_a_no_even_with_a_free_supervisor(self, store):
        store(pool=_pool(idle=2))

        with self._pressure(5.0):
            capacity = tools.get_pool_capacity()

        assert capacity["can_dispatch"] == "no"
        assert capacity["binding_limit"] == "boltz is saturated"

    def test_an_unreachable_gateway_is_unknown_rather_than_free(self, store):
        store(pool=_pool(idle=1))

        with self._pressure(None):
            capacity = tools.get_pool_capacity()

        assert capacity["can_dispatch"] == "unknown"
        assert capacity["scorer"]["queue_pressure"] is None

    def test_the_two_numbers_are_both_reported(self, store):
        """So a conductor can explain its own verdict."""
        store(pool=_pool(idle=1, busy=2))

        with self._pressure(1.5):
            capacity = tools.get_pool_capacity()

        assert capacity["supervisors"] == {"idle": 1, "busy": 2, "alive": 3}
        assert capacity["scorer"]["queue_pressure"] == 1.5


# ---------------------------------------------------------------------------
# Where the files are
# ---------------------------------------------------------------------------


class TestCampaignFiles:
    """The ledger says what a task achieved; this says where it wrote it."""

    @pytest.fixture
    def jobs_root(self, tmp_path, monkeypatch):
        root = tmp_path / "jobs"
        root.mkdir()

        class _Jobs:
            working_dir = str(root)

        class _Config:
            jobs = _Jobs()

        monkeypatch.setattr("reinvent4_mcp.config.get_config", lambda: _Config())
        return root

    @staticmethod
    def _job(root: Path, job_id: str, **files: str) -> Path:
        job_dir = root / f"job_{job_id}"
        job_dir.mkdir()
        for name, body in files.items():
            (job_dir / name.replace("__", ".")).write_text(body)
        return job_dir

    def test_a_finished_task_points_at_its_directory(self, store, jobs_root):
        self._job(jobs_root, "j1", results_1__csv="SMILES,Score\nCCO,0.5\n")
        store(
            tasks=[{"task_id": "t1", "state": "done", "created_at": "1"}],
            results=[{"task_id": "t1", "job_id": "j1"}],
        )

        files = tools.get_campaign_files("feed0001")

        entry = files["tasks"][0]
        assert entry["directory"] == str(jobs_root / "job_j1")
        assert entry["job_id"] == "j1"

    def test_the_notable_files_are_named_with_their_sizes(self, store, jobs_root):
        self._job(
            jobs_root, "j1",
            results_1__csv="SMILES,Score\n", reinvent__log="step 1\n",
            config__toml="[run]\n", tb_logs_0="noise",
        )
        store(
            tasks=[{"task_id": "t1", "state": "done", "created_at": "1"}],
            results=[{"task_id": "t1", "job_id": "j1"}],
        )

        named = {
            f["name"]: f["bytes"]
            for f in tools.get_campaign_files("feed0001")["tasks"][0]["files"]
        }

        assert set(named) == {"results_1.csv", "reinvent.log", "config.toml"}
        assert named["results_1.csv"] == len("SMILES,Score\n")

    def test_a_task_that_never_started_a_job_says_so(self, store, jobs_root):
        """Rather than pointing at a directory that was never written."""
        store(tasks=[{"task_id": "t1", "state": "failed", "created_at": "1"}])

        entry = tools.get_campaign_files("feed0001")["tasks"][0]

        assert entry["directory"] is None
        assert "did not get as far" in entry["note"]

    def test_a_job_whose_directory_is_gone_is_distinguished(self, store, jobs_root):
        """A cleaned volume is a different problem from a task that never ran."""
        store(
            tasks=[{"task_id": "t1", "state": "done", "created_at": "1"}],
            results=[{"task_id": "t1", "job_id": "vanished"}],
        )

        entry = tools.get_campaign_files("feed0001")["tasks"][0]

        assert entry["directory"].endswith("job_vanished")
        assert "directory is gone" in entry["note"]

    def test_tasks_are_newest_first_and_bounded(self, store, jobs_root):
        store(tasks=[
            {"task_id": f"t{i}", "state": "done", "created_at": f"{i:02d}"}
            for i in range(30)
        ])

        files = tools.get_campaign_files("feed0001", limit=5)

        assert files["n_tasks"] == 30
        assert files["n_listed"] == 5
        assert [t["task_id"] for t in files["tasks"]] == [
            "t29", "t28", "t27", "t26", "t25"
        ]

    def test_it_says_where_the_structures_are_instead(self, store, jobs_root):
        """Cofold outputs are on the shared volume, not in the job directory."""
        store(tasks=[])

        files = tools.get_campaign_files("feed0001")

        assert "list_shared_assets" in files["hint"]
        assert files["jobs_root"] == str(jobs_root)
