"""Live test conftest — all live tests are skipped unless RCSB_LIVE_TESTS=1.

Set the environment variable to run against the real RCSB APIs:

    RCSB_LIVE_TESTS=1 pytest test_rcsb/live/ -v
"""

import os

import pytest

# Module-level mark (metadata / -m filtering)
pytestmark = pytest.mark.skipif(
    not os.getenv("RCSB_LIVE_TESTS"),
    reason="Set RCSB_LIVE_TESTS=1 to run live RCSB API tests",
)


@pytest.fixture(autouse=True)
def _require_live_env():
    """Enforce the skip at runtime for every test in this directory."""
    if not os.getenv("RCSB_LIVE_TESTS"):
        pytest.skip("Set RCSB_LIVE_TESTS=1 to run live RCSB API tests")
