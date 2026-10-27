# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Background job management for REINVENT4 optimization runs.

Provides subprocess-based job tracking, TOML config generation,
score trajectory monitoring, and checkpoint management.
"""

from reinvent4_mcp.jobs.manager import JobManager, JobState

__all__ = ["JobManager", "JobState"]
