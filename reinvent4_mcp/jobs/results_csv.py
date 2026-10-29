# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reading the CSV REINVENT writes while a run is still going.

REINVENT fixes the header on the first batch and then appends one row per
molecule. A scoring component that starts returning extra fields part way
through a run -- an external service that was unreachable and came back, say
-- widens every later row without rewriting the header. pandas then refuses
the file from that row on, so a perfectly healthy run reports no step, no
score and no candidates. Read the shape we are actually given instead: name
the fields we recognise, park the surplus, and carry on.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Names REINVENT has used for the score aggregated over all components.
#: "Score" is what the RL stage writes; "total_score" comes from the scoring
#: API, and both reach us depending on how the run was started.
TOTAL_SCORE_COLUMNS = ("total_score", "Score", "score")

#: Written last on every row, which makes it the one field whose position is
#: still trustworthy when a row carries more fields than the header names.
STEP_COLUMNS = ("Step", "step", "batch")

#: What a component writes where it has no number. pandas reads these as
#: missing when it parses a file itself, so do the same when we parse one:
#: a column of nothing but these is an empty measurement, not text.
MISSING_VALUES = frozenset({"", "nan", "NaN", "NAN", "None", "null", "NULL", "NA", "N/A", "-"})

#: Identity and bookkeeping columns, plus the three likelihoods REINVENT
#: reports next to the score. None of them is a scoring component.
NON_COMPONENT_COLUMNS = frozenset(
    {"SMILES", "SMILES_state", "Input_SMILES", "Scaffold", "Agent", "Prior", "Target"}
    | set(TOTAL_SCORE_COLUMNS)
    | set(STEP_COLUMNS)
)


def total_score_column(df: Any) -> str | None:
    """The column holding the aggregated score, under whichever name."""

    for name in TOTAL_SCORE_COLUMNS:
        if name in df.columns:
            return name
    return None


def step_column(df: Any) -> str | None:
    """The column holding the step number, under whichever name."""

    for name in STEP_COLUMNS:
        if name in df.columns:
            return name
    return None


def read_results_csv(path: Path) -> Any:
    """A REINVENT results CSV as a frame, however ragged it has become.

    An empty file comes back as an empty frame rather than an error: it means
    the run has not written its first batch yet, or we caught it mid-write.
    """

    import pandas as pd

    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except pd.errors.ParserError as exc:
        logger.info("Re-reading %s as ragged: %s", path, exc)
        return _read_ragged(path)

    # pandas only complains when rows disagree with each other. When every row
    # is wide it reads the surplus as an index and shifts every column name one
    # place too far along, which is worse than an error because it looks fine.
    if isinstance(frame.index, pd.RangeIndex):
        return frame

    logger.info("Re-reading %s as ragged: every row outgrew the header", path)
    return _read_ragged(path)


def _read_ragged(path: Path) -> Any:
    """Parse a CSV whose rows have outgrown its header."""

    import pandas as pd

    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        rows = [row for row in csv.reader(handle) if row]
    if not rows:
        return pd.DataFrame()

    header, body = rows[0], rows[1:]
    if not body:
        return pd.DataFrame(columns=header)

    # Rows of one width can sit above rows of another, so each is placed on its
    # own terms: what we recognise read from the left, the step read from the
    # right, and whatever is left over gathered at the end of the frame.
    anchored = int(bool(header) and header[-1] in STEP_COLUMNS)
    known = len(header) - anchored
    placed = [_place(row, known, anchored) for row in body]
    surplus = max(len(row) for _, row in placed)
    names = header + [f"_extra_{i}" for i in range(1, surplus + 1)]
    shaped = [known_part + extras + [None] * (surplus - len(extras)) for known_part, extras in placed]
    return _numeric_where_possible(pd.DataFrame(shaped, columns=names))


def _place(row: list[str], known: int, anchored: int) -> tuple[list[Any], list[Any]]:
    """One row split into the fields the header names and the fields it does not."""

    named: list[Any] = list(row[:known]) + [None] * (known - len(row))
    if not anchored:
        return named, list(row[known:])
    if len(row) <= known:
        return named + [None], []
    return named + [row[-1]], list(row[known:-1])


def _numeric_where_possible(frame: Any) -> Any:
    """Give columns the dtypes ``read_csv`` would have inferred for them.

    A column counts as numeric when every field is a number or one of the
    ways a component writes "no value", so a measurement that failed all run
    stays a column of NaN that monitoring can report on, rather than text it
    has to ignore.
    """

    import pandas as pd

    for column in frame.columns:
        values = frame[column]
        if values.dtype.kind in "fiub":
            continue
        try:
            frame[column] = pd.to_numeric(values.mask(values.isin(MISSING_VALUES)))
        except (TypeError, ValueError):
            continue
    return frame
