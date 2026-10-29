# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reading results CSVs that grew extra columns part way through a run."""

from __future__ import annotations

import pytest

from reinvent4_mcp.jobs.results_csv import (
    read_results_csv,
    step_column,
    total_score_column,
)
from reinvent4_mcp.tools.results import get_top_candidates, read_results

HEADER = "Agent,Prior,Target,Score,SMILES,SMILES_state,Solubility,Solubility (raw),step"

#: What REINVENT writes while a scoring service is answering normally.
NARROW = "6.68,32.35,17.93,0.39,CCO,1,0.74,3.22,{step}"

#: The same run once a component starts reporting extra fields: four more
#: values per row, and the header never catches up.
WIDE = "4.79,29.04,-29.04,0.88,CCN,1,0.91,2.93,True,0.19,True,0.13,{step}"


def write_csv(path, rows):
    path.write_text(HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


class TestReadingARaggedFile:
    def test_rows_wider_than_the_header_are_still_read(self, tmp_path):
        csv_path = write_csv(
            tmp_path / "results_1.csv",
            [NARROW.format(step=0), NARROW.format(step=1), WIDE.format(step=2)],
        )

        df = read_results_csv(csv_path)

        assert len(df) == 3

    def test_every_row_keeps_its_step_whatever_its_width(self, tmp_path):
        csv_path = write_csv(
            tmp_path / "results_1.csv",
            [NARROW.format(step=7), WIDE.format(step=8), NARROW.format(step=9)],
        )

        df = read_results_csv(csv_path)

        assert step_column(df) == "step"
        assert df["step"].tolist() == [7, 8, 9]

    def test_the_columns_named_in_the_header_keep_their_values(self, tmp_path):
        csv_path = write_csv(tmp_path / "results_1.csv", [WIDE.format(step=2)])

        df = read_results_csv(csv_path)

        assert df["SMILES"].tolist() == ["CCN"]
        assert df["Score"].tolist() == [0.88]
        assert df["Solubility (raw)"].tolist() == [2.93]

    def test_the_surplus_fields_are_parked_out_of_the_way(self, tmp_path):
        csv_path = write_csv(tmp_path / "results_1.csv", [WIDE.format(step=2)])

        df = read_results_csv(csv_path)

        assert [c for c in df.columns if c.startswith("_extra_")] == [
            "_extra_1",
            "_extra_2",
            "_extra_3",
            "_extra_4",
        ]

    def test_numbers_are_numbers(self, tmp_path):
        csv_path = write_csv(
            tmp_path / "results_1.csv", [NARROW.format(step=0), WIDE.format(step=1)]
        )

        df = read_results_csv(csv_path)

        assert df["Score"].mean() == pytest.approx(0.635)

    def test_a_component_that_wrote_no_numbers_is_still_a_number_column(self, tmp_path):
        """A dead service leaves 'nan' in every row; that is a failing
        measurement, which monitoring can only report if it reads as one."""

        csv_path = write_csv(
            tmp_path / "results_1.csv",
            [
                NARROW.format(step=0).replace(",0.74,3.22,", ",nan,nan,"),
                WIDE.format(step=1).replace(",0.91,2.93,", ",nan,nan,"),
            ],
        )

        df = read_results_csv(csv_path)

        assert df["Solubility"].isna().all()
        assert df["Solubility"].dtype.kind == "f"

    def test_a_component_that_came_back_keeps_the_values_it_did_write(self, tmp_path):
        csv_path = write_csv(
            tmp_path / "results_1.csv",
            [NARROW.format(step=0).replace(",0.74,3.22,", ",nan,nan,"), WIDE.format(step=1)],
        )

        df = read_results_csv(csv_path)

        assert df["Solubility"].tolist()[1] == pytest.approx(0.91)
        assert df["Solubility"].isna().sum() == 1

    def test_a_file_with_no_rows_yet_is_empty_not_an_error(self, tmp_path):
        csv_path = tmp_path / "results_1.csv"
        csv_path.write_text("", encoding="utf-8")

        assert read_results_csv(csv_path).empty


class TestFindingTheScoreColumn:
    def test_the_rl_stage_writes_score(self, tmp_path):
        df = read_results_csv(write_csv(tmp_path / "r.csv", [NARROW.format(step=0)]))

        assert total_score_column(df) == "Score"

    def test_the_scoring_api_writes_total_score(self, tmp_path):
        path = tmp_path / "r.csv"
        path.write_text("SMILES,total_score\nCCO,0.5\n", encoding="utf-8")

        assert total_score_column(read_results_csv(path)) == "total_score"

    def test_a_file_without_one_says_so(self, tmp_path):
        path = tmp_path / "r.csv"
        path.write_text("SMILES\nCCO\n", encoding="utf-8")

        assert total_score_column(read_results_csv(path)) is None


class TestTheToolsSurviveIt:
    def test_read_results_reports_the_molecules(self, tmp_path):
        csv_path = write_csv(
            tmp_path / "results_1.csv",
            [NARROW.format(step=0), WIDE.format(step=1), WIDE.format(step=2)],
        )

        result = read_results(str(csv_path))

        assert "error" not in result
        assert result["n_rows"] == 3
        assert result["total_score_stats"]["max"] == pytest.approx(0.88)

    def test_likelihoods_are_not_reported_as_scoring_components(self, tmp_path):
        csv_path = write_csv(tmp_path / "results_1.csv", [NARROW.format(step=0)])

        result = read_results(str(csv_path))

        assert set(result["component_summary"]) == {"Solubility"}

    def test_nor_are_the_fields_nothing_named(self, tmp_path):
        csv_path = write_csv(
            tmp_path / "results_1.csv", [NARROW.format(step=0), WIDE.format(step=1)]
        )

        result = read_results(str(csv_path))

        assert set(result["component_summary"]) == {"Solubility"}

    def test_candidates_come_back_ranked(self, tmp_path):
        csv_path = write_csv(
            tmp_path / "results_1.csv", [NARROW.format(step=0), WIDE.format(step=1)]
        )

        result = get_top_candidates(csv_path=str(csv_path), top_n=5)

        assert [c["smiles"] for c in result["candidates"]] == ["CCN", "CCO"]
        assert result["candidates"][0]["total_score"] == pytest.approx(0.88)
