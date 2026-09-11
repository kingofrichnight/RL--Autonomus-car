import csv
import math

import pytest

from scripts.audit_geometry_development import COLUMNS, gates, metrics, paired, read_rows


def test_exact_paired_success_direction_and_no_discordance():
    no = [{"success": False}] * 6
    yes = [{"success": True}] * 6
    assert paired(no, yes) == {"rescues": 6, "regressions": 0,
                               "p_exact_two_sided": 0.03125, "favorable": True}
    assert not paired(yes, no)["favorable"]
    assert paired(yes, yes)["p_exact_two_sided"] == 1
    with pytest.raises(ValueError, match="length"):
        paired(no, yes[:-1])


def test_ttc_gate_does_not_relax_for_better_outcome_counts():
    comparisons = {name: {"favorable": True} for name in ("original", "control", "v1")}
    counts = {"success": 394, "collision": 98, "incomplete": 8}
    result = gates(counts, 0.6666153731858254, comparisons)
    assert [key for key, value in result.items() if not value] == ["v1_ttc"]
    assert all(gates(counts, 0.6728939419963959, comparisons).values())


@pytest.mark.parametrize("field,value", [("success", "1"), ("min_ttc", "nan"),
                                       ("min_ttc", "-1"), ("length", "1.5"),
                                       ("unsafe_ttc_events", "2"), ("collision", "True")])
def test_raw_validation_rejects_bad_episode(tmp_path, field, value):
    row = dict(zip(COLUMNS, ["1", "1", "True", "False", "0.2", "1", "0", "0"]))
    row[field] = value
    path = tmp_path / "bad.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError):
        read_rows(path, count=1)


def test_metrics_exclude_nonfinite_ttc_and_count_incomplete():
    row = dict(zip(COLUMNS, [1.0, 1, False, False, 0.2, math.inf, 0, 0]))
    result, counts = metrics([row, dict(row, min_ttc=2.0, success=True)])
    assert result["mean_min_ttc"] == 2.0
    assert result["success_rate"] == 0.5
    assert counts == {"success": 1, "collision": 0, "incomplete": 1,
                      "excluded_nonfinite_ttc": 1}
