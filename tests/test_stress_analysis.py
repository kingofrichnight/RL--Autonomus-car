import pytest

from scripts.analyze_predictive_stress import paired_success, verify_rows
from scripts.recover_predictive_stress import rows_match


def test_paired_success_exact_counts_and_p():
    a = [dict(seed=i, success=False) for i in range(5)]
    b = [dict(seed=i, success=True) for i in range(5)]
    result = paired_success(a, b)
    assert result["rescues"] == 5 and result["regressions"] == 0
    assert result["exact_two_sided_mcnemar_p"] == 0.0625
    assert paired_success(a, a)["exact_two_sided_mcnemar_p"] == 1


def test_pairing_rejects_seed_mismatch():
    with pytest.raises(ValueError):
        paired_success([dict(seed=1, success=True)], [dict(seed=2, success=True)])


def test_recovery_replay_checks_metrics_and_labels():
    a = {"reward": 1.23, "collision": False, "counts": {"normal": 5}}
    assert rows_match(a, {**a, "reward": 1.23 + 1e-12})
    assert not rows_match(a, {**a, "collision": True})
    assert not rows_match(a, {**a, "counts": {"normal": 4}})


def test_ledger_requires_exclusive_outcomes():
    with pytest.raises(ValueError, match="exclusive"):
        verify_rows([dict(seed=7, success=True, collision=True, incomplete=False)],
                    dict(first_seed=7, episodes_per_scenario=1))


def test_ledger_rejects_duplicate_seeds():
    with pytest.raises(ValueError, match="seed order"):
        verify_rows([dict(seed=7), dict(seed=7)], dict(first_seed=7, episodes_per_scenario=2))
