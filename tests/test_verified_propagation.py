import numpy as np
import pytest

from dasp_nids import verified_propagation as vp


def _window(seed=0, n=600, d=6):
    """Benign blob plus two tight attack bursts."""
    rng = np.random.default_rng(seed)
    benign = rng.normal(0, 1, (n - 120, d))
    burst_a = rng.normal(6, 0.05, (80, d))
    burst_b = rng.normal(-6, 0.05, (40, d))
    x = np.concatenate([benign, burst_a, burst_b]).astype(np.float32)
    y = np.r_[np.zeros(n - 120), np.ones(120)].astype(np.int64)
    return x, y


METHODS = {
    "purity": lambda x, c, o, r: vp.purity_verified_propagation(x, c, o, r, labels_per_test=4, epsilon=0.4, delta=0.2),
    "representative": lambda x, c, o, r: vp.representative_propagation(x, c, o, r, clusters=8),
    "uniform_knn": lambda x, c, o, r: vp.uniform_knn_propagation(x, c, o, r, budget=16),
    "hierarchical": lambda x, c, o, r: vp.hierarchical_sampling(x, c, o, r, budget=16),
}


@pytest.mark.parametrize("name", sorted(METHODS))
def test_only_queried_labels_matter_and_budget_holds(name):
    x, y = _window()
    candidates = np.arange(len(x))
    oracle = vp.LabelOracle(y, budget=16)
    first = METHODS[name](x, candidates, oracle, np.random.default_rng(5))
    assert len(oracle.queried) <= 16
    flipped = y.copy()
    hidden = np.setdiff1d(np.arange(len(y)), first.queried)
    flipped[hidden] = 1 - flipped[hidden]
    second = METHODS[name](x, candidates, vp.LabelOracle(flipped, budget=16), np.random.default_rng(5))
    np.testing.assert_array_equal(first.queried, second.queried)
    np.testing.assert_array_equal(first.alarms, second.alarms)


def test_oracle_rejects_over_budget():
    oracle = vp.LabelOracle(np.zeros(10, dtype=np.int64), budget=2)
    oracle.query([0, 1])
    oracle.query([1])  # repeated queries are free
    with pytest.raises(vp.LabelBudgetExceeded):
        oracle.query([2])


def test_purity_propagation_finds_pure_bursts_and_spares_benign():
    x, y = _window()
    result = vp.purity_verified_propagation(x, np.arange(len(x)), vp.LabelOracle(y, 32),
                                            np.random.default_rng(1), labels_per_test=4,
                                            epsilon=0.4, delta=0.2)
    assert y[result.alarms].mean() == 1.0
    assert result.alarms.size >= 80


def test_clopper_pearson_all_successes():
    assert vp.clopper_pearson_lower(5, 5, 0.2) == pytest.approx(0.2 ** (1 / 5))
    assert vp.clopper_pearson_lower(0, 5, 0.2) == 0.0
