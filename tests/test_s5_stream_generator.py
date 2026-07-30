from __future__ import annotations

import numpy as np

from tools.build_controlled_nf_streams import (
    DATASETS,
    DatasetSpec,
    S5ProbeBundle,
    _allocate,
    _build_stream,
    _select_matched_random_by_time_quantile,
    _select_top_by_time_quantile,
    _window_recipe,
)


class _FixedEvasionProbe:
    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        benign = 1.0 / (1.0 + np.exp(values[:, 0]))
        return np.column_stack([benign, 1.0 - benign])

    def predict(self, values: np.ndarray) -> np.ndarray:
        return (values[:, 0] > 1.0).astype(np.int64)


def _pool(
    x: np.ndarray,
    start_id: int,
) -> dict[str, np.ndarray]:
    count = len(x)
    return {
        "x": x.astype(np.float32),
        "source_ids": np.arange(
            start_id, start_id + count, dtype=np.int64
        ),
        "timestamps": np.arange(count, dtype=np.int64),
    }


def test_s5_recipe_is_abrupt_a_to_b_evasive() -> None:
    spec = DATASETS["nf-unsw-nb15-v3"]
    pre, pre_phase = _window_recipe(spec, "S5", 11, 100, 0.05)
    post, post_phase = _window_recipe(spec, "S5", 12, 100, 0.05)

    assert pre == _allocate(100, spec.concept_a)
    assert post == _allocate(100, spec.concept_b)
    assert pre_phase == "A"
    assert post_phase == "B_evasive"


def test_s5_selection_preserves_time_coverage_and_takes_hard_tail() -> None:
    values = np.arange(20, dtype=np.float32).reshape(-1, 1)
    pool = _pool(values, 100)
    scores = np.arange(20, dtype=float)
    selected = _select_top_by_time_quantile(
        pool,
        scores,
        count=4,
        n_quantiles=2,
        rng=np.random.default_rng(7),
    )

    assert set(selected["source_ids"].tolist()) == {108, 109, 118, 119}
    assert len(np.unique(selected["source_ids"])) == 4


def test_s5_matched_random_control_is_disjoint_and_time_matched() -> None:
    values = np.arange(40, dtype=np.float32).reshape(-1, 1)
    pool = _pool(values, 100)
    selected = _select_top_by_time_quantile(
        pool,
        np.arange(40, dtype=float),
        count=8,
        n_quantiles=4,
        rng=np.random.default_rng(7),
    )
    control = _select_matched_random_by_time_quantile(
        pool,
        count=8,
        n_quantiles=4,
        excluded_source_ids=selected["source_ids"],
        rng=np.random.default_rng(8),
    )

    assert not set(selected["source_ids"]).intersection(
        control["source_ids"]
    )
    assert len(np.unique(control["source_ids"])) == 8
    assert [
        int(np.sum((control["source_ids"] - 100) // 10 == index))
        for index in range(4)
    ] == [2, 2, 2, 2]


def test_s5_stream_marks_only_post_drift_attacks_as_evasive() -> None:
    spec = DatasetSpec(
        name="fixture",
        path="unused.csv",
        total_rows=2,
        attack_rows=1,
        family_counts={"benign": 10_000, "a": 10_000, "b": 10_000},
        concept_a={"a": 1.0},
        concept_b={"b": 1.0},
        concept_c={"b": 1.0},
        stable={"a": 1.0},
        local_a={"a": 1.0},
        local_b={"b": 1.0},
    )
    window_size = 32
    benign_per_window = 16
    attack_per_window = 16
    benign_x = np.zeros((benign_per_window * 36, 2), dtype=np.float32)
    a_x = np.full((attack_per_window * 12, 2), 2.0, dtype=np.float32)
    regular_b_x = np.full(
        (attack_per_window * 24, 2), 1.5, dtype=np.float32
    )
    candidate_first = np.tile(
        np.linspace(-2.0, 4.0, 300, dtype=np.float32), 4
    )
    candidate_b_x = np.column_stack(
        [candidate_first, np.zeros_like(candidate_first)]
    )
    pools = {
        ("benign", 0): _pool(benign_x, 0),
        ("a", 0): _pool(a_x, 10_000),
        ("b", 0): _pool(regular_b_x, 20_000),
    }
    candidate_pools = {
        ("b", 0): _pool(candidate_b_x, 30_000),
    }
    probes = [_FixedEvasionProbe() for _ in range(3)]
    arrays, audit = _build_stream(
        spec,
        ["f0", "f1"],
        pools,
        candidate_pools,
        S5ProbeBundle(probes, probes),
        slot=0,
        stream_seed=2101,
        scenario="S5",
        window_size=window_size,
        n_windows=36,
        warmup_windows=4,
        local_fraction=0.05,
        s5_time_quantiles=4,
        s5_min_scope=0.70,
        s5_min_recall_drop=0.30,
        s5_min_macro_f1_drop=0.20,
        s5_min_scope_uplift=0.10,
        s5_min_recall_drop_uplift=0.10,
        s5_min_macro_f1_drop_uplift=0.05,
        s5_gate_policy="enforce",
    )

    scores = arrays["s5_evasiveness_score"]
    expected = np.zeros_like(arrays["y"], dtype=bool)
    expected[12:] = arrays["y"][12:] == 1
    assert np.array_equal(np.isfinite(scores), expected)
    assert np.all(arrays["phases"][12:] == "B_evasive")
    assert audit is not None
    assert audit["passed"]
    assert audit["consensus_evasion_rate"] == 1.0
    assert audit["evasion_uplift_vs_matched_random"] >= 0.10
