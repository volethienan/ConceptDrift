from __future__ import annotations

import numpy as np
import pandas as pd

from tools.analyze_revision_cluster_aware import _cluster_statistics, _holm_adjust


def test_cluster_statistics_use_cluster_count_and_are_deterministic() -> None:
    differences = pd.Series([0.1, 0.2, -0.05, 0.0])
    first = _cluster_statistics(differences, "example")
    second = _cluster_statistics(differences, "example")
    assert first == second
    assert first["clusters"] == 4
    assert first["wins"] == 2
    assert first["ties"] == 1
    assert first["losses"] == 1
    assert np.isclose(first["mean_difference"], 0.0625)


def test_holm_adjustment_is_monotone_in_sorted_p_values() -> None:
    frame = pd.DataFrame({"p": [0.04, 0.01, 0.03]})
    _holm_adjust(frame, "p", "holm")
    ordered = frame.sort_values("p")
    assert ordered["holm"].is_monotonic_increasing
    assert (frame["holm"] >= frame["p"]).all()
    assert (frame["holm"] <= 1.0).all()
