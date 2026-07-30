from __future__ import annotations

import numpy as np
import pandas as pd

from tools.analyze_dasp_v16_five_dataset import (
    _cluster_sign_flip_test,
    _holm_adjust,
)


def test_cluster_sign_flip_uses_streams_not_episode_rows() -> None:
    paired = pd.DataFrame(
        {
            "dataset": ["d"] * 8,
            "stream_seed": [1, 1, 2, 2, 3, 3, 4, 4],
            "difference": [0.2, 0.4, 0.1, 0.3, 0.2, 0.2, 0.5, 0.1],
        }
    )
    result = _cluster_sign_flip_test(paired)

    assert result["independent_stream_clusters"] == 4
    assert result["sign_flip_mode"] == "exact"
    assert result["sign_flip_draws"] == 16
    assert np.isclose(result["cluster_sign_flip_p_two_sided"], 0.125)


def test_holm_adjustment_is_monotone_in_sorted_p_values() -> None:
    frame = pd.DataFrame({"p": [0.03, 0.001, 0.02]})
    adjusted = _holm_adjust(frame, "p").sort_values("p")

    assert np.allclose(adjusted["holm_adjusted_p"], [0.003, 0.04, 0.04])
    assert adjusted["holm_reject_0_05"].all()
