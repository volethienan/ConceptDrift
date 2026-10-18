from __future__ import annotations

import numpy as np
import pytest
import torch

from dasp_nids.feature_groups import GROUP_NAMES, NETFLOW_V3, schema_for, semantic_groups
from dasp_nids.model import ModularOnlineMLP
from tools.build_feature_shift_streams import REVERT, START, TRANSFORMS, shifted

NAMES = [name for group in GROUP_NAMES for name in NETFLOW_V3[group]]


def stream(seed=0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(NAMES))  # column order must not matter
    names = np.asarray(NAMES)[order]
    raw = rng.lognormal(size=(36, 32, len(NAMES))) + 64
    mean, scale = raw[:4].reshape(-1, len(NAMES)).mean(0), raw[:4].reshape(-1, len(NAMES)).std(0)
    return {'x': ((raw - mean) / scale).astype(np.float32), 'y': rng.integers(0, 2, (36, 32)),
            'feature_names': names, 'scaler_mean': mean, 'scaler_scale': scale,
            'phases': np.asarray(['p'] * 36)}, raw


def test_semantic_groups_partition_columns_by_name_in_any_order():
    arrays, _ = stream()
    names = arrays['feature_names'].tolist()
    groups = semantic_groups(names)
    assert sorted(sum(groups, [])) == list(range(len(names)))
    assert all(names[i] in NETFLOW_V3[g] for g, members in zip(GROUP_NAMES, groups) for i in members)
    with pytest.raises(ValueError):
        schema_for(names[:-1])


@pytest.mark.parametrize('group', GROUP_NAMES)
@pytest.mark.parametrize('revert', [False, True])
def test_shift_changes_only_the_target_features_in_the_target_windows(group, revert):
    arrays, raw = stream()
    result, detail = shifted(arrays, group, revert)
    names = arrays['feature_names'].tolist()
    recovered = result['x'].astype(np.float64) * arrays['scaler_scale'] + arrays['scaler_mean']
    target = [names.index(f) for f in TRANSFORMS['netflow_v3'][group][2]]
    others = [i for i in range(len(names)) if i not in target]
    stop = REVERT if revert else 36
    assert np.allclose(recovered[:, :, others], raw[:, :, others], rtol=1e-4, atol=1e-3)
    assert np.allclose(recovered[:START][:, :, target], raw[:START][:, :, target], rtol=1e-4, atol=1e-3)
    assert not np.allclose(recovered[START:stop][:, :, target], raw[START:stop][:, :, target], rtol=1e-3)
    if revert:
        assert np.allclose(recovered[REVERT:][:, :, target], raw[REVERT:][:, :, target], rtol=1e-4, atol=1e-3)
    assert np.array_equal(result['y'], arrays['y'])
    assert detail['stop_window'] == stop


def test_semantic_modules_see_only_their_group():
    arrays, _ = stream()
    groups = semantic_groups(arrays['feature_names'].tolist())
    model = ModularOnlineMLP(len(NAMES), 2, [16], n_modules=4, module_dim=8, feature_groups=groups)
    x = torch.randn(5, len(NAMES))
    _, before = model(x, return_representations=True)
    changed = x.clone()
    changed[:, groups[1]] += 5.0  # only the timing group
    _, after = model(changed, return_representations=True)
    unchanged = [name for index, name in enumerate(model.layer_names) if index != 1]
    assert all(torch.equal(before[name], after[name]) for name in unchanged)
    assert not torch.equal(before[model.layer_names[1]], after[model.layer_names[1]])
    with pytest.raises(ValueError):
        ModularOnlineMLP(len(NAMES), 2, [16], feature_groups=[groups[0]])
