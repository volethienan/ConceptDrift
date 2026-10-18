from __future__ import annotations

import pytest
import torch

from dasp_nids import dasp_v2
from dasp_nids.config import load_config
from dasp_nids.memory import ReservoirMemory
from dasp_nids.model import ModularOnlineMLP
from dasp_nids.runner import StreamingExperiment


def additive(seed=0):
    torch.manual_seed(seed)
    return ModularOnlineMLP(16, 2, [16], n_modules=4, module_dim=8, head_type='additive', additive_head_dim=4)


def test_additive_head_groups_own_their_heads_and_repairs_stay_output_local():
    model = additive()
    groups = model.adaptation_parameter_groups()
    assert len(groups['classifier']) == 1  # shared bias only
    head_ids = {id(p) for head in model.module_heads for p in head.parameters()}
    assert head_ids <= {id(p) for name in model.layer_names for p in groups[name]}
    x = torch.randn(10, 16)
    logits, reps = model(x, return_representations=True)
    assert torch.allclose(logits, model.logits_from_representations(reps))
    contribution = lambda m, k: m.module_heads[k](m(x, return_representations=True)[1][m.layer_names[k]])
    before = [contribution(model, k).detach() for k in range(4)]
    with torch.no_grad():
        for parameter in groups['module_2']:
            parameter.add_(0.5)
    after = [contribution(model, k).detach() for k in range(4)]
    assert not torch.allclose(before[1], after[1])
    assert all(torch.equal(before[k], after[k]) for k in (0, 2, 3))


def experiment_with_trust(tmp_path, model):
    cfg = load_config('configs/modular_dasp_smoke.yaml')
    cfg.output.directory, cfg.output.run_name = str(tmp_path), 'q'
    cfg.dasp.baseline = 'modular_dasp_v2'
    cfg.dasp_v2.quarantine = True
    cfg.dasp_v2.quarantine_samples = 128
    experiment = StreamingExperiment(cfg)
    memory = ReservoirMemory(512, 1)
    memory.add(torch.randn(512, 16), torch.arange(512) % 2)
    dasp_v2.initialise_trust(experiment, model, memory)
    return experiment


def test_quarantine_isolates_only_the_shifted_feature_group(tmp_path):
    model = additive(1)
    experiment = experiment_with_trust(tmp_path, model)
    clean = torch.randn(256, 16)
    with torch.no_grad():
        logits, diagnostics = dasp_v2.quarantine_logits(experiment, model, clean)
    assert diagnostics['quarantined_modules'] == ''
    assert torch.allclose(logits, model(clean))
    shifted = clean.clone()
    shifted[:, 4:8] = shifted[:, 4:8] * 6 + 10  # feature block of module_2
    with torch.no_grad():
        logits, diagnostics = dasp_v2.quarantine_logits(experiment, model, shifted)
        _, reps = model(shifted, return_representations=True)
        reps['module_2'] = experiment._dasp_v2_trust['neutral']['module_2'].expand_as(reps['module_2'])
        expected = model.logits_from_representations(reps)
    assert diagnostics['quarantined_modules'] == 'module_2'
    assert torch.allclose(logits, expected)
    dasp_v2.refresh_trust(experiment, model, shifted, ['module_2'])
    with torch.no_grad():
        _, diagnostics = dasp_v2.quarantine_logits(experiment, model, shifted)
    assert diagnostics['quarantined_modules'] == ''


@pytest.mark.parametrize('routing', ['quarantine', 'random'])
def test_end_to_end_with_additive_heads_and_quarantine(tmp_path, routing):
    cfg = load_config('configs/modular_dasp_smoke.yaml')
    cfg.output.directory, cfg.output.run_name = str(tmp_path), routing
    cfg.training.model_seed = 91
    cfg.dasp.baseline = 'modular_dasp_v2'
    cfg.baselines.update_label_budget = 32
    cfg.model.modular_head_type = 'additive'
    cfg.dasp_v2.quarantine = True
    cfg.dasp_v2.routing = routing
    result = StreamingExperiment(cfg).run()
    evaluated = result.windows[~result.windows.warmup.astype(bool)]
    assert 'quarantine_z_module_1' in evaluated.columns
    assert set(evaluated.ground_truth_update_labels) == {32}
