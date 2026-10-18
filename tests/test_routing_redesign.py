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


def quarantine_experiment(tmp_path):
    cfg = load_config('configs/modular_dasp_smoke.yaml')
    cfg.output.directory, cfg.output.run_name = str(tmp_path), 'q'
    cfg.dasp.baseline = 'modular_dasp_v2'
    cfg.data.warmup_windows = 4
    cfg.dasp_v2.quarantine = True
    cfg.dasp_v2.quarantine_samples = 256
    return StreamingExperiment(cfg)


def test_quarantine_needs_a_persistent_input_shift_in_one_group(tmp_path):
    torch.manual_seed(3)
    model = additive(1)
    experiment = quarantine_experiment(tmp_path)
    with torch.no_grad():
        for _ in range(4):  # warm-up windows only build the trusted reference
            logits, diagnostics = dasp_v2.quarantine_logits(experiment, model, torch.randn(256, 16))
            assert logits is None
        for _ in range(3):  # clean windows: no false quarantine
            _, diagnostics = dasp_v2.quarantine_logits(experiment, model, torch.randn(256, 16))
            assert diagnostics['quarantined_modules'] == ''
        shifted = torch.randn(256, 16)
        shifted[:, 4:8] = shifted[:, 4:8] * 6 + 10  # feature block of module_2
        _, first = dasp_v2.quarantine_logits(experiment, model, shifted)
        assert first['quarantined_modules'] == ''  # one window is not enough
        logits, second = dasp_v2.quarantine_logits(experiment, model, shifted)
        assert second['quarantined_modules'] == 'module_2'
        _, reps = model(shifted, return_representations=True)
        _, reference = model(experiment._dasp_v2_trust['reference']['module_2'], return_representations=True)
        reps['module_2'] = reference['module_2'].mean(0, keepdim=True).expand_as(reps['module_2'])
        assert torch.allclose(logits, model.logits_from_representations(reps))
    dasp_v2.refresh_trust(experiment, model, shifted, ['module_2'])
    with torch.no_grad():
        _, after = dasp_v2.quarantine_logits(experiment, model, shifted)
    assert after['quarantined_modules'] == ''


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


def blockwise(seed=0, module_dim=8, head_dim=6):
    torch.manual_seed(seed)
    return ModularOnlineMLP(16, 2, [16], n_modules=4, module_dim=module_dim, head_dim=head_dim,
                            head_type='blockwise')


def test_blockwise_head_equals_a_linear_layer_on_concatenated_modules():
    model = blockwise()
    x = torch.randn(9, 16)
    logits, reps = model(x, return_representations=True)
    weight = torch.cat([block.weight for block in model.head_blocks], dim=1)
    fused = torch.cat([reps[name] for name in model.layer_names], dim=1) @ weight.T + model.head_core_bias
    assert torch.allclose(logits, model.classifier(model.head_core(fused)), atol=1e-6)
    groups = model.adaptation_parameter_groups()
    assert {id(p) for p in model.head_blocks[1].parameters()} <= {id(p) for p in groups['module_2']}
    assert set(groups) == {*model.layer_names, 'head_core', 'classifier'}


def test_subspace_repair_changes_only_the_module_its_head_block_and_the_final_layer(tmp_path):
    from dasp_nids.runner import _train_dasp
    cfg = load_config('configs/modular_dasp_smoke.yaml')
    model = blockwise(2)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    plasticities = {'module_1': 0., 'module_2': 1., 'module_3': 0., 'module_4': 0., 'head_core': 0., 'classifier': 1.}
    _train_dasp(model, torch.randn(32, 16), torch.arange(32) % 2, plasticities, cfg, torch.device('cpu'),
                ReservoirMemory(8, 1), strict_freeze=True, use_projection_memory=False, epochs=2)
    changed = {name for name, p in model.named_parameters() if not torch.equal(p, before[name])}
    assert changed == {name for name in before
                       if name.startswith(('modules_list.1.', 'head_blocks.1.', 'classifier.'))}


def test_end_to_end_blockwise_head_reports_head_aware_locality(tmp_path):
    cfg = load_config('configs/modular_dasp_smoke.yaml')
    cfg.output.directory, cfg.output.run_name = str(tmp_path), 'blockwise'
    cfg.training.model_seed = 91
    cfg.dasp.baseline = 'modular_dasp_v2'
    cfg.baselines.update_label_budget = 32
    cfg.model.modular_head_type = 'blockwise'
    cfg.dasp_v2.routing, cfg.dasp_v2.routing_top_k = 'native', 1
    result = StreamingExperiment(cfg).run()
    evaluated = result.windows[~result.windows.warmup.astype(bool)]
    background = evaluated[evaluated.selected_candidate == 'background_classifier']
    assert (background.optimizer_included_parameter_fraction < 0.2).all()
    assert (evaluated.inactive_module_max_abs_change.fillna(0) == 0).all()
