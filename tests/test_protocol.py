"""Protocol and artifact checks, without running a training batch."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset
from common.config import load_experiments, get_experiment
from common.engine import better_checkpoint, run_lock
from common.data import make_loader
from common.runtime import atomic_json, atomic_torch_save, json_hash
from common.evaluation import require_test_permission
from scripts.export_report import export_report, collect_runs


def test_test_requires_explicit_permission():
    with pytest.raises(ValueError):
        require_test_permission('test', False)
    require_test_permission('test', True)
    require_test_permission('validation', False)


def test_checkpoint_selection_uses_accuracy_then_ce_only():
    best = {'accuracy': .9, 'loss': .4}
    assert better_checkpoint(.91, .5, best)
    assert better_checkpoint(.9, .3, best)
    assert not better_checkpoint(.89, .1, best)
    assert not better_checkpoint(.9, .4, best)


def test_lock_is_exclusive_and_released_after_failure(tmp_path):
    with pytest.raises(ValueError):
        with run_lock(tmp_path):
            with pytest.raises(RuntimeError):
                with run_lock(tmp_path):
                    pass
            raise ValueError('intentional test failure')
    assert not (tmp_path / '.train.lock').exists()


def test_fresh_loader_resume_preserves_shuffle_order():
    dataset = TensorDataset(torch.arange(12))
    g = torch.Generator().manual_seed(4)
    loader = make_loader(dataset, 4, 0, torch.device('cpu'), True, g)
    list(loader)
    saved = g.get_state()
    expected = torch.cat([x[0] for x in loader])
    resumed = torch.Generator(); resumed.set_state(saved)
    new_loader = make_loader(dataset, 4, 0, torch.device('cpu'), True, resumed)
    actual = torch.cat([x[0] for x in new_loader])
    torch.testing.assert_close(actual, expected)


def test_atomic_json_rejects_nan_without_corrupting_existing_file(tmp_path):
    path = tmp_path / 'result.json'
    atomic_json(path, {'value': 1})
    with pytest.raises(ValueError):
        atomic_json(path, {'value': float('nan')})
    assert json.loads(path.read_text()) == {'value': 1}


def test_all_required_comparisons_exist():
    experiments = load_experiments()
    assert len(experiments) == 20
    assert {x.model for x in experiments if x.exercise == 'E1'} == {'softmax', 'mlp', 'cnn'}
    assert {(x.tokenizer,x.backend) for x in experiments if x.exercise == 'E2'} == {
        (t,b) for t in ['rows','patches','cnn'] for b in ['manual','torch']}
    assert {(x.model,x.tokenizer) for x in experiments if x.exercise == 'E3' and x.group == 'required'} == {
        (m,t) for m in ['lstm','gru'] for t in ['rows','patches']}
    assert len({x.split_seed for x in experiments}) == 1


def test_unknown_config_and_bad_override_fail():
    with pytest.raises(ValueError):
        get_experiment('does_not_exist')
    with pytest.raises(ValueError):
        get_experiment('e1_cnn', epochs=0)


def test_empty_export_has_no_fabricated_measurements(tmp_path):
    target = tmp_path / 'report'
    rows = export_report(tmp_path / 'no-runs', target)
    assert len(rows) == 20
    assert all(r['test_accuracy_mean'] is None for r in rows)
    assert all(r['n_train_runs'] == r['n_test_runs'] == 0 for r in rows)
    assert all(r['parameters'] > 0 for r in rows)
    assert (target / 'generated/e1_table.tex').exists()
    assert 'Chưa huấn luyện' in (target / 'generated/status.tex').read_text()


def test_export_rejects_changed_split(tmp_path):
    experiment = get_experiment('e1_cnn').to_dict()
    for seed in [42,43]:
        directory = tmp_path / 'e1_cnn' / f'seed_{seed}'
        directory.mkdir(parents=True)
        atomic_json(directory / 'metrics.json', dict(status='trained', seed=seed,
            experiment=experiment, signature=str(seed), data_fingerprint=str(seed)))
    with pytest.raises(ValueError, match='split'):
        collect_runs(tmp_path)


def test_fashion_download_prefers_official_https_without_downloading():
    from common.data import fashion_mnist_type
    dataset_type = fashion_mnist_type()
    assert dataset_type.__name__ == 'FashionMNIST'
    assert dataset_type.mirrors[0].startswith('https://raw.githubusercontent.com/zalandoresearch/')


def test_normalization_metadata_tampering_is_detected():
    from common.data import validate_metadata
    definition = {'dataset':'FashionMNIST', 'version':2, 'split_seed':2026}
    metadata = dict(definition, mean=.2, std=.3)
    metadata['fingerprint'] = json_hash(metadata)
    validate_metadata(metadata, definition)
    metadata['std'] = .8
    with pytest.raises(ValueError, match='normalization'):
        validate_metadata(metadata, definition)


def test_checkpoint_safe_load_and_tensor_fingerprint(tmp_path):
    """Serialization contract only: tensors are untrained test fixtures."""
    from common.factory import build_model
    from common.runtime import state_fingerprint
    from common.evaluation import load_frozen_run
    config = get_experiment('e1_softmax')
    model = build_model(config)
    state = {k:v.detach().clone() for k,v in model.state_dict().items()}
    checksum = state_fingerprint(state)
    atomic_torch_save(tmp_path / 'best.pt', dict(model=state, signature='fixture',
                                               experiment=config.to_dict(), seed=42))
    atomic_json(tmp_path / 'metrics.json', dict(status='trained', signature='fixture',
                                              best_model_sha256=checksum))
    atomic_json(tmp_path / 'run.json', dict(deterministic=False))
    torch.backends.cudnn.allow_tf32 = True
    loaded, _, _, _, loaded_hash = load_frozen_run(tmp_path, torch.device('cpu'))
    assert torch.backends.cudnn.allow_tf32 is False
    assert loaded_hash == checksum
    for key, value in state.items():
        torch.testing.assert_close(loaded.state_dict()[key], value)
    state['classifier.1.weight'][0, 0] += 1
    atomic_torch_save(tmp_path / 'best.pt', dict(model=state, signature='fixture',
                                               experiment=config.to_dict(), seed=42))
    with pytest.raises(ValueError, match='changed'):
        load_frozen_run(tmp_path, torch.device('cpu'))


def test_export_one_seed_has_no_fabricated_standard_deviation(tmp_path):
    """Fixed JSON fixtures exercise aggregation only; never training data."""
    from common.factory import build_model, parameter_count
    config = get_experiment('e1_softmax')
    path = tmp_path / 'runs/e1_softmax/seed_42'; path.mkdir(parents=True)
    atomic_json(path / 'metrics.json', dict(status='trained', signature='fixture',
        experiment=config.to_dict(), seed=42, data_fingerprint='fixture-split',
        completed_epochs=30, n_parameters=parameter_count(build_model(config)),
        best_validation_accuracy=.6, best_validation_ce=1.2, median_train_seconds=3.,
        environment={'torch':'fixture'}, best_model_sha256='fixture-model'))
    atomic_json(path / 'run.json', dict(signature='fixture', source_fingerprint='fixture-code',
                                       dtype='float32', deterministic=False))
    atomic_json(path / 'evaluation_test.json', dict(signature='fixture', model_sha256='fixture-model',
        split='test', seed=42, accuracy=.5, macro_f1=.4, inference_images_per_second=10.,
        dataset_sha256='fixture-test'))
    rows = export_report(tmp_path / 'runs', tmp_path / 'report')
    row = next(r for r in rows if r['id'] == config.id)
    assert row['n_test_runs'] == 1 and row['test_accuracy_mean'] == .5
    assert row['test_accuracy_std'] is None
    assert all(r['n_test_runs'] == 0 for r in rows if r['id'] != config.id)
