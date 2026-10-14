"""A1 protocol tests: no data download, no pretrained weight download."""
import json
import random
import pytest


def tensor_dependencies():
    torch = pytest.importorskip('torch')
    pytest.importorskip('torchvision')
    torch.set_num_threads(1)
    return torch


def test_cli_lists_eight_and_blocks_unconfirmed_test(capsys):
    from A1.run_suite import main
    main(['--list'])
    assert '24 runs' in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        main(['--phase', 'test'])
    assert error.value.code == 2


def test_protocol_budget():
    from A1.protocol import load_protocol
    p = load_protocol()
    assert len(p['experiments']) == 8 and p['seeds'] == [42, 43, 44]
    assert (p['epochs'], p['batch_size'], p['image_size']) == (20, 128, 96)
    assert not p['tf32'] and not p['compile']
    for e in p['experiments']:
        assert e['lr'] == (1e-3 if e['regime'] in {'scratch', 'head'} else 1e-4)


def test_export_missing_is_explicit(tmp_path):
    from A1.export_report import export
    manifest = export(tmp_path / 'runs', tmp_path / 'generated')
    assert manifest['expected_runs'] == 24
    assert manifest['trained_runs'] == manifest['tested_runs'] == 0
    assert len(manifest['coverage']) == 24
    summary = json.loads((tmp_path / 'generated/summary.json').read_text())
    assert all(g['test_accuracy']['mean'] is None for g in summary['groups'])
    assert '--' in (tmp_path / 'generated/comparison_table.tex').read_text()


def test_split_all_classes_no_overlap():
    tensor_dependencies()
    import numpy as np
    from A1.data import stratified_split
    labels = np.repeat(np.arange(100), 500)
    train, val = stratified_split(labels)
    assert len(train) == 45000 and len(val) == 5000
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(range(50000))
    assert np.all(np.bincount(labels[train]) == 450)
    assert np.all(np.bincount(labels[val]) == 50)
    assert (train, val) == stratified_split(labels)
    assert val != stratified_split(labels, seed=2027)[1]
    with pytest.raises(ValueError):
        stratified_split(labels[:-1])


def test_accuracy_macro_f1():
    tensor_dependencies()
    from A1.metrics import classification_metrics
    result = classification_metrics([0, 0, 1, 2], [0, 1, 1, 1], classes=3)
    assert result['accuracy'] == 0.5
    assert result['macro_f1'] == pytest.approx((2/3 + 0.5 + 0) / 3)
    assert result['confusion'] == [[1, 1, 0], [0, 1, 0], [0, 1, 0]]
    assert classification_metrics([0], [0])['macro_f1'] == 0.01
    with pytest.raises(ValueError):
        classification_metrics([0], [100])


@pytest.mark.parametrize('family', ['resnet18', 'swin_t'])
@pytest.mark.parametrize('regime', ['scratch', 'head', 'partial', 'full'])
def test_freeze_parameters_buffers_and_stochastic_depth(family, regime):
    torch = tensor_dependencies()
    from A1.models import build_model, training_mode, frozen_state, parameter_counts
    from common.runtime import state_fingerprint
    from torchvision.ops import StochasticDepth
    model = build_model({'family': family, 'regime': regime}, initialize_pretrained=False)
    before = state_fingerprint(frozen_state(model))
    training_mode(model)
    head = 'fc' if family == 'resnet18' else 'head'
    roots = [head] + (['layer4'] if family == 'resnet18' else ['features.7', 'norm']) if regime == 'partial' else [head]
    for name, parameter in model.named_parameters():
        expected = regime in {'scratch', 'full'} or any(name.startswith(root + '.') for root in roots)
        assert parameter.requires_grad == expected
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.BatchNorm2d, StochasticDepth)):
            expected = regime in {'scratch', 'full'} or any(name == root or name.startswith(root + '.') for root in roots)
            assert module.training == expected
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    output = model(torch.randn(2, 3, 96, 96))
    torch.nn.functional.cross_entropy(output, torch.tensor([1, 2])).backward()
    optimizer.step()
    assert output.shape == (2, 100)
    assert state_fingerprint(frozen_state(model)) == before
    counts = parameter_counts(model)
    assert counts['trainable_params'] <= counts['total_params']
    if regime in {'scratch', 'full'}:
        assert counts['trainable_params'] == counts['total_params']
    else:
        assert counts['trainable_params'] < counts['total_params']


class RandomDataset:
    """Picklable fixture for worker processes; mirrors three augmentation RNGs."""
    def __len__(self):
        return 16

    def __getitem__(self, index):
        import torch
        import numpy as np
        return torch.tensor([index, torch.rand(()).item(), random.random(), np.random.random()], dtype=torch.float64), index


def test_augmentation_independent_of_workers_and_resume_epoch():
    torch = tensor_dependencies()
    from A1.data import EpochSubset, loader
    from common.runtime import capture_rng, restore_rng, seed_everything
    seed_everything(42)
    subset = EpochSubset(RandomDataset(), list(range(16)))
    subset.set_epoch(3, 42)
    rng = capture_rng()
    sample = subset[5][0]
    assert torch.equal(torch.get_rng_state(), rng['torch'])
    assert random.getstate() == rng['python']
    assert torch.equal(sample, subset[5][0])
    def get(workers, generator):
        return torch.cat([x for x, _ in loader(subset, 4, workers, generator, True)])
    generator = torch.Generator().manual_seed(123)
    generator_state = generator.get_state()
    baseline = get(0, generator)
    generator.set_state(generator_state)
    assert torch.equal(baseline, get(2, generator))
    # Epoch changes augmentation, restored seed+epoch reproduces all samples.
    subset.set_epoch(4, 42)
    assert not torch.equal(sample, subset[5][0])
    subset.set_epoch(3, 42)
    assert torch.equal(sample, subset[5][0])
    restore_rng(rng)


def test_checkpoint_resume_matches_uninterrupted_and_guards_signature(tmp_path):
    torch = tensor_dependencies()
    import numpy as np
    from A1.engine import save_last, restore_last
    from common.runtime import seed_everything, state_fingerprint
    def setup(device='cpu'):
        model = torch.nn.Sequential(torch.nn.Linear(3, 6), torch.nn.Dropout(0.2), torch.nn.Linear(6, 2)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 3)
        generators = {'train': torch.Generator().manual_seed(42), 'val': torch.Generator().manual_seed(43)}
        return model, optimizer, scheduler, generators
    def step(model, optimizer, scheduler, generators):
        model.train()
        inputs = torch.randn(4, 3, generator=generators['train']).to(next(model.parameters()).device)
        labels = torch.tensor([0, 1, 0, 1], device=inputs.device)
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(model(inputs), labels).backward()
        optimizer.step()
        scheduler.step()
        return (random.random(), np.random.random())
    seed_everything(42)
    args = setup()
    step(*args)
    signature = {'config': 'fixed', 'seed': 42}
    save_last(tmp_path / 'last.pt', signature, *args, history=[{'epoch': 1}], best=None,
              initial='initial', frozen_hash='frozen')
    expected_rng = step(*args)
    expected_hash = state_fingerprint(args[0].state_dict())
    expected_lr = args[1].param_groups[0]['lr']
    resumed = setup()
    state = restore_last(tmp_path / 'last.pt', signature, *resumed)
    assert state['history'] == [{'epoch': 1}]
    assert step(*resumed) == expected_rng
    assert state_fingerprint(resumed[0].state_dict()) == expected_hash
    assert resumed[1].param_groups[0]['lr'] == expected_lr
    with pytest.raises(RuntimeError, match='signature mismatch'):
        restore_last(tmp_path / 'last.pt', {'config': 'changed'}, *setup())
    if torch.cuda.is_available():
        cuda_args = setup('cuda')
        restore_last(tmp_path / 'last.pt', signature, *cuda_args)
        for parameter, values in cuda_args[1].state.items():
            assert values['exp_avg'].device == parameter.device
            assert values['exp_avg_sq'].device == parameter.device


def test_best_recovery_and_selection(tmp_path):
    torch = tensor_dependencies()
    from A1.engine import reconcile_best, better
    from common.runtime import state_fingerprint
    model = torch.nn.Linear(2, 2)
    best = {'epoch': 1, 'accuracy': 0.5, 'ce': 1.0, 'state_sha256': state_fingerprint(model.state_dict())}
    state = {'best': best, 'history': [{'epoch': 1}]}
    reconcile_best(tmp_path, {'seed': 42}, model, state)
    checkpoint = torch.load(tmp_path / 'best.pt', weights_only=False)
    assert checkpoint['selection'] == best
    assert better(0.5, 0.9, best)
    assert not better(0.5, 1.0, best)
    assert not better(0.4, 0.1, best)
    (tmp_path / 'best.pt').unlink()
    state['history'].append({'epoch': 2})
    with pytest.raises(RuntimeError, match='cannot be recovered'):
        reconcile_best(tmp_path, {'seed': 42}, model, state)
