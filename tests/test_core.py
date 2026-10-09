"""Algebra/protocol tests only: no dataset download and no training loop."""
import math
import numpy as np
import pytest
import torch
from common.data import stratified_split, channel_statistics
from common.metrics import classification_metrics
from common.runtime import capture_rng, restore_rng
from common.tokenization import image_sequence
from E1.models import SoftmaxClassifier, MLPClassifier, SmallCNN
from E2.models import ManualSelfAttention, TorchSelfAttention, ImageTransformer
from E3.models import RecurrentClassifier, orthogonality_penalty, unroll_recurrent


def test_stratified_split_is_disjoint_reproducible():
    y = np.repeat(np.arange(3), 20)
    tr, va = stratified_split(y, 0.2, 2026)
    tr2, va2 = stratified_split(y, 0.2, 2026)
    np.testing.assert_array_equal(tr, tr2)
    np.testing.assert_array_equal(va, va2)
    assert len(set(tr) & set(va)) == 0
    assert sorted(np.r_[tr, va].tolist()) == list(range(60))
    assert np.bincount(y[va]).tolist() == [4, 4, 4]


def test_split_rejects_small_class():
    with pytest.raises(ValueError):
        stratified_split(np.array([0, 1, 1]), 0.2, 1)


def test_statistics_only_selected_training_indices():
    images = torch.tensor([[[0, 255]], [[255, 255]], [[0, 0]]], dtype=torch.uint8)
    mean, std = channel_statistics(images, np.array([0]))
    assert mean == pytest.approx(0.5)
    assert std == pytest.approx(0.5)


def test_row_sequence_order():
    x = torch.arange(16.0).reshape(1, 1, 4, 4)
    out = image_sequence(x, 'rows', 2)
    assert out.shape == (1, 4, 4)
    assert out[0, 1].tolist() == [4, 5, 6, 7]


def test_patch_sequence_order():
    x = torch.arange(16.0).reshape(1, 1, 4, 4)
    out = image_sequence(x, 'patches', 2)
    assert out.shape == (1, 4, 4)
    assert out[0].tolist() == [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]


def test_patch_rejects_nondivisible_size():
    with pytest.raises(ValueError):
        image_sequence(torch.zeros(1, 1, 7, 7), 'patches', 4)


@pytest.mark.parametrize('model', [SoftmaxClassifier, MLPClassifier, SmallCNN])
def test_e1_logits_and_gradients(model):
    net = model()
    x = torch.randn(2, 1, 28, 28)
    logits = net(x)
    assert logits.shape == (2, 10)
    logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_attention_forward_and_all_gradients_match():
    torch.manual_seed(12)
    a = ManualSelfAttention(12, 3, 0.0).double()
    b = TorchSelfAttention.from_manual(a).double()
    xa = torch.randn(2, 5, 12, dtype=torch.double, requires_grad=True)
    xb = xa.detach().clone().requires_grad_(True)
    ya, yb = a(xa), b(xb)
    torch.testing.assert_close(ya, yb, rtol=1e-9, atol=1e-10)
    upstream = torch.randn_like(ya)
    ya.backward(upstream); yb.backward(upstream)
    torch.testing.assert_close(xa.grad, xb.grad, rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(a.qkv.weight.grad, b.mha.in_proj_weight.grad, rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(a.qkv.bias.grad, b.mha.in_proj_bias.grad, rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(a.proj.weight.grad, b.mha.out_proj.weight.grad, rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(a.proj.bias.grad, b.mha.out_proj.bias.grad, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize('tokenizer,n', [('rows', 28), ('patches', 49), ('cnn', 49)])
def test_full_transformer_paired_initialization(tokenizer, n):
    torch.manual_seed(19)
    a = ImageTransformer(tokenizer=tokenizer, backend='manual', dropout=0.0).double().eval()
    torch.manual_seed(19)
    b = ImageTransformer(tokenizer=tokenizer, backend='torch', dropout=0.0).double().eval()
    x = torch.randn(2, 1, 28, 28, dtype=torch.double)
    assert a.num_tokens == n
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    torch.testing.assert_close(a(x), b(x), rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize('kind', ['lstm', 'gru'])
@pytest.mark.parametrize('pooling', ['last', 'mean'])
def test_unrolled_recurrent_matches_native(kind, pooling):
    torch.manual_seed(9)
    net = RecurrentClassifier(kind=kind, hidden_size=7, pooling=pooling, dropout=0).double().eval()
    seq = torch.randn(2, 5, 28, dtype=torch.double, requires_grad=True)
    native, _ = net.rnn(seq)
    output, hs, cs, gates = unroll_recurrent(net, seq)
    torch.testing.assert_close(output, native, rtol=1e-8, atol=1e-9)
    assert len(hs) == 5
    net.classify(output).square().sum().backward()
    assert all(h.grad is not None for h in hs)
    if kind == 'lstm':
        assert len(cs) == len(gates) == 5
        assert all(c.grad is not None for c in cs)


def test_gatewise_orthogonal_initialization_and_zero_penalty():
    net = RecurrentClassifier(hidden_size=8, orthogonal_init=True)
    w = net.rnn.weight_hh_l0
    for block in w.chunk(4):
        torch.testing.assert_close(block.T @ block, torch.eye(8), atol=1e-6, rtol=1e-6)
    assert orthogonality_penalty(net).item() < 1e-10


def test_penalty_gradient_matches_derivation():
    net = RecurrentClassifier(hidden_size=5).double()
    penalty = orthogonality_penalty(net)
    penalty.backward()
    w = net.rnn.weight_hh_l0
    expected = torch.cat([4 / (4 * 5) * a @ (a.T @ a - torch.eye(5, dtype=a.dtype)) for a in w.chunk(4)])
    torch.testing.assert_close(w.grad, expected, atol=1e-10, rtol=1e-9)


def test_chrono_bias_is_combined_bias_not_double_counted():
    net = RecurrentClassifier(hidden_size=8, chrono_init=True)
    b = net.rnn.bias_ih_l0 + net.rnn.bias_hh_l0
    assert net.sequence_length == 28
    torch.testing.assert_close(b[:8], torch.full((8,), -math.log(27)))
    torch.testing.assert_close(b[8:16], torch.full((8,), math.log(27)))


def test_rng_restore():
    import random
    random.seed(3); np.random.seed(3); torch.manual_seed(3)
    state = capture_rng()
    first = (random.random(), np.random.rand(), torch.rand(3))
    restore_rng(state)
    assert random.random() == first[0]
    assert np.random.rand() == first[1]
    torch.testing.assert_close(torch.rand(3), first[2])


def test_metrics_include_absent_class_without_nan():
    m = classification_metrics(np.array([0, 0, 1]), np.array([0, 1, 1]), 3)
    assert m['accuracy'] == pytest.approx(2/3)
    assert m['confusion_matrix'] == [[1, 1, 0], [0, 1, 0], [0, 0, 0]]
    assert m['macro_f1'] == pytest.approx(4/9)
    assert m['per_class'][2]['f1'] == 0
