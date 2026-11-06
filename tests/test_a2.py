"""Checks for A2's controlled comparison and pixel evaluation."""

import numpy as np
from PIL import Image
import torch

from A2.data import letterbox, split_names
from A2.models import PetSegmenter, segmentation_loss
from A2.run import confusion_by_image, scores


def test_split_keeps_official_test_and_samples_each_breed():
    trainval = {"a": 1, "b": 1, "c": 2, "d": 2}
    splits = split_names(trainval, {"e": 2}, seed=2026, validation_per_breed=1)
    assert splits["test"] == ["e"]
    assert set(splits["train"]).isdisjoint(splits["validation"])
    assert sorted(trainval[name] for name in splits["validation"]) == [1, 2]


def test_letterbox_preserves_mask_ids_and_ignores_padding():
    mask = Image.fromarray(np.array([[1, 2, 3, 1], [3, 2, 1, 2]], dtype=np.uint8))
    image, target = letterbox(Image.new("RGB", mask.size), mask, size=4)
    assert image.shape == (4, 4, 3)
    assert target[:2].tolist() == [[0, 1, 2, 0], [2, 1, 0, 1]]
    assert np.all(target[2:] == 255)


def test_gate_is_the_only_variant_switch_for_semantic_gradients():
    fixed, guided = (PetSegmenter(variant).eval() for variant in ("fixed", "guided"))
    guided.load_state_dict(fixed.state_dict())
    image = torch.randn(2, 3, 32, 32)
    fixed_logits, fixed_edge = fixed(image)
    guided_logits, guided_edge = guided(image)
    torch.testing.assert_close(fixed_edge, guided_edge)
    assert fixed_logits.shape == (2, 3, 32, 32)
    fixed_logits.square().mean().backward()
    guided_logits.square().mean().backward()
    assert fixed.boundary[-1].weight.grad is None
    assert guided.boundary[-1].weight.grad.abs().sum() > 0


def test_ignored_pixels_are_excluded_from_both_losses_and_confusion():
    target = torch.tensor([[[0, 1], [2, 255]]])
    logits = torch.zeros(1, 3, 2, 2)
    edge = torch.zeros(1, 2, 2)
    expected = segmentation_loss(logits, edge, target, 0.25)
    logits[0, 0, 1, 1] = 100
    edge[0, 1, 1] = 100
    torch.testing.assert_close(segmentation_loss(logits, edge, target, 0.25), expected)
    matrix = confusion_by_image(np.array([[[0, 1], [1, 2]]]), target.numpy())[0]
    assert matrix.tolist() == [[1, 0, 0], [0, 1, 0], [0, 1, 0]]
    assert scores(matrix)["miou"] == 0.5
