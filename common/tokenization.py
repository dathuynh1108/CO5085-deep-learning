"""Image-to-sequence operations; there is no text tokenizer in this project."""
from __future__ import annotations
import torch
from torch import Tensor
from torch.nn import functional as F


def image_sequence(images: Tensor, mode: str, patch_size: int = 4) -> Tensor:
    """Return [batch, steps, features] in row-major spatial order."""
    if images.ndim != 4:
        raise ValueError('Expected images with shape [B, C, H, W].')
    b, c, h, w = images.shape
    if mode == 'rows':
        return images.permute(0, 2, 1, 3).reshape(b, h, c * w)
    if mode == 'patches':
        if patch_size <= 0 or h % patch_size or w % patch_size:
            raise ValueError('The positive patch size must divide both image dimensions.')
        return F.unfold(images, kernel_size=patch_size, stride=patch_size).transpose(1, 2)
    raise ValueError(f'Unknown sequence representation: {mode!r}')
