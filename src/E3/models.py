"""E3: image sequences, native LSTM/GRU, and controlled recurrent improvements."""
from __future__ import annotations
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from common.tokenization import image_sequence


class RecurrentClassifier(nn.Module):
    def __init__(self, kind: str = 'lstm', tokenizer: str = 'rows', hidden_size: int = 128,
                 patch_size: int = 4, pooling: str = 'last', dropout: float = 0.1,
                 orthogonal_init: bool = False, chrono_init: bool = False) -> None:
        super().__init__()
        if kind not in {'lstm', 'gru'} or pooling not in {'last', 'mean'}:
            raise ValueError('Invalid recurrent model or pooling.')
        if hidden_size < 1 or tokenizer not in {'rows', 'patches'}:
            raise ValueError('Invalid hidden size or image representation.')
        if tokenizer == 'patches' and (patch_size <= 0 or 28 % patch_size):
            raise ValueError('patch_size must divide 28.')
        if chrono_init and kind != 'lstm':
            raise ValueError('The chrono ablation is defined for LSTM only.')
        self.kind, self.tokenizer, self.patch_size = kind, tokenizer, patch_size
        self.pooling, self.hidden_size = pooling, hidden_size
        self.sequence_length = 28 if tokenizer == 'rows' else (28 // patch_size) ** 2
        features = 28 if tokenizer == 'rows' else patch_size ** 2
        cls = nn.LSTM if kind == 'lstm' else nn.GRU
        self.rnn = cls(features, hidden_size, num_layers=1, batch_first=True)
        self.readout = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 10))
        # All runs use zero initial biases except the explicitly named chrono ablation.
        nn.init.zeros_(self.rnn.bias_ih_l0)
        nn.init.zeros_(self.rnn.bias_hh_l0)
        if orthogonal_init:
            gates = 4 if kind == 'lstm' else 3
            with torch.random.fork_rng(devices=[]):
                for block in self.rnn.weight_hh_l0.chunk(gates, dim=0):
                    nn.init.orthogonal_(block)
        if chrono_init:
            # A fixed-timescale variant inspired by chrono initialization, not its
            # original random-timescale recipe. PyTorch's order is i, f, g, o.
            b = math.log(self.sequence_length - 1)
            with torch.no_grad():
                self.rnn.bias_ih_l0[:hidden_size].fill_(-b)
                self.rnn.bias_ih_l0[hidden_size:2 * hidden_size].fill_(b)

    def classify(self, outputs: Tensor) -> Tensor:
        pooled = outputs[:, -1] if self.pooling == 'last' else outputs.mean(dim=1)
        return self.readout(pooled)

    def forward(self, images: Tensor) -> Tensor:
        sequence = image_sequence(images, self.tokenizer, self.patch_size)
        outputs, _ = self.rnn(sequence)
        return self.classify(outputs)


def orthogonality_penalty(model: RecurrentClassifier) -> Tensor:
    """Mean gatewise ||W.T W - I||_F^2 / H (soft penalty, not a constraint)."""
    gates = 4 if model.kind == 'lstm' else 3
    h = model.hidden_size
    w = model.rnn.weight_hh_l0
    identity = torch.eye(h, device=w.device, dtype=w.dtype)
    return sum((a.T @ a - identity).square().sum() for a in w.chunk(gates)) / (gates * h)


def unroll_recurrent(model: RecurrentClassifier, sequence: Tensor):
    """Diagnostic unroll sharing the native module's actual parameters.

    Keeping the real intermediate states in this graph is important: retaining
    the gradient of the output tensor of a fused RNN is NOT a BPTT-state probe.
    This helper intentionally supports the single-layer/unidirectional models
    used here. It does not perform an optimizer step.
    """
    rnn, hdim = model.rnn, model.hidden_size
    h = sequence.new_zeros(sequence.shape[0], hdim)
    c = torch.zeros_like(h)
    hs, cs, gates = [], [], []
    for x in sequence.unbind(dim=1):
        a = F.linear(x, rnn.weight_ih_l0, rnn.bias_ih_l0)
        b = F.linear(h, rnn.weight_hh_l0, rnn.bias_hh_l0)
        if model.kind == 'lstm':
            i, f, g, o = (a + b).chunk(4, dim=-1)
            i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
            c = f * c + i * g
            h = o * c.tanh()
            if c.requires_grad:
                c.retain_grad()
            cs.append(c)
            gates.append(f)
        else:
            ar, az, an = a.chunk(3, dim=-1)
            br, bz, bn = b.chunk(3, dim=-1)
            r, z = (ar + br).sigmoid(), (az + bz).sigmoid()
            # This is the PyTorch reset-after formulation, including hidden bias.
            n = (an + r * bn).tanh()
            h = (1 - z) * n + z * h
            gates.append(z)
        if h.requires_grad:
            h.retain_grad()
        hs.append(h)
    return torch.stack(hs, dim=1), hs, cs, gates
