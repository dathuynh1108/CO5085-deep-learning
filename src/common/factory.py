from torch import nn
from common.config import Experiment
from E1.models import SoftmaxClassifier, MLPClassifier, SmallCNN
from E2.models import ImageTransformer
from E3.models import RecurrentClassifier


def build_model(config: Experiment) -> nn.Module:
    if config.exercise == 'E1':
        return {'softmax': SoftmaxClassifier, 'mlp': MLPClassifier, 'cnn': SmallCNN}[config.model]()
    if config.exercise == 'E2':
        return ImageTransformer(tokenizer=config.tokenizer, backend=config.backend,
                                dim=config.dim, heads=config.heads, depth=config.depth,
                                patch_size=config.patch_size)
    return RecurrentClassifier(kind=config.model, tokenizer=config.tokenizer,
                               hidden_size=config.hidden_size, patch_size=config.patch_size,
                               pooling=config.pooling, orthogonal_init=config.orthogonal_init,
                               chrono_init=config.chrono_init)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
