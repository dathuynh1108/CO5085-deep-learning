"""Torchvision backbones with explicit trainable boundaries."""
import torch
from torch import nn
from torchvision.models import resnet18, swin_t, ResNet18_Weights, Swin_T_Weights


def configure_freeze(model, family, regime):
    if family not in {'resnet18', 'swin_t'} or regime not in {'scratch', 'head', 'partial', 'full'}:
        raise ValueError('Unknown architecture or regime')
    head = 'fc' if family == 'resnet18' else 'head'
    names = [head]
    if regime == 'partial':
        names += ['layer4'] if family == 'resnet18' else ['features.7', 'norm']
    if regime in {'scratch', 'full'}:
        names = ['']
    model._a1_trainable_roots = names
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(any(root == '' or name == root or name.startswith(root + '.') for root in names))
    training_mode(model)
    return model


def training_mode(model):
    # Starting from eval also freezes BN buffers and stochastic depth in frozen
    # stages. Only entire explicitly trainable subtrees switch back to train.
    model.eval()
    for root in model._a1_trainable_roots:
        (model if root == '' else model.get_submodule(root)).train()


def build_model(experiment, initialize_pretrained=True):
    family, regime = experiment['family'], experiment['regime']
    pretrained = regime != 'scratch' and initialize_pretrained
    if family == 'resnet18':
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 100)
    elif family == 'swin_t':
        model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
        model.head = nn.Linear(model.head.in_features, 100)
    else:
        raise ValueError(f'Unknown family {family}')
    return configure_freeze(model, family, regime)


def frozen_state(model):
    roots = model._a1_trainable_roots
    return {name: value for name, value in model.state_dict().items()
            if not any(root == '' or name == root or name.startswith(root + '.') for root in roots)}


def parameter_counts(model):
    return {'total_params': sum(p.numel() for p in model.parameters()),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)}
