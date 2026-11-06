"""A controlled adaptation of the boundary gate in PIDNet's Bag module."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


def conv_block(cin, cout, kernel_size):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size, padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


def fuse(detail, context, gate):
    return gate * detail + (1 - gate) * context


class PetSegmenter(nn.Module):
    def __init__(self, variant, pretrained=False):
        super().__init__()
        if variant not in ("fixed", "guided"):
            raise ValueError(f"Unknown variant: {variant}")
        self.variant = variant
        self.encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        del self.encoder.fc, self.encoder.avgpool
        self.detail = conv_block(64, 64, 3)
        self.context = conv_block(512, 64, 1)
        self.boundary = nn.Sequential(conv_block(64, 32, 3), nn.Conv2d(32, 1, 1))
        self.head = nn.Sequential(conv_block(64, 64, 3), nn.Conv2d(64, 3, 1))

    def forward(self, image):
        backbone = self.encoder
        stem = backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(image))))
        shallow = backbone.layer1(stem)
        deep = backbone.layer4(backbone.layer3(backbone.layer2(shallow)))
        detail = self.detail(shallow)
        context = F.interpolate(
            self.context(deep), size=detail.shape[-2:], mode="bilinear", align_corners=False
        )
        edge = self.boundary(detail)
        gate = edge.sigmoid() if self.variant == "guided" else 0.5
        logits = self.head(fuse(detail, context, gate))
        logits = F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
        edge = F.interpolate(edge, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return logits, edge.squeeze(1)


def segmentation_loss(logits, edge, target, boundary_weight):
    logits, edge = logits.float(), edge.float()
    ce = F.cross_entropy(logits, target, ignore_index=255)
    positive = target == 2
    negative = target < 2
    pixel_bce = F.binary_cross_entropy_with_logits(edge, positive.float(), reduction="none")
    balanced_bce = 0.5 * (
        (pixel_bce * positive).sum() / positive.sum().clamp_min(1)
        + (pixel_bce * negative).sum() / negative.sum().clamp_min(1)
    )
    return ce + boundary_weight * balanced_bce
