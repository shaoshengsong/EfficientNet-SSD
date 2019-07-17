from ssd.modeling import registry
from .mobilenet import MobileNetV2
from .efficient_net import EfficientNet

__all__ = ['build_backbone', 'MobileNetV2', 'EfficientNet']


def build_backbone(cfg):
    return registry.BACKBONES[cfg.MODEL.BACKBONE.NAME](cfg, cfg.MODEL.BACKBONE.PRETRAINED)
