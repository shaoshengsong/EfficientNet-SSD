from ssd.modeling import registry
from .efficient_net import EfficientNet

__all__ = ['efficient_net_b0','EfficientNet']


@registry.BACKBONES.register('efficient_net-b3')
def efficient_net_b3(cfg, pretrained=True):
    if pretrained:
        model = EfficientNet.from_pretrained('efficientnet-b3')
    else:
        model = EfficientNet.from_name('efficientnet-b3')

    return model

@registry.BACKBONES.register('efficient_net-b0')
def efficient_net_b0(cfg, pretrained=True):
    if pretrained:
        model = EfficientNet.from_pretrained('efficientnet-b0')
    else:
        model = EfficientNet.from_name('efficientnet-b0')

    return model