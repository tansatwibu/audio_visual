import torch
import torchvision
import torch.nn as nn
from .networks import Resnet18, AudioVisual5layerUNet, AudioVisual7layerUNet, weights_init


def build_resnet18_pretrained(input_channel=3, pool_type='maxpool', with_fc=False,
                              fc_in=512, fc_out=256):
    """Shared resnet18 backbone.

    torchvision renamed ``pretrained`` to ``weights`` and the old keyword is
    gone in recent releases, so try the new one and fall back for older pins.
    """
    try:
        return torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    except (TypeError, AttributeError):
        return torchvision.models.resnet18(pretrained=True)


class ModelBuilder():
    # builder for visual stream
    def build_visual(self, pool_type='avgpool', input_channel=3, fc_out=256, weights=''):
        original_resnet = build_resnet18_pretrained()
        if pool_type == 'conv1x1': #if use conv1x1, use conv1x1 + fc to reduce dimension to 512 feature vector
            net = Resnet18(original_resnet, pool_type=pool_type, input_channel=3, with_fc=True, fc_in=6272, fc_out=fc_out)
        else:
            net = Resnet18(original_resnet, pool_type=pool_type)

        if len(weights) > 0:
            print('Loading weights for visual stream')
            net.load_state_dict(torch.load(weights,map_location='cpu'))
        return net

    #builder for audio stream
    def build_unet(self, unet_num_layers=7, ngf=64, input_nc=1, output_nc=1, weights=''):
        if unet_num_layers == 7:
            net = AudioVisual7layerUNet(ngf, input_nc, output_nc)
        elif unet_num_layers == 5:
            net = AudioVisual5layerUNet(ngf, input_nc, output_nc)
        else:
            raise ValueError('--unet_num_layers must be 5 or 7, got %r' % unet_num_layers)

        net.apply(weights_init)

        if len(weights) > 0:
            print('Loading weights for UNet')
            net.load_state_dict(torch.load(weights,map_location='cpu'))
        return net