import torch.nn as nn

from .pws_layer import PWSConv, pws_init, PWSLinear
from mmcv.cnn import build_norm_layer
from .layers import PLinear

def build_conv_layer(conv_cfg, in_channels, out_channels, kernel_size=1, stride=0, padding=0, norm_cfg=None):
    convtype = conv_cfg.pop('type')
    norm_layer = None
    if convtype == 'pws':
        conv_layer = PWSConv(in_channels, out_channels, 
                kernel_size=kernel_size, stride=stride, padding=padding, bias=True, **conv_cfg)
    else:
        conv_layer = nn.Conv2d(in_channels, out_channels, 
                kernel_size=kernel_size, stride=stride, padding=padding, bias=(norm_cfg is None))
        
        if norm_cfg is not None:
            norm_name, norm_layer = build_norm_layer(norm_cfg, out_channels)
    conv_cfg['type'] = convtype
    return conv_layer, norm_layer

def build_linear_layer(linear_cfg, in_features, out_features, norm=False):
    linear_type = linear_cfg.pop('type')
    norm_layer = None
    if linear_type == 'pws':
        layer = PWSLinear(in_features, out_features, **linear_cfg)
        if 'initalpha' in linear_cfg:
            linear_cfg.pop('initalpha')
        norm = False
    elif linear_type == 'plinear':
        layer = PLinear(in_features, out_features, bias=not norm)
    else:
        layer = nn.Linear(in_features, out_features, bias=not norm)

    if norm:
        norm_layer = nn.BatchNorm1d(out_features)
    linear_cfg['type'] = linear_type
    
    if norm_layer is not None:
        return [layer, norm_layer]
    else:
        return [layer]