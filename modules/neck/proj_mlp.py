import torch
import torch.nn as nn
import torchvision
import math
from ..utils import PWSLinear, build_linear_layer

from mmcv.cnn import (normal_init, constant_init, kaiming_init)

class Proj_MLP(nn.Module):
    def __init__(self, hidden_layer=3, in_channels=2048, hidden_channels=2048, out_channels=2048, linear_cfg=dict(type="linear")):
        super().__init__()
        layer_list = []
        if linear_cfg['type'] == 'pws':
            linear_cfg['initalpha'] = True
            
        for i in range(hidden_layer - 1):
            layer_list.extend(build_linear_layer(linear_cfg, in_channels, hidden_channels, norm=True))
            layer_list.append(nn.ReLU(inplace=True))
            in_channels = hidden_channels

        layer_list.extend(build_linear_layer(linear_cfg, hidden_channels, out_channels, norm=True))
        # if linear_cfg['type'] == "pws":
        #     layer_list.append(nn.BatchNorm1d(out_channels))

        self.layer = nn.Sequential(*layer_list)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, PWSLinear)):
                m.reset_parameters()

            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
                                nn.GroupNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer(x)
        return x 