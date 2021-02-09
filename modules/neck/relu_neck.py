import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (build_norm_layer, constant_init)
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmcv.runner import load_checkpoint

class LayerNorm1D(nn.Module):
    def __init__(self, num_channels, eps=1e-5, affine=True, requires_grad=True):
        super(LayerNorm1D, self).__init__()
        self.norm_shape = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_channels))
            self.bias = nn.Parameter(torch.Tensor(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input):
        N, C, H, W = input.size()
        input = input.reshape(N, C, H * W).transpose(2, 1)
        output = self.weight * F.layer_norm(input, [self.norm_shape], eps=self.eps) + self.bias
        return output.transpose(2, 1).reshape(N, C, H, W)

    def extra_repr(self) -> str:
        return '{norm_shape}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)

class ReluNeck(nn.Module):

    def __init__(self, in_channels, frozen_state=False, norm_cfg=None, avgpool=False):
        super(ReluNeck, self).__init__()
        self.norm_cfg = norm_cfg
        self.frozen_state = frozen_state
        if self.norm_cfg is not None:
            if "LN" == norm_cfg['type']:
                self.norm = LayerNorm1D(in_channels)
            else:
                self.norm = build_norm_layer(self.norm_cfg, in_channels)[1]
        else:
            self.norm = None

        self.relu = nn.ReLU(inplace=False)
        if avgpool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avgpool = None
            
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_state and self.norm is not None:
            self.norm.eval()
            for param in self.norm.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
        
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained)

    def train(self, mode=True):
        super(ReluNeck, self).train(mode)
        self._freeze_stages()

    def forward(self, x):
        if self.norm:
            x = self.norm(x)
        
        outs = self.relu(x)
        if self.avgpool is not None:
            outs = self.avgpool(outs).view(outs.size(0), -1)
        return outs

