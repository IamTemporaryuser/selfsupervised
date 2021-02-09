import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..utils import build_linear_layer
from mmcv.cnn import (normal_init, constant_init, kaiming_init)
from ..apis import train

def D(p, z, version='simplified'):
    return - F.cosine_similarity(p, z.detach(), dim=-1).mean()

class SimsiamHead(nn.Module):
    def __init__(self, in_channels=2048, hidden_channels=512, out_channels=2048, proj_layers=2, linear_cfg=dict(type="linear")):
        super().__init__()
        self.linear_cfg = linear_cfg

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        proj_list = []
        for _ in range(proj_layers-1):
            proj_list.extend(self.__build_layer(in_channels, out_channels, norm=True, relu=True))
            in_channels = out_channels
        proj_list.extend(self.__build_layer(out_channels, out_channels, norm=True, relu=False))
        self.proj = nn.Sequential(*proj_list)

        pred_list = self.__build_layer(out_channels, hidden_channels, norm=True, relu=True)
        pred_list.extend(self.__build_layer(hidden_channels, out_channels, norm=False, relu=False))
        
        self.pred = nn.Sequential(*pred_list)
    
    def __build_layer(self, in_features, out_features, norm=True, relu=True):
        ret = build_linear_layer(self.linear_cfg, in_features, out_features, norm=False)
        if norm:
            ret.append(nn.BatchNorm1d(out_features))
        if relu:
            ret.append(nn.ReLU(inplace=True))
        return ret

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                m.reset_parameters()

            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
                                nn.GroupNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x1:torch.Tensor, x2:torch.Tensor):

        def __cal_sim(x1, x2):
            z1 = self.proj(x1)
            z2 = self.proj(x2)

            p1 = self.pred(z1)
            p2 = self.pred(z2)
            
            loss_sim = D(p1, z2) / 2 + D(p2, z1) / 2
            return loss_sim, z1, z2
        
        N, C, H, W = x1.size()
        x1 = self.avgpool(x1).view(N, C)
        x2 = self.avgpool(x2).view(N, C)

        outputs = {}

        loss_sim, z1, z2 = __cal_sim(x1, x2)
        outputs['sim'] = loss_sim

        loss = loss_sim
        outputs['loss'] = loss
        
        return outputs