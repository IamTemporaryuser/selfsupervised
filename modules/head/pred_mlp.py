import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..utils import PWSLinear, build_linear_layer
from mmcv.cnn import (normal_init, constant_init, kaiming_init)

def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()
    elif version == 'simplified':
        return - F.cosine_similarity(p, z.detach(), dim=-1)
    else:
        raise Exception

class Pred_MLP(nn.Module):
    def __init__(self, in_channels=2048, hidden_channels=512, out_channels=2048, linear_cfg=dict(type="linear")): # bottleneck structure
        super().__init__()
        layer_list = build_linear_layer(linear_cfg, in_channels, hidden_channels, norm=True)
        layer_list.append(nn.ReLU(inplace=True))
        layer_list.extend(build_linear_layer(linear_cfg, hidden_channels, out_channels, norm=False))
        self.layer = nn.Sequential(*layer_list)

        self.hidden_channels = hidden_channels
        # self.layer = nn.Sequential(nn.Identity())

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
        if isinstance(self.layer[-1], (PWSLinear)):
            nn.init.uniform_(self.layer[-1].bias, -1 / math.sqrt(self.hidden_channels), 1 / math.sqrt(self.hidden_channels))

    def forward(self, z1:torch.Tensor, z2:torch.Tensor):
        p1 = self.layer(z1)
        p2 = self.layer(z2)
        
        losses = {}
        sim1 = D(p1, z2)
        sim2 = D(p2, z1)
        losses['loss'] = sim1.mean() / 2 + sim2.mean() / 2

        losses['sim_var'] = sim1.var()
        losses['output_std'] = z1.var(axis=1).sqrt().mean()
        losses['output_mean'] = z1.mean(axis=1).mean()
        
        return losses 
