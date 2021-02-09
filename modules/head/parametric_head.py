import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..utils import build_linear_layer
from mmcv.cnn import (normal_init, constant_init, kaiming_init)

class ParametricLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=1)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, input):
        weight = F.normalize(self.weight, dim=1)
        out = F.linear(input, weight, self.bias)
        out = F.normalize(out, dim=-1)
        return out

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None)

class BatchNormFC1d(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.norm = nn.BatchNorm1d(out_features)
        self.init_weights()
    
    def init_weights(self):
        nn.init.constant_(self.norm.weight, 1)
        nn.init.constant_(self.norm.bias, 0)
    
    def forward(self, x):
        N, L, C = x.size()
        x = x.view(N*L, C)
        out = self.norm(x)
        out = out.view(N, L, C)
        return out

def D(p, z, k=1):
    z = z.detach()
    sim = torch.bmm(z, p)
    val, _ = torch.topk(sim, k=k ,dim=1, largest=True)
    return -val.mean()

class ParametricHead(nn.Module):
    def __init__(self, in_features=2048, out_features=2048, num_features=512,
                projlayer=2, avgpool=(2,2), linear_cfg=dict(type="linear")):
        super().__init__()
        self.linear_cfg = linear_cfg
        self.num_features = num_features

        self.avgpool = nn.AdaptiveAvgPool2d(avgpool)
        self.avgpool_1x1 = nn.AdaptiveAvgPool2d((1, 1))
        
        proj_list = []
        for _ in range(projlayer-1):
            proj_list.extend(self.__build_layer(in_features, out_features, norm=True, relu=True))
            in_features = out_features
        proj_list.extend(self.__build_layer(out_features, out_features, norm=True, relu=False))
        self.proj = nn.Sequential(*proj_list)

        pred_list = []
        pred_list.extend(self.__build_layer(out_features, int(out_features / 4), norm=True, relu=True))
        pred_list.extend(self.__build_layer(int(out_features / 4), out_features, norm=False, relu=False))
        self.pred = nn.Sequential(*pred_list)

        self.feature_bank = ParametricLinear(out_features, num_features)
    
    def __build_layer(self, in_features, out_features, norm=True, relu=True):
        ret = build_linear_layer(self.linear_cfg, in_features, out_features, norm=False)
        if norm:
            ret.append(BatchNormFC1d(out_features))
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

    def v(self, x, output):
        N, C, H, W = x.size()
        x = x.reshape(N, C, -1).transpose(2, 1)
        x_var = x.var(dim=1).mean(dim=0)
        output['l2norm'] = torch.norm(x, dim=[1, 2]).mean()
        output['l2normvar'] = torch.norm(x, dim=2).var(dim=1).mean()
        output['var'] = x.var(dim=[1, 2]).mean()
        output['x_varmean'] = x_var.mean()
        output['nozero'] = (x != 0).sum(dim=[1, 2]).type(torch.FloatTensor).mean()

    def forward(self, x1:torch.Tensor, x2:torch.Tensor):

        def __cal_forward(x1, x2, outputs, temperature=0.5):
            N, C, H, W = x1.size()
            x1 = self.avgpool(x1).view(N, C, -1).transpose(2, 1)
            x2 = self.avgpool(x2).view(N, C, -1).transpose(2, 1)

            z1 = self.proj(x1)
            z2 = self.proj(x2)

            p1 = self.pred(z1)
            p2 = self.pred(z2)
            
            z1norm = F.normalize(z1, dim=2)
            z2norm = F.normalize(z2, dim=2)
            p1norm = F.normalize(p1, dim=2)
            p2norm = F.normalize(p2, dim=2)

            p1norm = p1norm.transpose(2, 1)
            p2norm = p2norm.transpose(2, 1)
            
            sim = D(p1norm, z2norm) / 2 + D(p2norm, z1norm) / 2 
            outputs["sim"] = sim
            # # N, L, F -> N, 4, F
            # z1 = self.feature_bank(z1norm)
            # # N, L, F -> N, 1, F
            # z2 = self.feature_bank(z2norm)

            # p1 = torch.softmax(z1 / temperature, dim=2)
            # p2 = torch.softmax(z2 / temperature, dim=2).expand_as(p1)

            # uniform = -(p2 * torch.log(p2)).mean()
            # outputs["uniform"] = uniform

            loss = sim
            outputs['loss'] = loss
            return loss
        
        outputs = {}

        self.v(x1, outputs)
        __cal_forward(x1, x2, outputs)
        
        return outputs 
