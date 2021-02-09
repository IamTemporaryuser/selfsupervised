import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..utils import Accuracy
from ..utils import build_linear_layer
from mmcv.cnn import (normal_init, constant_init, kaiming_init)

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

class ParametricLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, norm=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.setproj = False
        if self.setproj:
            self.proj = nn.Linear(in_features=in_features, out_features=in_features)

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        if norm:
            self.norm = nn.BatchNorm1d(out_features, affine=False)
        else:
            self.norm = None

        self.reset_parameters()

    def reset_parameters(self):
        if self.setproj:
            self.proj.reset_parameters()
            nn.init.constant_(self.proj.bias, 0)

        nn.init.kaiming_uniform_(self.weight, a=1)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        weight = F.normalize(self.weight, dim=1)

        if self.setproj:
            x = self.proj(x)
            
        x = F.normalize(x, dim=-1)

        out = F.linear(x, weight, self.bias)

        if self.norm is not None:
            out = self.norm(out)
        return out

def cross_entropy(pred, label):
    loss = F.cross_entropy(pred, label, reduction='none')
    loss = loss.mean()
    return loss

class SaliencyCLSHead(nn.Module):
    def __init__(self, num_classes=1000, in_channels=2048, topk=(1,)):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        out_channels = 2048
        num_features = 512
        self.feature_bank = ParametricLinear(out_channels, num_features)
        self.feature_cls = nn.Linear(num_features, num_classes)

        proj_layers = 3
        proj_list = []
        for _ in range(proj_layers-1):
            proj_list.extend(self.__build_layer(in_channels, out_channels, norm=True, relu=True))
            in_channels = out_channels
        proj_list.extend(self.__build_layer(out_channels, out_channels, norm=True, relu=False))
        self.proj = nn.Sequential(*proj_list)

        self.topk = topk
        self.accuracy = Accuracy(self.topk)

    def __build_layer(self, in_features, out_features, norm=True, relu=True):
        ret = [nn.Linear(in_features, out_features)]
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
    
    def loss(self, cls_score, gt_label, outputs):
        num_samples = len(cls_score)
        loss = cross_entropy(cls_score, gt_label)
        # compute accuracy
        acc = self.accuracy(cls_score, gt_label)
        outputs['loss'] = loss
        outputs['accuracy'] = {f'top-{k}': a for k, a in zip(self.topk, acc)}

    def forward(self, x, gt_label):

        def __cal_loss(x, outputs):
            x = self.proj(x).squeeze(1)
            x = self.feature_bank(x)
            logits = self.feature_cls(x)
            self.loss(logits, gt_label, outputs)

        outputs = dict()
        out = self.avgpool(x).view(x.size(0), 1, -1)
        __cal_loss(out, outputs)
        
        return outputs 
