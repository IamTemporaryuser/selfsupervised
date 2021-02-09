from .. import backbone
from .. import head
from .. import neck
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint
import logging
import os

__all__ = ["Model", "SSModel", "SaliencyModel", "FeatureModel"]

backbone_dict = {}
for name in getattr(backbone, "__all__"):
    backbone_dict[name] = getattr(backbone, name)

neck_dict = {}
for name in getattr(neck, "__all__"):
    neck_dict[name] = getattr(neck, name)

head_dict = {}
for name in getattr(head, "__all__"):
    head_dict[name] = getattr(head, name)

module_dict = {
    "backbone":backbone_dict,
    "neck":neck_dict,
    "head":head_dict
}

def register(model: nn.Module, cfg, key, value):
    if key in cfg:
        module_cfg = cfg.pop(key)
        module_type = module_cfg.pop("type")

        if module_type in module_dict[value]:
            module = module_dict[value][module_type](**module_cfg)
        else:
            raise ValueError(f'{key}_type->{value}_type={module_type} does not support.')
    else:
        module = None
    
    model.add_module(key, module)

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        build_dict = {
            "backbone":"backbone",
            "neck":"neck"
        }

        for k, v in build_dict.items():
            register(self, cfg, k, v)

        self.backbone.init_weights()
        if self.neck is not None:
            self.neck.init_weights()
        if "pretrained" in cfg:
            pretrained = cfg.pop("pretrained")
            if pretrained is not None:
                load_checkpoint(self, pretrained, strict=True, map_location="cpu")

        register(self, cfg, "head", "head")
        self.head.init_weights()
        
    
    def forward(self, x, label=None, forward_knn=False):
        if forward_knn:
            return self.forward_knn(x)

        out = self.backbone(x)
        if self.neck is not None:
            out = self.neck(out)
        out = self.head(out, label)
        return out
    
    def forward_knn(self, x):
        out = self.backbone(x)
        if self.neck is not None:
            out = self.neck(out)
        
        if len(out.size()) == 4:
            out = out.mean(axis=[2, 3])
        return out

class SSModel(nn.Module):
    def __init__(self, cfg):
        super(SSModel, self).__init__()
        build_dict = {
            "backbone":"backbone",
            "neck":"neck",
            "proj":"neck",
            "head":"head"
        }

        for k, v in build_dict.items():
            register(self, cfg, k, v)

        self.backbone.init_weights()
        if self.neck is not None:
            self.neck.init_weights()
        if self.proj is not None:
            self.proj.init_weights()
        self.head.init_weights()
        
        if "pretrained" in cfg:
            pretrained = cfg.pop("pretrained")
            if pretrained is not None:
                load_checkpoint(self, pretrained, strict=True, map_location="cpu")
    
    def forward(self, x1, x2=None, forward_knn=False):
        if forward_knn:
            return self.forward_knn(x1)

        z1 = self.forward_knn(x1)
        z2 = self.forward_knn(x2)
        
        if self.proj is not None:
            z1 = self.proj(z1)
            z2 = self.proj(z2)

        out = self.head(z1, z2)
        return out
    
    def forward_knn(self, x):
        out = self.backbone(x)
        if self.neck is not None:
            out = self.neck(out)

        return out

class SaliencyModel(nn.Module):
    def __init__(self, cfg):
        super(SaliencyModel, self).__init__()
        build_dict = {
            "backbone":"backbone",
            "neck":"neck",
            "head":"head"
        }

        for k, v in build_dict.items():
            register(self, cfg, k, v)

        self.backbone.init_weights()
        if self.neck is not None:
            self.neck.init_weights()
        self.head.init_weights()
        if "pretrained" in cfg:
            pretrained = cfg.pop("pretrained")
            if pretrained is not None:
                load_checkpoint(self, pretrained, strict=False, map_location="cpu")
    
    def forward(self, x, label=None, forward_knn=False):
        if forward_knn:
            return self.forward_knn(x)

        if x.dim() == 5:
            x1, x2 = torch.chunk(x, 2, dim=1)
            x1 = x1.squeeze(dim=1)
            x2 = x2.squeeze(dim=1)

            z1 = self.backbone(x1)
            z1 = self.neck(z1)

            z2 = self.backbone(x2)
            z2 = self.neck(z2)
            out = torch.stack((z1, z2), dim=0)
        else:
            out = self.backbone(x)
            out = self.neck(out)
        out = self.head(out, label)
        return out
    
    def forward_knn(self, x):
        out = self.backbone(x)
        if self.neck is not None:
            out = self.neck(out)
        
        if len(out.size()) == 4:
            out = out.mean(axis=[2, 3])
        return out

class FeatureModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        build_dict = {
            "backbone":"backbone",
            "neck":"neck",
            "head":"head"
        }
        for k, v in build_dict.items():
            register(self, cfg, k, v)

        self.backbone.init_weights()
        if self.neck is not None:
            self.neck.init_weights()
        self.head.init_weights()

        if "pretrained" in cfg:
            pretrained = cfg.pop("pretrained")
            if pretrained is not None:
                load_checkpoint(self, pretrained, strict=False, map_location="cpu")
    
    def forward(self, x, label):
        raise ValueError("forward does not support.")
    
    def forward_knn(self, x):
        out = self.backbone(x)
        if self.neck is not None:
            out = self.neck(out)
        out = self.head.evaluate(out)
        return out