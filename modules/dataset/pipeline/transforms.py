import torchvision.transforms as T
import torch
from PIL import Image, ImageFilter
import cv2
import numpy as np
import random

# class GaussianBlur(object):
#     """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709."""

#     def __init__(self, sigma_min, sigma_max, kernel_size=0):
#         if hasattr(T, "GaussianBlur"):
#             self.torchvision = getattr(T, "GaussianBlur")(kernel_size, (sigma_min, sigma_max))
#         else:
#             self.torchvision = None

#         self.sigma_min = sigma_min
#         self.sigma_max = sigma_max

#     def __call__(self, img):
#         if self.torchvision is not None:
#             return self.torchvision(img)
#         else:
#             sigma = np.random.uniform(self.sigma_min, self.sigma_max)
#             img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
#             return img

    # def __repr__(self):
    #     if self.torchvision is not None:
    #         return self.torchvision.__repr__()
            
    #     repr_str = self.__class__.__name__
    #     return repr_str

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma_min=0.1, sigma_max=2.0, kernel_size=0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, x):
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

    def __repr__(self):
        return self.__class__.__name__

__EXCLUDE_DICT = {
    "GaussianBlur":GaussianBlur
}

def build_transforms(transform_list):
    assert len(transform_list) > 0
    if isinstance(transform_list[0], list):
        assert len(transform_list) == 2
        return [build_transforms(transform_list[0]), build_transforms(transform_list[1])]
    
    trans_funcs = []
    for cfg in transform_list:
        t = cfg.pop("type")
        p = cfg.pop("rand_apply") if "rand_apply" in cfg else None
        if t in __EXCLUDE_DICT:
            func = __EXCLUDE_DICT[t](**cfg)
        else:
            func = getattr(T, t)(**cfg)
        
        if p is not None:
            func = T.RandomApply([func], p=p)
        
        trans_funcs.append(func)
    return T.Compose(trans_funcs)