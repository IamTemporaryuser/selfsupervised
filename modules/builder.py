import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from .utils import register
from .utils import LARC

def build_model(cfg, mode='Model'):
    model_dict = {}
    for n in register.__all__:
        model_dict[n] = getattr(register, n)
    model = model_dict[mode](cfg)
    return model

def build_optimizer(cfg, model, ckpt=None):

    para_alpha = [p[1] for p in model.named_parameters() if 'alpha' in p[0]]
    paras = [p[1] for p in model.named_parameters() if not ('alpha' in p[0])]

    cfgoptimizer = cfg.pop("optimizer")
    opt_type = cfgoptimizer.pop("type")

    if "alpha_wd" in cfgoptimizer:
        weightdecay_alpha = cfgoptimizer.pop("alpha_wd")
    else:
        weightdecay_alpha = 0

    if opt_type == "SGD":
        optimizer = torch.optim.SGD([{'params': para_alpha, 'weight_decay': weightdecay_alpha}, {'params': paras}], **cfgoptimizer)
    elif opt_type == "LARC":
        base_optimizer = torch.optim.SGD([{'params': para_alpha, 'weight_decay': weightdecay_alpha}, {'params': paras}], **cfgoptimizer)
        optimizer = LARC(base_optimizer, 0.001, False)
    else:
        raise ValueError(f'optimizer={opt_type} does not support.')

    if ckpt:
        print('==> Resuming...')
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        # print(optimizer.state_dict())
    return optimizer

def build_lrscheduler(cfg, optimizer, last_epoch=-1):
    lr_cfg = cfg.pop("lr_config")
    lr_policy = lr_cfg.pop("policy")

    if lr_policy == "step":
        scheduler = lr_scheduler.MultiStepLR(optimizer, last_epoch=last_epoch, **lr_cfg)
    elif lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, last_epoch=last_epoch, **lr_cfg)
    else:
        raise ValueError(f'scheduler={lr_policy} does not support.')

    return scheduler
