import argparse

from mmcv.runner import load_checkpoint
from mmcv import Config
import torch
import random
import numpy as np

from modules import *

parser = argparse.ArgumentParser()
parser.add_argument('--config', default="configs/resnet50_imagenet.py", type=str)
parser.add_argument('--ckpt', default="/home/t/ckpt/cifar10.pth", type=str)
parser.add_argument('--knn', action="store_true")
parser.add_argument('--mode', default="Model", type=str)
parser.add_argument('--local_rank', default=0, type=int)


def sinkhorn(scores, eps=0.25, niters=3): 
    Q = torch.exp(scores / eps).T 
    Q /= torch.sum(Q) 
    K, B = Q.shape 
    # r, c = 1 / K, 1 / B
    u, r, c = torch.zeros(K).cuda(), torch.ones(K).cuda() / K, torch.ones(B).cuda() / B 
    for _ in range(niters):
        u = torch.sum(Q, dim=1) 
        Q *= (r / u).unsqueeze(1) 
        Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0) 
    return (Q / torch.sum(Q, dim=0, keepdim=True)).T

def getv(t):
    print(t[:4].T)
    return t[:4].T.detach()

def tostr(t):
    t = t.cpu().numpy()
    s = ""
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            s += "{},".format(t[i,j])
        s += '0\n'
    
    return s

def write(t):
    s = tostr(t)
    with open("test.csv", "w") as f:
        f.write(s)


if __name__ == "__main__":
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = Config.fromfile(args.config)
    
    dataset = data_loader(config, usemultigpu=False)
    testsize = dataset.testsize()
    train_loader, test_loader = dataset.get_loader()
    
    model = build_model(config, args.mode)
    print(f'==> evaluate from {args.ckpt}..')
    # load_checkpoint(model, args.ckpt, strict=False)

    state_dict = torch.load(args.ckpt)['state_dict']
    for k in list(state_dict.keys()):
        if ('proj' in k or 'pred' in k) and 'norm' in k:
            state_dict[k.replace("norm.", "")] = state_dict[k]
            del state_dict[k]
    msg = model.load_state_dict(state_dict, strict=False)

    model = model.to(device)

    model.train()
    
    data_iter = iter(train_loader)

    inputs, targets = next(data_iter)
    inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

    outputs = model(inputs, targets)
    # z1, p1, q1 = outputs['probresult']
