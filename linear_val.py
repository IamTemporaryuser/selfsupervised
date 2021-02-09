import argparse

from mmcv.runner import load_checkpoint
from mmcv import Config
import torch
import random
import numpy as np

from modules import *

parser = argparse.ArgumentParser()
parser.add_argument('--config', default="configs/resnet50_imagenet.py", type=str)
parser.add_argument('--ckpt', default="test.pth", type=str)
parser.add_argument('--type', default="cls", type=str)
parser.add_argument('--mode', default="Model", type=str)
parser.add_argument('--multigpu', default=0, type=int)
parser.add_argument('--local_rank', default=0, type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = Config.fromfile(args.config)
    
    dataset = data_loader(config, usemultigpu=False)
    testsize = dataset.testsize()
    train_loader, test_loader = dataset.get_loader()

    logger = Logger(args, config, save_file=False)
    
    model = build_model(config, args.mode)
    print(f'==> evaluate from {args.ckpt}..')
    load_checkpoint(model, args.ckpt)
    model = model.to(device)
    if args.type == "cls":
        evaluate_cls(model, device, test_loader, testsize, logger=logger)()
    elif args.type == "trainknn":
        evaluate_knn(model, device, test_loader, testsize, config, logger=logger)()
    elif args.type == "knn":
        traditional_knn(model, device, [train_loader, test_loader], dataset.trainset.__len__(), config, logger)()
    elif args.type == "nocls":
        evaluate_nocls(model, device, [train_loader, test_loader], dataset.trainset.__len__(), config, logger)()