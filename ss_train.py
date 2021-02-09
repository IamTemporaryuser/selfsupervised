import argparse
import os
from mmcv import Config
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from modules import *

parser = argparse.ArgumentParser()
parser.add_argument('--config', default="configs/simsiam.py", type=str)
parser.add_argument('--workdir', default="", type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--mode', default="SSModel", type=str)
parser.add_argument('--multigpu', default=0, type=int)
parser.add_argument('--local_rank', default=0, type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    
    usemultigpu = (args.multigpu > 0)
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if usemultigpu:
        device=torch.device('cuda:{}'.format(args.local_rank))
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', world_size=args.multigpu)
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.environ['RANK'] = "0"

    config = Config.fromfile(args.config)
    
    if usemultigpu:
        config['dataset']['num_workers'] = int(config['dataset']['num_workers'] / args.multigpu)
        config['dataset']['batchsize'] = int(config['dataset']['batchsize'] / args.multigpu)

    dataset = data_loader(config, usemultigpu)
    testsize = dataset.testsize()
    train_loader, test_loader = dataset.get_loader()

    logger = Logger(args, config)
    logger.print(f"{config.text}")
    logger.print(f"{dataset.transform_train}")
    logger.print(f"{dataset.transform_test}")
    saver = Saver(config, logger)

    if args.resume is None:
        print('==> Training from scratch..')
        ckpt = None
        start_epoch = -1
    else:
        print(f'==> Resuming from {args.resume}..')
        ckpt = torch.load(args.resume, map_location='cpu')
        start_epoch = ckpt['epoch']

    model = build_model(config, args.mode)
    logger.print(f"{model}")
    if usemultigpu:
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model.to(device), device_ids=[args.local_rank], output_device=args.local_rank)
    else:
        model = model.to(device)

    optimizer = build_optimizer(config, model, ckpt)

    scheduler = build_lrscheduler(config, optimizer, start_epoch)
    total_epoch = config["total_epochs"]

    evaluate = evaluate_knn(model, device, test_loader, testsize, config, logger)
    evaluate_interval = config['evaluate_interval'] if 'evaluate_interval' in config else 1

    update_interval = config['update_interval'] if 'update_interval' in config else 1

    cudnn.benchmark = True

    train = epoch_train_multigpu if usemultigpu else epoch_train

    for epoch in range(start_epoch + 1, total_epoch):
        dataset.set_epoch(epoch)
        train(model, epoch, device, train_loader, optimizer, scheduler, logger, saver, update_interval)
        
        if (epoch + 1) % evaluate_interval == 0 or (epoch + 1) == total_epoch:
            if args.local_rank == 0:
                evaluate(epoch)
        
        
        