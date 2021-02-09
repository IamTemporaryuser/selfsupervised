#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# # runModel simsiam ${DATETIME}_0
python linear_val.py --config configs/imagenet_all/test.py --ckpt result/2021_01_25_11_0/epoch_end.pth --mode FeatureModel --type nocls
# python linear_val.py --config configs/imagenet/simsiam.py --ckpt backup_myselfsup/logs/R50e100_bs512lr0.1/checkpoint_0019.pth.tar --mode Model --knn
# #multigpu=4
# #PORT=29500
# #runModel interclass1 ${DATETIME}_1

# # linearModel simsiam 2021_01_12_11_0
# # evalModel simsiam 2021_01_03_0