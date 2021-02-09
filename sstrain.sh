#!/bin/bash
ssModel(){
    CONFIG_FILE=configs/${DATASET}/$1.py
    WORK_DIR=result/$2
    rm -rf ${WORK_DIR}

    if [ ${multigpu} -gt 1 ];then
        python -m torch.distributed.launch --nproc_per_node=${multigpu} --master_addr 127.0.0.1 --master_port ${PORT} \
            ss_train.py --config ${CONFIG_FILE} --workdir ${WORK_DIR} --multigpu ${multigpu}
    else
        python ss_train.py --config ${CONFIG_FILE} --workdir ${WORK_DIR}
    fi
}
transCKPT(){
    WORK_DIR=result/$1
    LAST_CKPT=${WORK_DIR}/epoch_end.pth
    BACKBONE_CKPT=${WORK_DIR}/backbone.pth
    python tools/process_ckpt.py --checkpoint ${LAST_CKPT} --output ${BACKBONE_CKPT}
}
linearModel(){
    LINEAR_CFG_FILE=configs/${DATASET}/simsiam_linear.py
    LINEAR_WORK_DIR=result/$1/linear
    BACKBONE_CKPT=$2

    rm -rf ${LINEAR_WORK_DIR}

    if [ ${multigpu} -gt 1 ];then
        python -m torch.distributed.launch --nproc_per_node=${multigpu} --master_addr 127.0.0.1 --master_port ${PORT} \
            linear_train.py --config ${LINEAR_CFG_FILE} --workdir ${LINEAR_WORK_DIR} --pretrained ${BACKBONE_CKPT} --multigpu ${multigpu}
    else
        python linear_train.py --config ${LINEAR_CFG_FILE} --workdir ${LINEAR_WORK_DIR} --pretrained ${BACKBONE_CKPT}
    fi

    python tools/get_max.py --workdir ${LINEAR_WORK_DIR}
}
evalModel(){
    CONFIG_FILE=configs/${DATASET}/$1.py
    WORK_DIR=result/$2
    BACKBONE_CKPT=${WORK_DIR}/epoch_end.pth

    LINEAR_WORK_DIR=${WORK_DIR}/$2_linear.py
    rm -rf ${LINEAR_WORK_DIR}
    python linear_val.py --config ${CONFIG_FILE} --ckpt ${BACKBONE_CKPT} --mode FeatureModel --knn
}

saliencyModel(){
    LINEAR_CFG_FILE=configs/${DATASET}/saliency_linear.py
    WORK_DIR=result/$2

    BACKBONE_CKPT=${WORK_DIR}/epoch_end.pth

    LINEAR_WORK_DIR=${WORK_DIR}/$2_linear.py
    rm -rf ${LINEAR_WORK_DIR}

    python -m pdb linear_train.py --config ${LINEAR_CFG_FILE} --workdir ${LINEAR_WORK_DIR} --pretrained ${BACKBONE_CKPT} --mode SaliencyModel

    python tools/get_max.py --workdir ${LINEAR_WORK_DIR}
}

runModel(){
    ssModel $1 $2
    transCKPT $2
    BACKBONE_CKPT=result/$2/backbone.pth
    linearModel $2 ${BACKBONE_CKPT}
}
linclsModel(){
    transCKPT $1
    BACKBONE_CKPT=result/$1/backbone.pth
    linearModel $1 ${BACKBONE_CKPT}
}

export CUDA_VISIBLE_DEVICES=0,1,2,3

DATASET=cifar10
DATETIME=`date +"%Y_%m_%d_%H" `
multigpu=4
PORT=28500

runModel simsiam ${DATETIME}_0
# linclsModel 2021_01_25_11_0

# linearModel test result/2021_01_25_11_0/backbone.pth
# runModel interclass2 ${DATETIME}_2
# linclsModel 2021_01_22_15_0
# linearModel test /home/lyx/ckpt/github_simsiam_500.pth
# ssModel interclass1 ${DATETIME}_1

# runModel interclass1 ${DATETIME}_0
# runModel interclass2 ${DATETIME}_1

# linearModel test /home/t/ckpt/interclass_bs32.pth
#multigpu=4
#PORT=29500
#runModel interclass1 ${DATETIME}_1

# linearModel simsiam 2021_01_12_11_0
# evalModel simsiam 2021_01_03_0