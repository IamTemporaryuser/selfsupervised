#!/bin/bash
runModel(){
    CONFIG_FILE=configs/${DATASET}/$1.py
    WORK_DIR=result/$2
    rm -rf ${WORK_DIR}
    if [ ${multigpu} -gt 1 ];then
        python -m torch.distributed.launch --nproc_per_node=${multigpu} --master_addr 127.0.0.1 --master_port ${PORT} \
            linear_train.py --config ${CONFIG_FILE} --workdir ${WORK_DIR} --multigpu ${multigpu}
    else
        python -m pdb linear_train.py --config ${CONFIG_FILE} --workdir ${WORK_DIR}
    fi
    python tools/get_max.py --workdir ${WORK_DIR}
}
pdbModel(){
    CONFIG_FILE=configs/${DATASET}/$1.py
    WORK_DIR=result/$2
    python -m pdb linear_train.py --config ${CONFIG_FILE} --workdir ${WORK_DIR}
}

export CUDA_VISIBLE_DEVICES=0,1,2,3

DATASET=cifar10
DATETIME=`date +"%Y_%m_%d_%H" `
multigpu=4
PORT=28500
# runModel pws ${DATETIME}_pws
runModel bn cifar_bn