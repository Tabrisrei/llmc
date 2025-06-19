#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
export DATASET_SOURCE='HF'
export HF_ENDPOINT=https://hf-mirror.com.

llmc=/home/gsb/LLMCMed/llmc
export PYTHONPATH=$llmc:$PYTHONPATH

company_name=meta-llama
# task_name=sparsegpt_unstructured
task_name=wanda_unstructured
config=${llmc}/configs/XXXDevelopmentG/${task_name}.yml
log_dir=${llmc}/xxx_experiment_logs/${task_name}.log
rm -rf atrans_models/${company_name}/${task_name}

nnodes=1
nproc_per_node=1

find_unused_port() {
    while true; do
        port=$(shuf -i 10000-60000 -n 1)
        if ! ss -tuln | grep -q ":$port "; then
            echo "$port"
            return 0
        fi
    done
}
UNUSED_PORT=$(find_unused_port)


MASTER_ADDR=127.0.0.1
MASTER_PORT=$UNUSED_PORT
task_id=$UNUSED_PORT

# nohup \
torchrun \
--nnodes $nnodes \
--nproc_per_node $nproc_per_node \
--rdzv_id $task_id \
--rdzv_backend c10d \
--rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
${llmc}/llmc/__main__.py --config $config --task_id $task_id #\
# > ${log_dir} 2>&1 &


# debug
# torchrun --nnodes $nnodes --nproc_per_node $nproc_per_node --rdzv_id $task_id --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT ${llmc}/llmc/__main__.py --config $config --task_id $task_id

# sleep 2
# ps aux | grep '__main__.py' | grep $task_id | awk '{print $2}' > ${task_name}.pid

# You can kill this program by 
# xargs kill -9 < xxx.pid
# xxx.pid is ${task_name}.pid file
