#!/bin/bash

llmc=/home/gsb/LLMCMed/llmc
export PYTHONPATH=$llmc:$PYTHONPATH


# task_name=awq_w_only_8b_Meta-Llama-3-8B
# task_name=gptq_w_only_8b_Meta-Llama-3-8B
# task_name=omqu_w_only_8b_Meta-Llama-3-8B  # bug exists
# task_name=omqu_w_a_8b_Meta-Llama-3-8B
# task_name=smqu_w_only_8b_Meta-Llama-3-8B
# task_name=omqu_w_only_8b_Qwen2.5-7B
# task_name=wanda_50_Meta-Llama-3-8B
# task_name=shortgpt_9_Meta-Llama-3-8B
# task_name=spqr_w_only_8b_Meta-Llama-3.2-1Ba
# task_name=quarot_w_only_8b_Meta-Llama-3-8B

company_name=meta-llama
# company_name=Qwen
# company_name=FreedomIntelligence
# company_name=zhaohe9701

# task_name=gptq_w_only_8b_Meta-Llama-3-70B
task_name=gptq_w_only_8b_Meta-Llama-3.1-70B
# task_name=gptq_w_only_8b_Qwen2.5-7B
# task_name=gptq_w_only_8b_Qwen2.5-32B
# task_name=gptq_w_only_8b_Qwen2.5-72B
# task_name=gptq_w_only_8b_HuatuoGPT-o1-70B
# task_name=gptq_w_only_8b_HuatuoGPT-o1-72B


# task_name=awq_w_only_4b_Meta-Llama-3-8B
# task_name=gptq_w_only_4b_Meta-Llama-3-8B
# task_name=omqu_w_only_4b_Meta-Llama-3-8B  # bug exists
# task_name=smqu_w_only_4b_Meta-Llama-3-8B
# task_name=wanda_25_Meta-Llama-3-8B
# task_name=shortgpt_5_Meta-Llama-3-8B

config=${llmc}/configs/AFormalExperimentG/${task_name}.yml
log_dir=${llmc}/a_experiment_logs/${task_name}.log

rm -rf atrans_models/${company_name}/${task_name}
# rm -rf atrans_models/Qwen/${task_name}


export CUDA_VISIBLE_DEVICES=5
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

nohup \
torchrun \
--nnodes $nnodes \
--nproc_per_node $nproc_per_node \
--rdzv_id $task_id \
--rdzv_backend c10d \
--rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
${llmc}/llmc/__main__.py --config $config --task_id $task_id \
> ${log_dir} 2>&1 &

# debug
# torchrun --nnodes $nnodes --nproc_per_node $nproc_per_node --rdzv_id $task_id --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT ${llmc}/llmc/__main__.py --config $config --task_id $task_id


# sleep 2
# ps aux | grep '__main__.py' | grep $task_id | awk '{print $2}' > ${task_name}.pid

# You can kill this program by 
# xargs kill -9 < xxx.pid
# xxx.pid is ${task_name}.pid file
