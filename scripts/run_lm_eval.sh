export CUDA_VISIBLE_DEVICES=1,7
llmc=/home/gsb/LLMCMed/llmc
lm_eval=/home/gsb/LLMCMed/llmc/lm-evaluation-harness
export PYTHONPATH=$llmc:$PYTHONPATH
export PYTHONPATH=$llmc:$lm_eval:$PYTHONPATH
# Replace the config file (i.e., RTN with algorithm-transformed model path or notate quant with original model path) 
# with the one you want to use. `--quarot` is depend on the transformation algorithm used before.
accelerate launch --multi_gpu --num_processes 2 /home/gsb/LLMCMed/llmc/tools/llm_eval.py \
    --config /home/gsb/LLMCMed/llmc/configs/quantization/methods/AAAExperimentG/gptq_w_only_HuatuoGPT_o1_7B.yml \
    --model hf \
    --quarot \
    --tasks lambada_openai,arc_easy \
    --model_args parallelize=False,trust_remote_code=True \
    --batch_size 64 \
    --output_path ./save/lm_eval \
    --log_samples