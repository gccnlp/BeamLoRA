#!/bin/bash
GPU_IDS="0,1,2,3"
export CUDA_VISIBLE_DEVICES=$GPU_IDS
ABS_PATH="path_to_base_model_folder"
BASE_MODEL="Llama-2-7b-hf"
export WANDB_MODE=offline

batch_size=32
lr_scheduler_type="cosine"
imp_criteria="gate"

RANK=64
echo $RANK
LORA_ALPHA=$((RANK*2))
echo $LORA_ALPHA

cumsum_threshold=0.95
echo $cumsum_threshold
RSTEP=3000
echo ${RSTEP}

SAVE_PROJ="beamlora_"${cumsum_threshold}"_"${RSTEP}"_math_r"${RANK}"_bsz"${batch_size}"_"${lr_scheduler_type}"_395K"

torchrun   --nproc_per_node=4 --master_port=8896 train.py \
           --base_model ${ABS_PATH}/${BASE_MODEL} --micro_batch_size 8 \
            --wandb_run_name ${SAVE_PROJ} --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj \
            --num_epochs 3 --wandb_project lora-math-395K --batch_size ${batch_size} \
            --lora_r ${RANK} --lora_alpha ${LORA_ALPHA} \
            --data_path meta-math/MetaMath \
            --save_steps 8000 \
            --learning_rate 3e-4 \
            --logging_steps 5  --use_bf16  --use_16bit --lr_scheduler_type ${lr_scheduler_type} \
            --use_beamlora --use_beamlora_step ${RSTEP} --imp_criteria ${imp_criteria} \
            --use_inherit_param --reset_optimizer --cumsum_threshold ${cumsum_threshold} --reset_end_step 24688 \
            --dataset_split "train"


cd math_infer
source ~/miniconda3/bin/activate math_infer
sleep 1
SAVE_PATH="../ckpts/"${SAVE_PROJ}
rm -rf ${SAVE_PATH}/"check*"
sleep 1
CUDA_VISIBLE_DEVICES=$GPU_IDS python gsm8k_infer.py --model ${ABS_PATH}/${BASE_MODEL} \
--data_file 'data/gsm8k_test.jsonl' \
--peft_id $SAVE_PATH \
--tensor_parallel_size 4 \
--batch_size 240
CUDA_VISIBLE_DEVICES=$GPU_IDS python math_infer.py --model ${ABS_PATH}/${BASE_MODEL} \
--data_file 'data/MATH_test.jsonl' \
--peft_id $SAVE_PATH \
--tensor_parallel_size 4 \
--batch_size 200