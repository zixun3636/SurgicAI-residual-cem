#!/bin/bash


BASE_MODEL="/home/suzixun/SurgicAI/RL/Insert/TD3_HER_BC/sparse/seed_0/final_model.zip"

RES_MODEL="/home/suzixun/SurgicAI/RL/insert_residual/TD3_HER_BC/sparse/seed_100/final_model.zip"

POS_CEM_PARAM="/home/suzixun/SurgicAI/RL/evolution_results/multibias_rot_scale_bias.json"

NEG_CEM_PARAM="/home/suzixun/SurgicAI/RL/evolution_results/negative_multibias_rot_scale_bias.json"


for noise in 0.0 0.05 0.1 0.15 0.2
do


echo "================================="
echo "Action noise std=$noise"
echo "================================="


python3 eval_direction_conditioned_cem.py \
  --postprocess_mode rot_scale_bias \
  --algorithm TD3_HER_BC \
  --task_name Insert \
  --reward_type sparse \
  --trans_error 0.5 \
  --angle_error 10 \
  --eval_seed 100 \
  --episodes 100 \
  --bias_list=-2,-1.5,-1,-0.5,0.5,1,1.5,2 \
  --action_noise_std $noise \
  --model_path "$RES_MODEL" \
  --base_insert_model_path "$BASE_MODEL" \
  --base_insert_algo TD3_HER_BC \
  --residual_scale 1.0 \
  --positive_params_path "$POS_CEM_PARAM" \
  --negative_params_path "$NEG_CEM_PARAM"


done
