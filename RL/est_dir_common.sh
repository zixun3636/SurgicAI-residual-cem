#!/bin/bash
# Shared paths for the estimated-direction experiment (same models/params as 0918).
RL_DIR="/home/suzixun/SurgicAI/RL"
BASE_MODEL="$RL_DIR/Insert/TD3_HER_BC/sparse/seed_0/final_model.zip"
RES_MODEL="$RL_DIR/insert_residual/TD3_HER_BC/sparse/seed_100/final_model.zip"
POS_CEM_PARAM="$RL_DIR/evolution_results/multibias_rot_scale_bias.json"
NEG_CEM_PARAM="$RL_DIR/evolution_results/negative_multibias_rot_scale_bias.json"
OUT_DIR="$RL_DIR/evolution_results/estimated_direction"
LOG_DIR="$RL_DIR/eval_logs_estimated_direction"
mkdir -p "$OUT_DIR" "$LOG_DIR"

COMMON_ARGS=(
  --algorithm TD3_HER_BC --task_name Insert --reward_type sparse
  --trans_error 0.5 --angle_error 10
  --model_path "$RES_MODEL" --base_insert_model_path "$BASE_MODEL" --base_insert_algo TD3_HER_BC
  --residual_scale 1.0 --postprocess_mode rot_scale_bias
  --positive_params_path "$POS_CEM_PARAM" --negative_params_path "$NEG_CEM_PARAM"
)
