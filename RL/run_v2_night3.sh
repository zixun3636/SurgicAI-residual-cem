#!/bin/bash
# Night 3 (~5-6 h)
#  [1] exp4 part 2: TD3_HER_BC residual at alpha_eval = 0.1
#  [2] exp1b: commanded-action filter estimators (est_cmd, est_cmd_seq) under pose noise
#  [3] model check: residual seed_0 (used in the 0918 random 3-DOF experiment) on the SAME
#      random 3-DOF bias bank as exp2 (which used residual seed_100)
cd "$(dirname "$0")" && source ./est_dir_common.sh
V2="$OUT_DIR/v2"; mkdir -p "$V2"

echo "===== [1/3] TD3_HER_BC residual, alpha_eval=0.1 ====="
python3 eval_estimated_direction_v2.py "${COMMON_ARGS[@]}" \
  --residual_scale 0.1 --eval_seed 0 --episodes 50 \
  --bias_list=-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5 --methods residual_only \
  --save_json "$V2/exp4_td3_scale0.1.json" 2>&1 | tee "$LOG_DIR/v2_exp4_td3_scale0.1.log"

for noise in 0.5 1.0; do
  echo "===== [2/3] exp1b commanded-action filter, est_noise_deg=$noise ====="
  python3 eval_estimated_direction_v2.py "${COMMON_ARGS[@]}" \
    --eval_seed 0 --episodes 50 --bias_list=-2,-0.5,0.5,2 \
    --methods est_cmd,est_cmd_seq --warmup_steps 20 --est_noise_deg $noise \
    --save_json "$V2/exp1b_cmd_noise_${noise}.json" 2>&1 | tee "$LOG_DIR/v2_exp1b_cmd_noise_${noise}.log"
done

echo "===== [3/3] residual seed_0 on the exp2 random 3-DOF bias bank ====="
python3 eval_estimated_direction_v2.py "${COMMON_ARGS[@]}" \
  --model_path "$RL_DIR/insert_residual/TD3_HER_BC/sparse/seed_0/final_model.zip" \
  --bias_mode random3d --eval_seed 2000 --episodes 200 --methods residual_only \
  --save_json "$V2/exp2_random3d_res_seed0.json" 2>&1 | tee "$LOG_DIR/v2_exp2_random3d_res_seed0.log"
echo "===== night 3 done ====="
