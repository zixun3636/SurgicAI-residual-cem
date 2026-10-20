#!/bin/bash
# Night 1 (~7-8 h): BC-only training, exp.1 (estimator variants + signed saturation), exp.2 (random 3-DOF)
cd "$(dirname "$0")" && source ./est_dir_common.sh
V2="$OUT_DIR/v2"; mkdir -p "$V2"

echo "===== [1/4] train BC-only residual (seed 0, 100k gradient steps) ====="
python3 train_residual_bc_only.py --seed 0 --gradient_steps 100000 2>&1 | tee "$LOG_DIR/bc_train_seed0.log"

for noise in 0.5 1.0; do
  echo "===== [2/4] exp1 estimators, est_noise_deg=$noise ====="
  python3 eval_estimated_direction_v2.py "${COMMON_ARGS[@]}" \
    --eval_seed 0 --episodes 50 --bias_list=-2,-0.5,0.5,2 \
    --methods est_v1,est_pitch,est_seq,est_seq_pitch --warmup_steps 20 \
    --est_noise_deg $noise \
    --save_json "$V2/exp1_noise_${noise}.json" 2>&1 | tee "$LOG_DIR/v2_exp1_noise_${noise}.log"
done

echo "===== [3/4] exp2 random 3-DOF ====="
python3 eval_estimated_direction_v2.py "${COMMON_ARGS[@]}" \
  --bias_mode random3d --eval_seed 2000 --episodes 200 \
  --methods oracle,residual_only,est_v1,est_seq,est_seq_pitch --warmup_steps 20 \
  --save_json "$V2/exp2_random3d.json" 2>&1 | tee "$LOG_DIR/v2_exp2_random3d.log"

echo "===== [4/4] exp1 clean run (signed saturation diagnostics) ====="
python3 eval_estimated_direction_v2.py "${COMMON_ARGS[@]}" \
  --eval_seed 0 --episodes 50 --bias_list=-2,-1,-0.5,0.5,1,2 \
  --methods oracle,est_seq,est_seq_pitch \
  --save_json "$V2/exp1_clean.json" 2>&1 | tee "$LOG_DIR/v2_exp1_clean.log"
echo "===== night 1 done ====="
