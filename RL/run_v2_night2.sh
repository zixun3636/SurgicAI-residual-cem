#!/bin/bash
# Night 2 (~7 h): exp.3 (bias flips sign at step 20), exp.4 part 1 (BC-only residual, alpha_eval = 1.0)
cd "$(dirname "$0")" && source ./est_dir_common.sh
V2="$OUT_DIR/v2"; mkdir -p "$V2"

echo "===== [1/3] exp3 flip, clean ====="
python3 eval_estimated_direction_v2.py "${COMMON_ARGS[@]}" \
  --eval_seed 0 --episodes 100 --bias_list=-2,-1,1,2 --flip_step 20 \
  --methods oracle,est_seq,adaptive,residual_only \
  --save_json "$V2/exp3_flip.json" 2>&1 | tee "$LOG_DIR/v2_exp3_flip.log"

echo "===== [2/3] exp3 flip, est_noise_deg=0.5 (adaptive only) ====="
python3 eval_estimated_direction_v2.py "${COMMON_ARGS[@]}" \
  --eval_seed 0 --episodes 100 --bias_list=-2,-1,1,2 --flip_step 20 \
  --methods adaptive --est_noise_deg 0.5 \
  --save_json "$V2/exp3_flip_noise0.5.json" 2>&1 | tee "$LOG_DIR/v2_exp3_flip_noise0.5.log"

# exp4: BC-only residual at both deployment gains (seeds matched with main.json).
# Uses the 25k-step checkpoint: val MSE is lowest near 20k and rises afterwards (overfitting),
# so the final 100k model is NOT used (early stopping on the held-out 10% of expert data).
# alpha_eval=0.1 matches the BC target scale; alpha_eval=1.0 matches the TD3_HER_BC deployment.
for sc in 0.1 1.0; do
  echo "===== [3/3] exp4 BC-only residual, alpha_eval=$sc ====="
  python3 eval_estimated_direction_v2.py "${COMMON_ARGS[@]}" \
    --algorithm BC --model_path "$RL_DIR/insert_residual/BC/sparse/seed_0/checkpoints/bc_25000_steps.zip" \
    --residual_scale $sc --eval_seed 0 --episodes 50 \
    --bias_list=-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5 --methods residual_only \
    --save_json "$V2/exp4_bc_scale${sc}.json" 2>&1 | tee "$LOG_DIR/v2_exp4_bc_scale${sc}.log"
done
echo "===== night 2 done ====="
