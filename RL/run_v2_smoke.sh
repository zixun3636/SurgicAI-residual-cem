#!/bin/bash
# Quick check (~10 min) before the overnight runs. Run with AMBF open.
cd "$(dirname "$0")" && source ./est_dir_common.sh
echo "===== [1/2] eval v2 smoke ====="
python3 eval_estimated_direction_v2.py "${COMMON_ARGS[@]}" \
  --eval_seed 0 --episodes 2 --bias_list=-2,2 --flip_step 20 \
  --methods oracle,est_seq_pitch,adaptive \
  --save_json "$OUT_DIR/v2_smoke.json" 2>&1 | tee "$LOG_DIR/v2_smoke.log" | grep -E "bias=|Error|error|Traceback"
echo "===== [2/2] BC-only training smoke (2000 steps, seed 999) ====="
python3 train_residual_bc_only.py --seed 999 --gradient_steps 2000 --ckpt_every 1000000 \
  2>&1 | tee "$LOG_DIR/bc_smoke.log" | grep -E "step|Saved|clipped|transitions|Error|error|Traceback"
echo "===== [3/3] load the BC model in eval ====="
python3 eval_estimated_direction_v2.py "${COMMON_ARGS[@]}" \
  --algorithm BC --model_path "$RL_DIR/insert_residual/BC/sparse/seed_999/final_model.zip" \
  --eval_seed 0 --episodes 2 --bias_list=0 --methods residual_only \
  --save_json "$OUT_DIR/v2_smoke_bc.json" 2>&1 | grep -E "bias=|Error|error|Traceback"
echo "===== smoke done ====="
