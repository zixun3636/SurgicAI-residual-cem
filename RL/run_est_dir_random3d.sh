#!/bin/bash
# Step 3 (optional): random 3-DOF bias U(-2,2) per axis, 200 episodes.
# The estimator pools all axes, so this tests whether a single sign still helps.
cd "$(dirname "$0")" && source ./est_dir_common.sh
python3 eval_estimated_direction_cem.py "${COMMON_ARGS[@]}" \
  --bias_mode random3d --eval_seed 2000 --episodes 200 \
  --methods estimated,residual_only,pos_only \
  --save_json "$OUT_DIR/random3d.json" 2>&1 | tee "$LOG_DIR/random3d.log"
