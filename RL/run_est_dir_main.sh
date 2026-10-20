#!/bin/bash
# Step 1 (main experiment, run overnight):
# 11 scalar biases (incl. unseen +-2.5 and d=0) x 4 methods x 100 episodes, matched seeds.
cd "$(dirname "$0")" && source ./est_dir_common.sh
python3 eval_estimated_direction_cem.py "${COMMON_ARGS[@]}" \
  --eval_seed 0 --episodes 100 \
  --bias_list=-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5 \
  --methods oracle,estimated,residual_only,pos_only \
  --warmup_steps 5 \
  --save_json "$OUT_DIR/main.json" 2>&1 | tee "$LOG_DIR/main.log"
