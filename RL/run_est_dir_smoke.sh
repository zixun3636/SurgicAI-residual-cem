#!/bin/bash
# Step 0: quick check (about 5 minutes). 2 biases x 4 methods x 3 episodes.
cd "$(dirname "$0")" && source ./est_dir_common.sh
python3 eval_estimated_direction_cem.py "${COMMON_ARGS[@]}" \
  --eval_seed 0 --episodes 3 --bias_list=-2,2 \
  --methods oracle,estimated,residual_only,pos_only \
  --save_json "$OUT_DIR/smoke.json" 2>&1 | tee "$LOG_DIR/smoke.log"
