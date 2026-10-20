#!/bin/bash
# Step 2: estimator under pose-reading noise; how many warm-up steps are needed?
# Only the estimated method; smallest (+-0.5) and largest (+-2) biases; 50 episodes.
cd "$(dirname "$0")" && source ./est_dir_common.sh
mkdir -p "$OUT_DIR/noise"
for noise in 0.5 1.0; do
  for warm in 5 10 20; do
    echo "===== est_noise_deg=$noise warmup=$warm ====="
    python3 eval_estimated_direction_cem.py "${COMMON_ARGS[@]}" \
      --eval_seed 0 --episodes 50 --bias_list=-2,-0.5,0.5,2 \
      --methods estimated --est_noise_deg $noise --warmup_steps $warm \
      --save_json "$OUT_DIR/noise/noise_${noise}_warm_${warm}.json" \
      2>&1 | tee "$LOG_DIR/noise_${noise}_warm_${warm}.log"
  done
done
