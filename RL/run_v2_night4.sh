#!/bin/bash
# Run 4 (~6-7 h): final pipeline = adaptive estimator with the COMMANDED-action sample filter
#  [A] bias flips at step 20, pose noise 0.5 / 1.0 deg, window 10 / 20
#  [B] final table: oracle vs adaptive(commanded), 11 scalar biases, pose noise 0.5 deg, 100 ep
cd "$(dirname "$0")" && source ./est_dir_common.sh
V2="$OUT_DIR/v2"; mkdir -p "$V2"

for noise in 0.5 1.0; do
  for win in 10 20; do
    echo "===== [A] flip, adaptive(commanded), noise=$noise, window=$win ====="
    python3 eval_estimated_direction_v2.py "${COMMON_ARGS[@]}" \
      --eval_seed 0 --episodes 100 --bias_list=-2,-1,1,2 --flip_step 20 \
      --methods adaptive --adaptive_filter commanded --window $win --est_noise_deg $noise \
      --save_json "$V2/exp3b_flip_cmd_noise${noise}_win${win}.json" \
      2>&1 | tee "$LOG_DIR/v2_exp3b_flip_cmd_noise${noise}_win${win}.log"
  done
done

echo "===== [B] final table: oracle vs adaptive(commanded), noise 0.5 ====="
python3 eval_estimated_direction_v2.py "${COMMON_ARGS[@]}" \
  --eval_seed 0 --episodes 100 --bias_list=-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5 \
  --methods oracle,adaptive --adaptive_filter commanded --window 10 --est_noise_deg 0.5 \
  --save_json "$V2/final_table_noise0.5.json" 2>&1 | tee "$LOG_DIR/v2_final_table_noise0.5.log"
echo "===== run 4 done ====="
