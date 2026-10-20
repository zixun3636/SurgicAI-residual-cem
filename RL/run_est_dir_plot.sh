#!/bin/bash
# Step 4: make figures + print tables from whatever results exist.
cd "$(dirname "$0")" && source ./est_dir_common.sh
ARGS=(--out_dir "$RL_DIR/figures/estimated_direction")
[ -f "$OUT_DIR/main.json" ] && ARGS+=(--main "$OUT_DIR/main.json")
[ -d "$OUT_DIR/noise" ] && ARGS+=(--noise_dir "$OUT_DIR/noise")
[ -f "$OUT_DIR/random3d.json" ] && ARGS+=(--random3d "$OUT_DIR/random3d.json")
python3 plot_estimated_direction.py "${ARGS[@]}"
