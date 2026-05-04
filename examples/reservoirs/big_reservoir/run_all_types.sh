#!/bin/bash
# Run the full pipeline (generate + visualize + 50-realization entropy on
# 3 GPUs) for every channel-type subdir.  pv_shoestring already has the
# no-wells reservoirs from the sanity check, so we just regenerate to keep
# things uniform.

set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
TYPES=(pv_shoestring meander_oxbow cb_jigsaw cb_labyrinth sh_distal sh_proximal)

for t in "${TYPES[@]}"; do
    echo ""
    echo "############################################################"
    echo "# $t  ($(date))"
    echo "############################################################"
    cd "$HERE/$t"
    # Skip generate if it already produced the npz files (pv_shoestring case)
    if [ ! -f results/reservoir_hard_ov24.npz ]; then
        echo "[generate.py] $t"
        python generate.py
    else
        echo "[generate.py] $t — skipped (npz already exists)"
    fi
    echo "[visualize.py] $t"
    python visualize.py
    echo "[wells_entropy.py] $t (50 reals across 3 GPUs)"
    python wells_entropy.py --overlap 16 --n-real 50
done
echo ""
echo "############################################################"
echo "# ALL TYPES DONE  ($(date))"
echo "############################################################"
