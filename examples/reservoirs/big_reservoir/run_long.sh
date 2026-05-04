#!/bin/bash
# Drive the LONG (1 x 10) channel-reservoir suite for all 6 channel families.
# Phase 1: generate.py (no-wells base) — 3 at a time across 3 GPUs.
# Phase 2: visualize.py (CPU figures) — sequential, fast.
# Phase 3: wells_entropy.py at overlap=16 with 30 reals — sequential per type
#          (each invocation internally fans out to 3 GPUs).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
TYPES=(pv_shoestring meander_oxbow cb_jigsaw cb_labyrinth sh_distal sh_proximal)

echo ""
echo "############################################################"
echo "# Phase 1: generate.py for all 6 long subdirs  ($(date))"
echo "############################################################"
# Run 3 at a time, one per GPU.
for batch_start in 0 3; do
    PIDS=()
    for offset in 0 1 2; do
        idx=$((batch_start + offset))
        t=${TYPES[$idx]}
        sub="${HERE}/${t}_long"
        log="${sub}/gen.log"
        mkdir -p "$sub"
        echo "  [GPU ${offset}] launching ${t}_long  -> ${log}"
        (
            cd "$sub"
            CUDA_VISIBLE_DEVICES=$offset python generate.py \
                > "$log" 2>&1
        ) &
        PIDS+=($!)
    done
    echo "  waiting on batch starting at ${batch_start}..."
    for pid in "${PIDS[@]}"; do
        wait "$pid"
    done
    echo "  batch starting at ${batch_start} done."
done

echo ""
echo "############################################################"
echo "# Phase 2: visualize.py for all 6 long subdirs  ($(date))"
echo "############################################################"
for t in "${TYPES[@]}"; do
    sub="${HERE}/${t}_long"
    echo "  visualize ${t}_long ..."
    (cd "$sub" && python visualize.py > viz.log 2>&1)
done

echo ""
echo "############################################################"
echo "# Phase 3: wells_entropy.py (overlap=16, n-real=30)  ($(date))"
echo "############################################################"
for t in "${TYPES[@]}"; do
    sub="${HERE}/${t}_long"
    echo ""
    echo "----  ${t}_long  ($(date))  ----"
    (cd "$sub" && python wells_entropy.py --overlap 16 --n-real 30)
done

echo ""
echo "############################################################"
echo "# DONE  ($(date))"
echo "############################################################"
