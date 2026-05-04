#!/bin/bash
# Run no-wells generate + visualize for every type subdir.  No wells, no
# entropy.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
TYPES=(lobes pv_shoestring meander_oxbow cb_jigsaw cb_labyrinth sh_distal sh_proximal)
for t in "${TYPES[@]}"; do
    echo ""
    echo "############################################################"
    echo "# $t  ($(date))"
    echo "############################################################"
    cd "$HERE/$t"
    rm -rf results figures
    python generate.py
    python visualize.py
done
echo ""
echo "############################################################"
echo "# DONE  ($(date))"
echo "############################################################"
