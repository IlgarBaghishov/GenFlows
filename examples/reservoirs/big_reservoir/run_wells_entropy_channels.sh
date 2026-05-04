#!/bin/bash
# Run wells_entropy.py for the 6 channel subdirs (skip lobes — done already
# with 50 reals; skip delta entirely).  30 realizations each, 3 GPUs in
# parallel via the worker subprocesses inside wells_entropy.py.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
TYPES=(pv_shoestring meander_oxbow cb_jigsaw cb_labyrinth sh_distal sh_proximal)
for t in "${TYPES[@]}"; do
    echo ""
    echo "############################################################"
    echo "# $t  (wells + 30-real entropy)  $(date)"
    echo "############################################################"
    cd "$HERE/$t"
    python wells_entropy.py --overlap 16 --n-real 30
done
echo ""
echo "############################################################"
echo "# DONE  ($(date))"
echo "############################################################"
