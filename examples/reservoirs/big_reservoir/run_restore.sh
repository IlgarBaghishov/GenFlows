#!/bin/bash
# Full restore: regenerate lobes (gradient version) with all 3 plot suites
# (no-wells + with-wells + 50-realization entropy), then run no-wells +
# visualize for the 4 remaining channel types.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "############################################################"
echo "# lobes  (full restore: gen + viz + 50-real entropy)"
echo "# $(date)"
echo "############################################################"
cd "$HERE/lobes"
rm -rf results figures
python generate.py
python visualize.py
python wells_entropy.py --overlap 16 --n-real 50

for t in cb_jigsaw cb_labyrinth sh_distal sh_proximal; do
    echo ""
    echo "############################################################"
    echo "# $t  (uniform median, gen + viz only)"
    echo "# $(date)"
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
