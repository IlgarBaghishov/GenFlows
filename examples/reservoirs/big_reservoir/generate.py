"""Generate a multi-type big reservoir over an Option-A geological sequence.

Sequence (proximal -> distal -> basinal) along X:
    SH_PROXIMAL -> MEANDER_OXBOW -> CB_JIGSAW -> CB_LABYRINTH ->
    PV_SHOESTRING -> SH_DISTAL -> delta -> lobe

Y direction is 3 nearby azimuths (75 / 90 / 105 degrees).
Z is one block (32 deep).

Universal scalars (same in every block, from the global per-type intersection
of training ranges):
    NTG = 0.42, width_cells = 12, depth_cells = 7

Family scalars: per-block values picked to sit at the *pairwise* intersection
with each neighbour wherever such an intersection exists in the training
data, otherwise as close to the neighbour's range as the block's own range
allows. See the discussion in CLAUDE/PR notes for the full table.

Six runs are produced: 3 transition strategies x 2 overlaps.
"""
import os
import time

import numpy as np
import torch

from resflow.models.unet3d import UNet3D
from resflow.methods.flow_matching import FlowMatching
from resflow.assembly import (
    BlockSpec, COND_DIM, LAYER_TYPE_TO_IDX,
    expand_blockspecs_for_transition,
    generate_big_reservoir_multi, grid_layout_info,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CKPT = os.path.join(os.environ.get('SCRATCH', '.'), 'genflows_runs/reservoirs_inpainting/checkpoints/flow_matching.pt')
COND_STATS = os.path.join(os.environ.get('SCRATCH', '.'), 'genflows_runs/reservoirs_inpainting/checkpoints/cond_stats.npz')

OUT_DIR = os.environ.get('RESERVOIR_OUT_DIR', 'results')
LOBE_WIDTH_RAW = float(os.environ.get('RESERVOIR_LOBE_WIDTH', '12'))   # raw width_cells for lobe block; default = same as universal
SEED = 42
N_STEPS = 50
CFG_SCALE = 3.0

LAYER_SEQUENCE = [
    'channel:SH_PROXIMAL',
    'channel:MEANDER_OXBOW',
    'channel:CB_JIGSAW',
    'channel:CB_LABYRINTH',
    'channel:PV_SHOESTRING',
    'channel:SH_DISTAL',
    'delta',
    'lobe',
]
_default_az = [350.0, 0.0, 10.0]   # 3-row fan, central flow along +X (toward lobes)
_ny_override = int(os.environ.get('RESERVOIR_NY', '0'))   # 0 = keep default
if _ny_override == 1:
    ROW_AZIMUTHS_DEG = [0.0]   # one row, central azimuth
elif _ny_override > 1:
    ROW_AZIMUTHS_DEG = _default_az[:_ny_override] if _ny_override <= 3 else _default_az + [0.0] * (_ny_override - 3)
else:
    ROW_AZIMUTHS_DEG = _default_az

# Universal scalars (raw values; same in every conditional block).
UNIVERSAL_RAW = {'ntg': 0.42, 'width_cells': 12.0, 'depth_cells': 7.0}

# Family scalars per layer type (raw); 0/missing for any column not listed.
FAMILY_RAW = {
    'channel:SH_PROXIMAL':  {'mCHsinu': 1.40, 'mFFCHprop': 0.05, 'probAvulInside': 0.20},
    'channel:MEANDER_OXBOW': {'mCHsinu': 1.40, 'mFFCHprop': 0.00, 'probAvulInside': 0.00},
    'channel:CB_JIGSAW':    {'mCHsinu': 1.40, 'mFFCHprop': 0.10, 'probAvulInside': 0.20},
    'channel:CB_LABYRINTH': {'mCHsinu': 1.40, 'mFFCHprop': 0.10, 'probAvulInside': 0.10},
    'channel:PV_SHOESTRING': {'mCHsinu': 1.40, 'mFFCHprop': 0.10, 'probAvulInside': 0.10},
    'channel:SH_DISTAL':    {'mCHsinu': 1.25, 'mFFCHprop': 0.10, 'probAvulInside': 0.10},
    'delta':                {'mCHsinu': 1.10, 'mFFCHprop': 0.10, 'probAvulInside': 0.50,
                             'trunk_length_fraction': 0.10},
    'lobe':                 {'asp': 1.00},
}
# Optional per-block override of width_cells for the lobe block (raw value).
# When != the universal value, the lobe block receives a bigger feature scale.
if LOBE_WIDTH_RAW != 12.0:
    FAMILY_RAW['lobe']['width_cells'] = LOBE_WIDTH_RAW
    print(f"[override] lobe width_cells = {LOBE_WIDTH_RAW} (universal stays at 12)")

BLOCK_SHAPE = (64, 64, 32)
TRANSITION_MODES = ['hard', 'soft', 'soft_nobuffer', 'buffer']
OVERLAPS = [32, 24, 16]
SKIP_EXISTING = True   # don't re-do runs whose NPZ is already on disk

# Some blocks per row x 3 Y rows. Single A100 fits everything in one batch.
MAX_BATCH = 64


def build_grid_for_row(layer_seq, azimuth_deg):
    specs = []
    for name in layer_seq:
        idx = LAYER_TYPE_TO_IDX[name]
        scalars = dict(UNIVERSAL_RAW)
        scalars.update(FAMILY_RAW.get(name, {}))
        specs.append(BlockSpec(layer_idx=idx, azimuth_deg=azimuth_deg,
                               raw_scalars=scalars))
    return specs


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load training-time normalization stats
    stats = np.load(COND_STATS, allow_pickle=True)
    cont_min = stats['cont_min']
    cont_max = stats['cont_max']
    print(f"cont_min: {cont_min}\ncont_max: {cont_max}")

    # Load model
    print(f"\nLoading checkpoint: {CKPT}")
    model = UNet3D(in_channels=3, out_channels=1, num_cond=COND_DIM,
                   num_time_embs=1, expand_angle_idx=None).to(device)
    sd = torch.load(CKPT, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    method = FlowMatching(model)

    # Per-Y-row base specs (8 blocks each)
    base_specs_by_row = [build_grid_for_row(LAYER_SEQUENCE, az)
                         for az in ROW_AZIMUTHS_DEG]

    summary = []

    for mode in TRANSITION_MODES:
        # Expand each row according to transition strategy
        rows_expanded = [expand_blockspecs_for_transition(row, mode)
                         for row in base_specs_by_row]
        nx_eff = len(rows_expanded[0])
        ny = len(rows_expanded)
        for overlap in OVERLAPS:
            tag = f"{mode}_ov{overlap:02d}"
            out_npz = os.path.join(OUT_DIR, f"reservoir_{tag}.npz")
            if SKIP_EXISTING and os.path.exists(out_npz):
                print(f"\n[skip] {out_npz} already exists")
                continue
            print(f"\n{'='*70}")
            print(f"Run: mode={mode}  overlap={overlap}  grid={ny}x{nx_eff}")
            print(f"{'='*70}")

            torch.manual_seed(SEED)

            t0 = time.time()
            x_global, step_times = generate_big_reservoir_multi(
                method, rows_expanded, cont_min, cont_max,
                block_shape=BLOCK_SHAPE, overlap_xy=overlap,
                n_steps=N_STEPS, cfg_scale=CFG_SCALE,
                max_batch=MAX_BATCH, device=device,
            )
            elapsed = time.time() - t0

            (Tx, Ty, Sz), boxes = grid_layout_info(
                rows_expanded, BLOCK_SHAPE, overlap_xy=overlap,
            )
            volume = x_global.numpy()  # float, in [-1, 1] approx
            binary = (volume > 0).astype(np.int8)

            # Track which X-blocks are pure layer types vs transitions for
            # the visualizer. Pure positions are:
            #   - "hard"   : every j in range(8)
            #   - "soft"/"buffer" : even j (0, 2, 4, ..., 14)
            if mode in ('hard', 'soft_nobuffer'):
                pure_x_indices = list(range(nx_eff))
                pure_x_layer = LAYER_SEQUENCE
                trans_x_indices = []
                trans_kind = mode
            else:
                pure_x_indices = list(range(0, nx_eff, 2))
                pure_x_layer = LAYER_SEQUENCE
                trans_x_indices = list(range(1, nx_eff, 2))
                trans_kind = mode

            np.savez(out_npz,
                     volume=volume.astype(np.float32),
                     binary=binary,
                     boxes=np.array([(k, *v) for k, v in
                                     [((i, j), boxes[(i, j)])
                                      for i in range(ny) for j in range(nx_eff)]],
                                    dtype=object),
                     mode=mode,
                     overlap=overlap,
                     block_shape=np.array(BLOCK_SHAPE),
                     ny=ny, nx=nx_eff,
                     pure_x_indices=np.array(pure_x_indices, dtype=np.int32),
                     pure_x_layer=np.array(pure_x_layer, dtype=object),
                     trans_x_indices=np.array(trans_x_indices, dtype=np.int32),
                     trans_kind=trans_kind,
                     row_azimuths_deg=np.array(ROW_AZIMUTHS_DEG, dtype=np.float32),
                     elapsed_s=elapsed,
                     )
            print(f"  saved -> {out_npz}")
            print(f"  volume shape: {volume.shape}  NTG: {binary.mean():.3f}  "
                  f"elapsed: {elapsed:.1f}s")
            summary.append((mode, overlap, volume.shape, float(binary.mean()),
                            elapsed))

    # Summary
    print("\n" + "=" * 70)
    print("Summary of runs")
    print("=" * 70)
    print(f"{'mode':10s} {'ovrlp':6s} {'shape':18s} {'NTG':>6s} {'time':>8s}")
    for mode, overlap, shape, ntg, t in summary:
        print(f"{mode:10s} {overlap:6d} {str(shape):18s} {ntg:6.3f} {t:7.1f}s")


if __name__ == '__main__':
    main()
