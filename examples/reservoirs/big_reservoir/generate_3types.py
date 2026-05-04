"""Generate a 5-block, 3-type fluvial-deltaic reservoir.

Sequence along X (1 Y row, no Y overlap):
    PV_SHOESTRING (x2)  ->  delta  ->  lobe (x2)

PV_SHOESTRING ("shoestring" sands) are narrow, near-straight, ribbon-like
channel-form sand bodies -- this is the geologically correct class for a
straight, continuous channel that feeds a delta and downstream lobes.
MEANDER_OXBOW was tried first but always produced oxbow-cutoff "knots"
(circular blobs that interrupt the channel); those are a defining feature
of the meander-oxbow class encoded in the layer-type one-hot, so no value
of mCHsinu can suppress them.

Per-block parameters chosen for visual contrast while staying inside each
type's training range:

    shoestring: width_cells=8   (narrow ribbons)
                NTG=0.15        (very sand-poor; thin, well-isolated ribbons
                                 against a mud-dominated background)
                mCHsinu=1.10    (low sinuosity -- in-distribution for
                                 shoestring sands, which are naturally
                                 straight; not a meander class so this
                                 does not produce oxbow blobs)
    delta:    width_cells=8   (narrow distributaries that visibly bifurcate
                               rather than merging into a broad sand sheet)
              NTG=0.30        (lower; distinct distributaries against mud)
              mCHsinu=1.10, mFFCHprop=0.20, probAvulInside=0.60,
              trunk_length_fraction=0.30
    lobe:     width_cells=25  (big sand bodies, requested)
              NTG=0.62        (very sand-rich basinal lobe complex)
              asp=1.75        (mid of [1.0, 2.5])

depth_cells = 7 in every block. azimuth = 0 deg (flow toward +X).
Only hard transition mode; overlaps 16 and 12.
"""
import os
import time
import numpy as np
import torch

from resflow.models.unet3d import UNet3D
from resflow.methods.flow_matching import FlowMatching
from resflow.assembly import (
    BlockSpec, COND_DIM, LAYER_TYPE_TO_IDX,
    generate_big_reservoir_multi, grid_layout_info,
)


CKPT = os.path.join(os.environ.get('SCRATCH', '.'), 'genflows_runs/reservoirs_inpainting/checkpoints/flow_matching.pt')
COND_STATS = os.path.join(os.environ.get('SCRATCH', '.'), 'genflows_runs/reservoirs_inpainting/checkpoints/cond_stats.npz')
OUT_DIR = os.environ.get('RESERVOIR_OUT_DIR', 'results_3types')
SEED = 42
N_STEPS = 50
CFG_SCALE = 3.0
BLOCK_SHAPE = (64, 64, 32)
OVERLAPS = [16, 12]
AZIMUTH_DEG = 0.0

DEPTH_CELLS = float(os.environ.get('RESERVOIR_DEPTH_CELLS', '10.0'))
DELTA_MFFCHPROP = float(os.environ.get('RESERVOIR_DELTA_MFFCHPROP', '0.40'))
SHOE_MFFCHPROP = float(os.environ.get('RESERVOIR_SHOE_MFFCHPROP', '0.0'))

BLOCK_DEFS = [
    ('channel:PV_SHOESTRING',
     {'ntg': 0.18, 'width_cells': 18.0, 'depth_cells': DEPTH_CELLS,
      'mCHsinu': 1.10, 'mFFCHprop': SHOE_MFFCHPROP, 'probAvulInside': 0.0}),
    ('channel:PV_SHOESTRING',
     {'ntg': 0.18, 'width_cells': 18.0, 'depth_cells': DEPTH_CELLS,
      'mCHsinu': 1.10, 'mFFCHprop': SHOE_MFFCHPROP, 'probAvulInside': 0.0}),
    ('delta',
     {'ntg': 0.40, 'width_cells': 7.0, 'depth_cells': DEPTH_CELLS,
      'mCHsinu': 1.30, 'mFFCHprop': DELTA_MFFCHPROP,
      'probAvulInside': 1.0, 'trunk_length_fraction': 0.0}),
    ('lobe',
     {'ntg': 0.62, 'width_cells': 25.0, 'depth_cells': DEPTH_CELLS,
      'asp': 1.75}),
    ('lobe',
     {'ntg': 0.62, 'width_cells': 25.0, 'depth_cells': DEPTH_CELLS,
      'asp': 1.75}),
]
LAYER_SEQUENCE = [name for name, _ in BLOCK_DEFS]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Device: {device}  out: {OUT_DIR}")

    stats = np.load(COND_STATS, allow_pickle=True)
    cont_min, cont_max = stats['cont_min'], stats['cont_max']

    model = UNet3D(in_channels=3, out_channels=1, num_cond=COND_DIM,
                   num_time_embs=1, expand_angle_idx=None).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    model.eval()
    method = FlowMatching(model)

    specs_row = [
        BlockSpec(layer_idx=LAYER_TYPE_TO_IDX[name],
                  azimuth_deg=AZIMUTH_DEG, raw_scalars=dict(scalars))
        for name, scalars in BLOCK_DEFS
    ]
    grid = [specs_row]
    ny, nx = 1, len(specs_row)

    for overlap in OVERLAPS:
        tag = f"hard_ov{overlap:02d}"
        print(f"\n== {tag} == grid={ny}x{nx}")
        torch.manual_seed(SEED)
        t0 = time.time()
        x_global, _ = generate_big_reservoir_multi(
            method, grid, cont_min, cont_max,
            block_shape=BLOCK_SHAPE, overlap_xy=overlap,
            n_steps=N_STEPS, cfg_scale=CFG_SCALE,
            max_batch=64, device=device,
        )
        elapsed = time.time() - t0
        volume = x_global.numpy()
        binary = (volume > 0).astype(np.int8)

        out_npz = os.path.join(OUT_DIR, f"reservoir_{tag}.npz")
        np.savez(out_npz,
                 volume=volume.astype(np.float32),
                 binary=binary,
                 mode='hard',
                 overlap=overlap,
                 block_shape=np.array(BLOCK_SHAPE),
                 ny=ny, nx=nx,
                 pure_x_indices=np.array(list(range(nx)), dtype=np.int32),
                 pure_x_layer=np.array(LAYER_SEQUENCE, dtype=object),
                 trans_x_indices=np.array([], dtype=np.int32),
                 trans_kind='hard',
                 row_azimuths_deg=np.array([AZIMUTH_DEG], dtype=np.float32),
                 elapsed_s=elapsed)
        print(f"  saved -> {out_npz}")
        print(f"  shape={binary.shape}  NTG={binary.mean():.3f}  {elapsed:.1f}s")


if __name__ == '__main__':
    main()
