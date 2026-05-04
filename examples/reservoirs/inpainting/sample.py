"""Sample from the trained FM-inpaint reservoir model + plot 3x3 cross-sections.

For each of the 8 layer types:
  - randomly pick one real test cube of that type (seeded, reproducible)
  - sample one unconditional cube using that cube's own cond
  - extract 5 vertical wells in a "+" pattern from that real cube and
    sample one inpainted cube using those wells as the known voxels

Wells are placed so that:
  - XZ slice at y=cy shows 3 wells (center + ±X/4 along x at y=cy)
  - YZ slice at x=cx shows 3 wells (center + ±Y/4 along y at x=cx)
  - all 5 visible as dots in any XY slice

Plot uses the canonical lobe-inpainting palette (grey=shale, yellow=sand)
with a red translucent overlay + pixel-aligned red border on the well
voxels in the conditioned row.

Run from this script's directory:  python sample.py
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from resflow.models.unet3d import UNet3D
from resflow.methods.flow_matching import FlowMatching
from resflow.utils.data_reservoirs import (
    ReservoirDataset, COND_DIM, LAYER_TYPES, VOLUME_SHAPE,
)
from resflow.utils.masking import apply_inpaint_output
from resflow.utils.plotting_lobes import custom_cmap, draw_mask_boundary


DEFAULT_CKPT = os.path.join(os.environ.get('SCRATCH', '.'),
                            'genflows_runs/reservoirs_inpainting/checkpoints/inference_epoch040.pt')
DEFAULT_OUT = 'results/samples_epoch040'
N_STEPS = 50
CFG = 3.0
SEED = 0


def make_5well_mask(volume_shape):
    """5 fully-vertical wells in a "+" pattern. Returns (mask (1,X,Y,Z), centers)."""
    X, Y, Z = volume_shape
    cx, cy = X // 2, Y // 2
    dx, dy = X // 4, Y // 4
    centers = [(cx, cy),
               (cx, cy - dy), (cx, cy + dy),
               (cx - dx, cy), (cx + dx, cy)]
    mask = torch.zeros(1, X, Y, Z)
    for x, y in centers:
        mask[0, x, y, :] = 1.0
    return mask, centers, (cx, cy)


def pick_test_cube(dataset, layer_idx, rng):
    """Random test-set cube of the given layer type (seeded). Returns (facies, cond, idx)."""
    hits = np.where(dataset.layer_idx == layer_idx)[0]
    if len(hits) == 0:
        raise RuntimeError(f"No cube with layer_idx={layer_idx} in this split")
    pick = int(rng.choice(hits))
    facies, cond = dataset[pick]
    return facies, cond, pick


def slice_views(cube_3d, cx, cy):
    """Return three (Z-or-Y on vertical, origin='lower') slices of (X,Y,Z) cube.

    XY @ z=Z//2, XZ @ y=cy, YZ @ x=cx.  Each is .T-ed so imshow with
    origin='lower' puts (X|Y) on the horizontal and (Y|Z|Z) on the vertical.
    """
    X, Y, Z = cube_3d.shape
    return [
        (cube_3d[:, :, Z // 2].T, f'XY  (z={Z // 2})', 'X', 'Y'),
        (cube_3d[:, cy, :].T,     f'XZ  (y={cy})',     'X', 'Z'),
        (cube_3d[cx, :, :].T,     f'YZ  (x={cx})',     'Y', 'Z'),
    ]


def plot_row(ax_row, cube_3d, label, cx, cy, mask_3d=None):
    """Plot 3 slices of cube_3d into ax_row. If mask_3d given, overlay red on
    its known voxels in each slice (used only for the well-conditioned row)."""
    cube_b = (cube_3d > 0).astype(np.float32)

    for col, (img, title, xlabel, ylabel) in enumerate(slice_views(cube_b, cx, cy)):
        ax = ax_row[col]
        ax.imshow(img, cmap=custom_cmap, vmin=0, vmax=1, origin='lower')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if mask_3d is not None:
            # Same .T transpose so mask aligns with the cube slice
            mask_slice = slice_views(mask_3d, cx, cy)[col][0]
            red_overlay = np.zeros((*mask_slice.shape, 4))
            red_overlay[mask_slice > 0.5] = [1.0, 0.0, 0.0, 0.30]
            ax.imshow(red_overlay, origin='lower')
            draw_mask_boundary(ax, mask_slice, color='red', linewidth=1.5)

    ax_row[0].text(-0.22, 0.5, label, transform=ax_row[0].transAxes,
                   ha='right', va='center', rotation=90, fontsize=11,
                   fontweight='bold')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default=DEFAULT_CKPT)
    parser.add_argument('--out-dir', default=DEFAULT_OUT)
    parser.add_argument('--data-dir',
                        default=os.environ.get(
                            'RESERVOIR_DATA_DIR',
                            os.path.join(os.environ.get('SCRATCH', '.'),
                                         'SiliciclasticReservoirs')))
    parser.add_argument('--cond-stats',
                        default=os.path.join(
                            os.environ.get('SCRATCH', '.'),
                            'genflows_runs/reservoirs_inpainting/checkpoints/cond_stats.npz'))
    parser.add_argument('--n-steps', type=int, default=N_STEPS)
    parser.add_argument('--cfg', type=float, default=CFG)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device:   {device}")
    print(f"Ckpt:     {args.ckpt}")
    print(f"Data dir: {args.data_dir}")

    stats = np.load(args.cond_stats, allow_pickle=True)
    cont_min, cont_max = stats['cont_min'], stats['cont_max']

    print("Opening test split...")
    test_set = ReservoirDataset(args.data_dir, split='test',
                                cont_min=cont_min, cont_max=cont_max)

    model = UNet3D(in_channels=3, out_channels=1, num_cond=COND_DIM,
                   num_time_embs=1, expand_angle_idx=None).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    method = FlowMatching(model)

    os.makedirs(args.out_dir, exist_ok=True)
    well_mask, centers, (cx, cy) = make_5well_mask(VOLUME_SHAPE)
    print(f"Wells (x, y): {centers}  cx={cx} cy={cy}")

    rng = np.random.default_rng(args.seed)

    for li, layer_name in enumerate(LAYER_TYPES):
        print(f"\n[{li+1}/{len(LAYER_TYPES)}] {layer_name}")
        try:
            real_cube, real_cond, real_idx = pick_test_cube(test_set, li, rng)
        except RuntimeError as e:
            print(f"  skip: {e}")
            continue
        print(f"  test idx: {real_idx}")

        cond = real_cond.to(device).unsqueeze(0)              # (1, COND_DIM)
        mask = well_mask.to(device).unsqueeze(0)              # (1, 1, X, Y, Z)
        real = real_cube.to(device).unsqueeze(0)              # (1, 1, X, Y, Z)

        with torch.no_grad():
            # --- Unconditional sample (empty mask, real cube's cond) ---
            model.set_inpaint_context(torch.zeros_like(mask), torch.zeros_like(real))
            torch.manual_seed(args.seed)
            s_uncond = method.sample((1, 1, *VOLUME_SHAPE), device,
                                     cond=cond, cfg_scale=args.cfg, n_steps=args.n_steps)

            # --- Well-conditioned sample (5 wells from the real cube) ---
            known = real * mask
            model.set_inpaint_context(mask, known)
            torch.manual_seed(args.seed)
            s_well = method.sample((1, 1, *VOLUME_SHAPE), device,
                                   cond=cond, cfg_scale=args.cfg, n_steps=args.n_steps)
            s_well = apply_inpaint_output(s_well, mask, known)
            model.clear_inpaint_context()

        real_np = real_cube[0].numpy()
        uncond_np = s_uncond[0, 0].cpu().numpy()
        well_np = s_well[0, 0].cpu().numpy()
        mask_np = well_mask[0].numpy()

        fig, axes = plt.subplots(3, 3, figsize=(12, 11))
        fig.suptitle(f'{layer_name}  —  test idx {real_idx}',
                     fontsize=14, fontweight='bold')
        plot_row(axes[0], real_np,   'Real (test cube)', cx, cy)
        plot_row(axes[1], uncond_np, 'Uncond sample',    cx, cy)
        plot_row(axes[2], well_np,   'Wells sample',     cx, cy, mask_3d=mask_np)
        plt.tight_layout()
        out_path = os.path.join(
            args.out_dir, f'{li:02d}_{layer_name.replace(":", "-")}.png',
        )
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  saved {out_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
