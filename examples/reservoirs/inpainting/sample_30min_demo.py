"""Sample from the 30-min trained inpaint model + plot 3 cross-sections.

Per layer type (8): one unconditional sample (empty mask) + one with 5
fully-vertical wells in a "+" pattern. The well placement is chosen so
that:
  - center well at (cx, cy)
  - two more along x=cx     -> 3 wells visible in YZ slice at x=cx
  - two more along y=cy     -> 3 wells visible in XZ slice at y=cy
  - all 5 visible in any XY slice as the same "+" of dots

For the well "log" values we mask a real ground-truth cube of the same
layer type from the test split, so the model is asked to fill in the
rest of the cube consistent with rock that actually exists at those
voxels.

Run from this script's directory:  python sample_30min_demo.py
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch

from resflow.models.unet3d import UNet3D
from resflow.methods.flow_matching import FlowMatching
from resflow.utils.data_reservoirs import (
    ReservoirDataset, COND_DIM, LAYER_TYPES, VOLUME_SHAPE,
    NUM_LAYERS, UNIVERSAL_CONT, FAMILY_CONT,
)
from resflow.utils.masking import apply_inpaint_output


CKPT = 'checkpoints_30min/flow_matching.pt'
COND_STATS = 'checkpoints_30min/cond_stats.npz'
OUT_DIR = 'results_30min/samples'
N_STEPS = 50
CFG = 3.0

custom_cmap = LinearSegmentedColormap.from_list('grey_yellow', ['grey', 'yellow'])


def make_5well_mask(volume_shape):
    """5 fully-vertical wells in a "+" pattern. Returns (mask (1,X,Y,Z), centers list)."""
    X, Y, Z = volume_shape
    cx, cy = X // 2, Y // 2
    dx, dy = X // 4, Y // 4
    centers = [(cx, cy),
               (cx, cy - dy), (cx, cy + dy),
               (cx - dx, cy), (cx + dx, cy)]
    mask = torch.zeros(1, X, Y, Z)
    for x, y in centers:
        mask[0, x, y, :] = 1.0
    return mask, centers


def build_cond(layer_idx):
    """Cond vector for the given layer type: one-hot + mid-range scalars + azimuth=0."""
    onehot = np.zeros(NUM_LAYERS, dtype=np.float32)
    onehot[layer_idx] = 1.0
    cont_universal = np.full(len(UNIVERSAL_CONT), 0.5, dtype=np.float32)
    cont_family = np.zeros(len(FAMILY_CONT), dtype=np.float32)
    az = np.array([0.0, 1.0], dtype=np.float32)  # sin(0), cos(0)
    return np.concatenate([onehot, cont_universal, az, cont_family])


def find_real_cube(dataset, layer_idx, max_scan=5000):
    """Pull one cube of the given layer_idx from `dataset`."""
    hits = np.where(dataset.layer_idx == layer_idx)[0]
    if len(hits) == 0:
        raise RuntimeError(f"No cube with layer_idx={layer_idx} in this split")
    facies, cond = dataset[int(hits[0])]
    return facies, cond


def sample_two(method, cond_vec, real_cube, well_mask, device):
    """Returns (s_uncond, s_well) tensors of shape (X,Y,Z) in [-1,1]."""
    raw = method.model
    cond = cond_vec.to(device).unsqueeze(0)              # (1, COND_DIM)
    mask = well_mask.to(device).unsqueeze(0)             # (1, 1, X, Y, Z)
    real = real_cube.to(device).unsqueeze(0)             # (1, 1, X, Y, Z)

    # Unconditional: empty mask (model still gets the 3-channel input)
    raw.set_inpaint_context(torch.zeros_like(mask), torch.zeros_like(real))
    torch.manual_seed(0)
    s_uncond = method.sample((1, 1, *VOLUME_SHAPE), device,
                             cond=cond, cfg_scale=CFG, n_steps=N_STEPS)

    # Wells: known voxels = real cube * mask; hard-paste them at the end
    known = real * mask
    raw.set_inpaint_context(mask, known)
    torch.manual_seed(0)
    s_well = method.sample((1, 1, *VOLUME_SHAPE), device,
                           cond=cond, cfg_scale=CFG, n_steps=N_STEPS)
    s_well = apply_inpaint_output(s_well, mask, known)
    raw.clear_inpaint_context()

    return s_uncond[0, 0].cpu(), s_well[0, 0].cpu()


def plot_three(ax_row, cube, row_label, well_xy=None):
    """Plot XY (mid-z), XZ (mid-y), YZ (mid-x) of `cube` (X, Y, Z) into ax_row.

    well_xy: optional list of (x, y) tuples — vertical wells; drawn as red lines.
    """
    cube_b = (cube > 0).numpy().astype(np.float32)
    X, Y, Z = cube_b.shape
    z_mid, y_mid, x_mid = Z // 2, Y // 2, X // 2

    # XY @ z=z_mid — transpose so y is on the vertical axis
    ax_row[0].imshow(cube_b[:, :, z_mid].T,
                     cmap=custom_cmap, vmin=0, vmax=1, origin='lower')
    ax_row[0].set_title(f'XY  (z={z_mid})', fontsize=10)
    ax_row[0].set_xlabel('X'); ax_row[0].set_ylabel('Y')

    # XZ @ y=y_mid — transpose so z is on the vertical axis
    ax_row[1].imshow(cube_b[:, y_mid, :].T,
                     cmap=custom_cmap, vmin=0, vmax=1, origin='lower')
    ax_row[1].set_title(f'XZ  (y={y_mid})', fontsize=10)
    ax_row[1].set_xlabel('X'); ax_row[1].set_ylabel('Z')

    # YZ @ x=x_mid
    ax_row[2].imshow(cube_b[x_mid, :, :].T,
                     cmap=custom_cmap, vmin=0, vmax=1, origin='lower')
    ax_row[2].set_title(f'YZ  (x={x_mid})', fontsize=10)
    ax_row[2].set_xlabel('Y'); ax_row[2].set_ylabel('Z')

    if well_xy is not None:
        xs_in_xz = sorted({x for (x, y) in well_xy if y == y_mid})
        ys_in_yz = sorted({y for (x, y) in well_xy if x == x_mid})
        for x in xs_in_xz:
            ax_row[1].axvline(x, color='red', linewidth=1.0, alpha=0.85)
        for y in ys_in_yz:
            ax_row[2].axvline(y, color='red', linewidth=1.0, alpha=0.85)
        wx = np.array([x for (x, _) in well_xy])
        wy = np.array([y for (_, y) in well_xy])
        ax_row[0].scatter(wx, wy, s=18, c='red', marker='x', linewidths=1.5)

    ax_row[0].text(-0.22, 0.5, row_label, transform=ax_row[0].transAxes,
                   ha='right', va='center', rotation=90, fontsize=11,
                   fontweight='bold')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_dir = os.environ.get(
        'RESERVOIR_DATA_DIR',
        os.path.join(os.environ['SCRATCH'], 'SiliciclasticReservoirs'),
    )
    stats = np.load(COND_STATS, allow_pickle=True)
    cont_min, cont_max = stats['cont_min'], stats['cont_max']

    print("Opening test split for ground-truth wells...")
    test_set = ReservoirDataset(data_dir, split='test',
                                cont_min=cont_min, cont_max=cont_max)

    model = UNet3D(in_channels=3, out_channels=1, num_cond=COND_DIM,
                   num_time_embs=1, expand_angle_idx=None).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    model.eval()
    method = FlowMatching(model)

    os.makedirs(OUT_DIR, exist_ok=True)
    well_mask, centers = make_5well_mask(VOLUME_SHAPE)
    print(f"Wells (x, y): {centers}")

    for li, layer_name in enumerate(LAYER_TYPES):
        print(f"\n[{li+1}/{len(LAYER_TYPES)}] {layer_name}")
        try:
            real_cube, _ = find_real_cube(test_set, li)
        except RuntimeError as e:
            print(f"  skip: {e}")
            continue

        cond_vec = torch.from_numpy(build_cond(li))
        with torch.no_grad():
            s_uncond, s_well = sample_two(method, cond_vec, real_cube,
                                          well_mask, device)

        fig, axes = plt.subplots(3, 3, figsize=(12, 11))
        plot_three(axes[0], real_cube[0], 'Real (test cube)')
        plot_three(axes[1], s_uncond, 'Uncond sample')
        plot_three(axes[2], s_well, 'Wells sample', well_xy=centers)
        fig.suptitle(layer_name, fontsize=14, fontweight='bold')
        plt.tight_layout()
        out_path = os.path.join(
            OUT_DIR, f'{li:02d}_{layer_name.replace(":", "-")}.png',
        )
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  saved {out_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
