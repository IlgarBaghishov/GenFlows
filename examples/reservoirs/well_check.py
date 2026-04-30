"""Well visualization on real high-NTG lobe-class HF cubes.

CONVENTIONS (corrected to match examples/lobes/inpainting):
  - cube tensor is (1, X=64, Y=64, Z=32). z is the LAST axis.
  - z = ELEVATION (z=0 is the deepest paleo-layer, z=Z-1 is the SURFACE).
    Plots use origin='lower' so z=0 lands at the BOTTOM of the image,
    matching `genflows/utils/plotting_lobes.py`.
  - A well drills from the surface (z=Z-1) DOWN to lower z. The direction
    vector therefore has d_z = -cos(theta) <= 0. Backward-extension from a
    sampled midpoint hits the +z face (top) for steep wells and a side face
    for near-horizontal wells, exactly matching real entry geometry.

Sampler matches agreed spec:
  theta ~ U(0, 90°), phi ~ U(0, 360°), midpoint M ~ U(cube), straight line,
  voxelized 0.5-vox along entry-exit boundary segment, n_wells ~ U(1, 5),
  non-overlap retry up to 100×, with truncation: 50% boundary-to-boundary,
  50% stop mid-cube (entry on a face, exit at random TD inside the cube).
"""
import os, random
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from genflows.utils.data_reservoirs import ReservoirDataset

OUT = os.path.join(os.path.dirname(__file__), 'well_check.png')
GREY_YELLOW = LinearSegmentedColormap.from_list('grey_yellow', ['#5e5e5e', '#f6c542'])


# ---- Straight-line well sampler (z = LAST axis, drilling DOWN i.e. dz<=0) -
def _ray_box_t_max(origin, direction, shape):
    t_max = np.inf
    for i in range(3):
        if direction[i] > 1e-12:
            t = (shape[i] - origin[i]) / direction[i]
        elif direction[i] < -1e-12:
            t = (0.0 - origin[i]) / direction[i]
        else:
            continue
        if 0 <= t < t_max:
            t_max = t
    return t_max


def sample_one_well(volume_shape, occupied, max_retries=100, p_through=0.5):
    """volume_shape = (X, Y, Z); z is the LAST axis; surface is at z=Z-1.

    Well drills from z=Z-1 (surface) toward z=0 — direction has d_z <= 0.
    """
    X, Y, Z = volume_shape
    shape = np.array([X, Y, Z], dtype=np.float64)
    for _ in range(max_retries):
        theta = np.random.uniform(0.0, np.pi / 2.0)   # inclination from vertical
        phi   = np.random.uniform(0.0, 2.0 * np.pi)   # azimuth in x-y plane
        d = np.array([
            np.sin(theta) * np.cos(phi),  # dx
            np.sin(theta) * np.sin(phi),  # dy
            -np.cos(theta),               # dz  <= 0  (drilling DOWN)
        ])
        m = np.array([
            np.random.uniform(0.0, X),
            np.random.uniform(0.0, Y),
            np.random.uniform(0.0, Z),
        ])
        t_fwd  = _ray_box_t_max(m,  d, shape)
        t_back = _ray_box_t_max(m, -d, shape)
        if t_fwd <= 0 or t_back <= 0:
            continue
        if np.random.random() >= p_through:
            t_fwd = np.random.uniform(0.0, t_fwd)  # truncate at random TD
        n = max(2, int(np.ceil((t_fwd + t_back) * 2.0)) + 1)
        ts = np.linspace(-t_back, t_fwd, n)
        pts = m[None, :] + ts[:, None] * d[None, :]
        idx = pts.astype(int)
        valid = ((idx[:, 0] >= 0) & (idx[:, 0] < X)
                 & (idx[:, 1] >= 0) & (idx[:, 1] < Y)
                 & (idx[:, 2] >= 0) & (idx[:, 2] < Z))
        idx = idx[valid]
        voxels = {tuple(v) for v in idx}
        if voxels & occupied:
            continue
        return voxels, (theta, phi, t_fwd + t_back)
    return None, None


def sample_wells(volume_shape, n_wells):
    occupied = set()
    metas, voxels_list = [], []
    for _ in range(n_wells):
        voxels, meta = sample_one_well(volume_shape, occupied)
        if voxels is None:
            continue
        voxels_list.append(voxels); metas.append(meta)
        occupied |= voxels
    mask = np.zeros(volume_shape, dtype=np.float32)
    for voxels in voxels_list:
        for x, y, z in voxels:
            mask[x, y, z] = 1.0
    return mask, metas, voxels_list


# ---- Pick high-NTG lobes and render ------------------------------------
ds = ReservoirDataset('/scratch/11316/rustamzade17/SiliciclasticReservoirs', split='train')
lobe_pos_all = np.where(ds.layer_idx == 0)[0]
lobe_ntg = ds.cont_raw[lobe_pos_all, 0]
candidates = lobe_pos_all[lobe_ntg >= 0.50]
print(f'lobe samples with ntg >= 0.50: {len(candidates)}')

SEEDS = [3, 27, 73, 101]
N_CASES = len(SEEDS)
X, Y, Z = 64, 64, 32

fig = plt.figure(figsize=(20, 4.4 * N_CASES))
for row, seed in enumerate(SEEDS):
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    sample_pos = int(candidates[row * 7 % len(candidates)])
    facies, _ = ds[sample_pos]
    fac = facies[0].numpy()  # (X, Y, Z)
    n_wells = int(np.random.randint(1, 6))
    mask, metas, voxels_list = sample_wells((X, Y, Z), n_wells)
    angles_str = '  '.join(f'(θ={np.rad2deg(t):.0f}°,φ={np.rad2deg(p):.0f}°,L={L:.0f})'
                            for t, p, L in metas)

    z_mid, y_mid, x_mid = Z // 2, Y // 2, X // 2

    # 2D slices (transpose so the LATERAL axis is the image x and z is the image y).
    fac_xz, mask_xz = fac[:, y_mid, :].T, mask[:, y_mid, :].T   # (Z, X)
    fac_yz, mask_yz = fac[x_mid, :, :].T, mask[x_mid, :, :].T   # (Z, Y)
    fac_xy, mask_xy = fac[:, :, z_mid].T, mask[:, :, z_mid].T   # (Y, X)

    # Max-projection of all well voxels onto each plane (light-red overlay).
    proj_xz = mask.max(axis=1).T   # collapse y -> (Z, X)
    proj_yz = mask.max(axis=0).T   # collapse x -> (Z, Y)
    proj_xy = mask.max(axis=2).T   # collapse z -> (Y, X)

    ax_xz = fig.add_subplot(N_CASES, 4, row * 4 + 1)
    ax_yz = fig.add_subplot(N_CASES, 4, row * 4 + 2)
    ax_xy = fig.add_subplot(N_CASES, 4, row * 4 + 3)
    ax_3d = fig.add_subplot(N_CASES, 4, row * 4 + 4, projection='3d')

    panels = [
        (ax_xz, fac_xz, mask_xz, proj_xz, f'XZ slice @ y={y_mid}', 'x', 'z (elev: surface up)', 'auto'),
        (ax_yz, fac_yz, mask_yz, proj_yz, f'YZ slice @ x={x_mid}', 'y', 'z (elev: surface up)', 'auto'),
        (ax_xy, fac_xy, mask_xy, proj_xy, f'XY slice @ z={z_mid}', 'x', 'y',                    'equal'),
    ]
    for ax, fac_2d, mask_2d, proj_2d, title, xl, yl, asp in panels:
        ax.imshow((fac_2d + 1) / 2, cmap=GREY_YELLOW, vmin=0, vmax=1,
                  aspect=asp, origin='lower')
        # Light-red max-projection of all wells (so we always see a silhouette).
        if proj_2d.sum() > 0:
            ax.imshow(np.where(proj_2d > 0, 1.0, np.nan), cmap='Reds', alpha=0.30,
                      aspect=asp, origin='lower', vmin=0, vmax=1.5)
        # Solid red — voxels that ACTUALLY pass through this slice.
        if mask_2d.sum() > 0:
            ax.imshow(np.where(mask_2d > 0, 1.0, np.nan), cmap='autumn', alpha=0.95,
                      aspect=asp, origin='lower')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xl); ax.set_ylabel(yl)

    # 3D faithful view; per-well color from voxels_list.
    cmap_w = plt.get_cmap('tab10')
    for wi, voxels in enumerate(voxels_list):
        if not voxels:
            continue
        arr = np.array(list(voxels))  # (n, 3): (x, y, z)
        ax_3d.scatter(arr[:, 0], arr[:, 1], arr[:, 2],
                      c=[cmap_w(wi % 10)], s=12, alpha=0.95, marker='s')
    # Cube wireframe (z plotted directly so surface z=Z-1 is at TOP of 3D box).
    for s, e in [
        ((0, 0, 0), (X-1, 0, 0)), ((0, 0, 0), (0, Y-1, 0)), ((0, 0, 0), (0, 0, Z-1)),
        ((X-1, Y-1, Z-1), (0, Y-1, Z-1)), ((X-1, Y-1, Z-1), (X-1, 0, Z-1)),
        ((X-1, Y-1, Z-1), (X-1, Y-1, 0)), ((X-1, 0, 0), (X-1, Y-1, 0)),
        ((X-1, 0, 0), (X-1, 0, Z-1)), ((0, Y-1, 0), (X-1, Y-1, 0)),
        ((0, Y-1, 0), (0, Y-1, Z-1)), ((0, 0, Z-1), (X-1, 0, Z-1)),
        ((0, 0, Z-1), (0, Y-1, Z-1)),
    ]:
        ax_3d.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]],
                   'k-', linewidth=0.6, alpha=0.5)
    ax_3d.set_xlabel('x'); ax_3d.set_ylabel('y'); ax_3d.set_zlabel('z (surface = top)')
    ax_3d.set_title(f'3D — {n_wells} wells (faithful)')
    ax_3d.set_xlim(0, X); ax_3d.set_ylim(0, Y); ax_3d.set_zlim(0, Z)
    ax_3d.view_init(elev=18, azim=-65)

    ax_xz.text(-0.30, 0.5,
               f'CASE {row}\nseed={seed}\nsample idx={sample_pos}\n'
               f'ntg={ds.cont_raw[sample_pos, 0]:.2f}\nn_wells={n_wells}\n{angles_str}',
               transform=ax_xz.transAxes, ha='right', va='center', fontsize=8)

fig.suptitle('Real high-NTG lobe cubes + straight-line wells (50% boundary-to-boundary, 50% truncated).  '
             '2D panels: facies (grey/yellow) + light-red well max-projection + '
             'solid-red true slice intersection.  3D panel: every well voxel.\n'
             'z is elevation (surface at top); wells drill from surface DOWN.',
             fontsize=11, y=1.001)
plt.tight_layout()
plt.savefig(OUT, dpi=110, bbox_inches='tight')
print('saved', OUT)
