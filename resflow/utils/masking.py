"""Wells-only mask generation + InpaintDataset wrapper for 3D inpainting.

Conventions (used uniformly by both lobes and reservoirs after the
2026-04-29 fix):
  - cube tensor is (C, X, Y, Z); z is the LAST spatial axis. Matches the
    custom geological package's on-disk (nx, ny, nz) layout.
  - z is ELEVATION: z=0 is the paleo-floor, z=Z-1 is the depositional
    surface. A well drills from the surface (z=Z-1) DOWN to lower z, so
    its direction has d_z <= 0.

Mask convention: 1 = known (keep), 0 = unknown (generate).
"""
import random
import numpy as np
import torch
from torch.utils.data import Dataset


# -- straight-line well sampler ------------------------------------------
def _ray_box_t_max(origin, direction, shape):
    """Largest t >= 0 such that origin + t*direction stays in [0, shape]."""
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


def sample_one_well(volume_shape, occupied=None, max_retries=100, p_through=0.5):
    """Sample one straight-line well as a set of voxel tuples (x, y, z).

    volume_shape : (X, Y, Z) — z is the LAST axis; z=Z-1 is the surface.
    occupied     : set of already-claimed (x, y, z) tuples for non-overlap.
    p_through    : probability the well goes boundary-to-boundary;
                   otherwise it stops mid-cube at random TD.

    Returns (voxels, (theta, phi, length)) or (None, None) on failure.
    """
    if occupied is None:
        occupied = set()
    X, Y, Z = volume_shape
    shape = np.array([X, Y, Z], dtype=np.float64)
    for _ in range(max_retries):
        # NOTE: stdlib `random` (auto-seeded per PyTorch DataLoader worker),
        # NOT np.random — numpy state is forked identically across workers.
        theta = random.uniform(0.0, np.pi / 2.0)
        phi   = random.uniform(0.0, 2.0 * np.pi)
        d = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            -np.cos(theta),                # d_z <= 0 — drilling DOWN
        ])
        m = np.array([
            random.uniform(0.0, X),
            random.uniform(0.0, Y),
            random.uniform(0.0, Z),
        ])
        t_fwd  = _ray_box_t_max(m,  d, shape)
        t_back = _ray_box_t_max(m, -d, shape)
        if t_fwd <= 0 or t_back <= 0:
            continue
        if random.random() >= p_through:
            t_fwd = random.uniform(0.0, t_fwd)
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


def generate_well_mask(volume_shape, n_wells=None, max_wells=5, p_through=0.5):
    """Mask with multiple non-intersecting straight-line wells.

    Returns: (1, X, Y, Z) float tensor, 1 = known voxel.
    """
    X, Y, Z = volume_shape
    mask = torch.zeros(1, X, Y, Z)
    if n_wells is None:
        n_wells = random.randint(1, max_wells)
    occupied = set()
    for _ in range(n_wells):
        voxels, _ = sample_one_well(volume_shape, occupied, p_through=p_through)
        if voxels is None:
            continue
        for x, y, z in voxels:
            mask[0, x, y, z] = 1.0
        occupied |= voxels
    return mask


def generate_training_mask(volume_shape, uncond_prob=0.30, max_wells=5,
                           p_through=0.5):
    """Wells-only training mask:
        uncond_prob (=0.30) : empty mask (unconditional generation)
        else                : 1..max_wells straight-line wells
                              (50% boundary-to-boundary, 50% truncated)
    """
    if random.random() < uncond_prob:
        return torch.zeros(1, *volume_shape)
    return generate_well_mask(volume_shape, max_wells=max_wells, p_through=p_through)


def apply_inpaint_output(samples, mask, known_data):
    """Hard-replace known voxels in generated samples.

    Args:
        samples    : (B, 1, X, Y, Z) generated output
        mask       : (B, 1, X, Y, Z) binary, 1=known
        known_data : (B, 1, X, Y, Z) clean values where mask=1
    """
    return samples * (1 - mask) + known_data * mask


# -- dataset wrapper -----------------------------------------------------
class InpaintDataset(Dataset):
    """Wraps a base dataset (yielding (facies, cond)) and adds an on-the-fly
    training mask via generate_training_mask().

    base_dataset[i] -> (facies, cond) where facies has shape (1, X, Y, Z).
    """

    def __init__(self, base_dataset, volume_shape):
        self.base_dataset = base_dataset
        self.volume_shape = tuple(volume_shape)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        facies, cond = self.base_dataset[idx]
        mask = generate_training_mask(self.volume_shape)
        return facies, cond, mask
