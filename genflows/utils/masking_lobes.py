"""Mask generation for 3D lobe inpainting.

Generates random masks for training and provides utilities for sampling.
Volume convention: (1, D, H, W) = (1, Z, Y, X) where Z is depth (index 0 = top).
Mask convention: 1 = known (keep), 0 = unknown (generate).
"""

import math
import random

import torch


def generate_well_mask(volume_shape=(50, 50, 50), n_wells=None, max_wells=5,
                       horiz_prob=0.5):
    """Generate a mask with 1D well paths through the volume.

    Wells are 1 voxel wide with variable depth. Vertical wells go straight
    down; L-shaped wells have a vertical leg then a horizontal leg at a
    kickoff depth. Wells never intersect.

    Args:
        volume_shape: (D, H, W) tuple
        n_wells: number of wells (random 1..max_wells if None)
        max_wells: max wells when n_wells is None
        horiz_prob: probability a well is L-shaped horizontal

    Returns:
        mask: (1, D, H, W) float tensor
    """
    D, H, W = volume_shape
    mask = torch.zeros(1, D, H, W)
    occupied_xy = set()     # (x, y) positions used by vertical segments
    occupied_voxels = set() # (z, y, x) voxels used by any well segment

    if n_wells is None:
        n_wells = random.randint(1, max_wells)

    for _ in range(n_wells):
        placed = False
        for _retry in range(100):
            x = random.randint(0, W - 1)
            y = random.randint(0, H - 1)
            if (x, y) in occupied_xy:
                continue

            # Variable depth: minimum 5 voxels
            z_start = random.randint(0, D - 5)
            z_end = random.randint(z_start + 4, D - 1)

            is_horizontal = random.random() < horiz_prob

            if not is_horizontal:
                # Pure vertical well
                new_voxels = {(z, y, x) for z in range(z_start, z_end + 1)}
                if new_voxels & occupied_voxels:
                    continue
                for z in range(z_start, z_end + 1):
                    mask[0, z, y, x] = 1.0
                occupied_xy.add((x, y))
                occupied_voxels |= new_voxels
                placed = True
                break
            else:
                # L-shaped: vertical leg + horizontal leg
                z_kick = random.randint(z_start, z_end)
                vert_voxels = {(z, y, x) for z in range(z_start, z_kick + 1)}

                # Random cardinal direction for horizontal leg
                direction = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
                dx, dy = direction
                horiz_len = random.randint(3, 15)

                horiz_voxels = set()
                for step in range(1, horiz_len + 1):
                    nx = x + dx * step
                    ny = y + dy * step
                    if 0 <= nx < W and 0 <= ny < H:
                        horiz_voxels.add((z_kick, ny, nx))
                    else:
                        break

                all_new = vert_voxels | horiz_voxels
                if all_new & occupied_voxels:
                    continue

                for z, vy, vx in all_new:
                    mask[0, z, vy, vx] = 1.0
                occupied_xy.add((x, y))
                # Also reserve x,y for horizontal endpoints to prevent vertical
                # wells from intersecting the horizontal leg
                for z, vy, vx in horiz_voxels:
                    occupied_xy.add((vx, vy))
                occupied_voxels |= all_new
                placed = True
                break

        # If placement fails after retries, skip this well silently

    return mask


def generate_boundary_mask(volume_shape=(50, 50, 50), n_faces=None):
    """Generate a mask with boundary faces of the volume.

    Each face has random thickness 1-3 voxels.

    Args:
        volume_shape: (D, H, W) tuple
        n_faces: number of faces to include (random 1..6 if None)

    Returns:
        mask: (1, D, H, W) float tensor
    """
    D, H, W = volume_shape
    mask = torch.zeros(1, D, H, W)

    all_faces = ['d_min', 'd_max', 'h_min', 'h_max', 'w_min', 'w_max']
    if n_faces is None:
        n_faces = random.randint(1, 6)
    chosen = random.sample(all_faces, n_faces)

    for face in chosen:
        t = random.randint(1, 3)
        if face == 'd_min':
            mask[0, :t, :, :] = 1.0
        elif face == 'd_max':
            mask[0, -t:, :, :] = 1.0
        elif face == 'h_min':
            mask[0, :, :t, :] = 1.0
        elif face == 'h_max':
            mask[0, :, -t:, :] = 1.0
        elif face == 'w_min':
            mask[0, :, :, :t] = 1.0
        elif face == 'w_max':
            mask[0, :, :, -t:] = 1.0

    return mask


def generate_cross_section_mask(volume_shape=(50, 50, 50), max_tilt_deg=30):
    """Generate a 1-voxel-thick 2D cross-section mask through the volume.

    The cross-section is a plane at any angle in the x-y plane, with up to
    max_tilt_deg tilt from vertical (z-axis).

    Args:
        volume_shape: (D, H, W) tuple
        max_tilt_deg: maximum tilt from vertical in degrees

    Returns:
        mask: (1, D, H, W) float tensor
    """
    D, H, W = volume_shape

    # Random orientation in x-y plane
    theta = random.uniform(0, math.pi)
    # Random tilt from vertical
    max_tilt_rad = math.radians(max_tilt_deg)
    phi = random.uniform(-max_tilt_rad, max_tilt_rad)

    # Plane normal vector
    nx = math.cos(theta) * math.cos(phi)
    ny = math.sin(theta) * math.cos(phi)
    nz = math.sin(phi)

    # Plane passes through a random interior point (avoid edges)
    margin = 5
    cx = random.uniform(margin, W - 1 - margin)
    cy = random.uniform(margin, H - 1 - margin)
    cz = random.uniform(margin, D - 1 - margin)

    # Vectorized distance computation
    z_coords = torch.arange(D, dtype=torch.float32)
    y_coords = torch.arange(H, dtype=torch.float32)
    x_coords = torch.arange(W, dtype=torch.float32)
    # (D, H, W) grids
    zg, yg, xg = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')

    dist = nx * (xg - cx) + ny * (yg - cy) + nz * (zg - cz)
    mask = (dist.abs() < 0.5).float().unsqueeze(0)  # (1, D, H, W)

    return mask


def generate_combination_mask(volume_shape, mask_types):
    """Generate a mask combining multiple mask types (union).

    Args:
        volume_shape: (D, H, W) tuple
        mask_types: list of strings from ['wells', 'boundaries', 'cross_sections']

    Returns:
        mask: (1, D, H, W) float tensor
    """
    generators = {
        'wells': generate_well_mask,
        'boundaries': generate_boundary_mask,
        'cross_sections': generate_cross_section_mask,
    }
    mask = torch.zeros(1, *volume_shape)
    for mt in mask_types:
        mask = torch.clamp(mask + generators[mt](volume_shape), 0, 1)
    return mask


def generate_training_mask(volume_shape=(50, 50, 50)):
    """Generate a random training mask following the specified distribution.

    Distribution:
        30%: empty mask (unconditional generation)
        70%: mask present, subdivided as:
            1/6: wells only
            1/6: boundaries only
            1/6: cross-section only
            1/2: combinations (uniform across 4 combo types)

    Returns:
        mask: (1, D, H, W) float tensor, 1=known 0=unknown
    """
    if random.random() < 0.30:
        return torch.zeros(1, *volume_shape)

    r = random.random()
    if r < 1 / 6:
        return generate_well_mask(volume_shape)
    elif r < 2 / 6:
        return generate_boundary_mask(volume_shape)
    elif r < 3 / 6:
        return generate_cross_section_mask(volume_shape)
    else:
        combos = [
            ['wells', 'boundaries'],
            ['wells', 'cross_sections'],
            ['boundaries', 'cross_sections'],
            ['wells', 'boundaries', 'cross_sections'],
        ]
        return generate_combination_mask(volume_shape, random.choice(combos))


def apply_inpaint_output(samples, mask, known_data):
    """Hard-replace known voxels in generated samples.

    Args:
        samples: (B, 1, D, H, W) generated output
        mask: (B, 1, D, H, W) binary, 1=known
        known_data: (B, 1, D, H, W) clean values where mask=1

    Returns:
        result: (B, 1, D, H, W) with known regions replaced
    """
    return samples * (1 - mask) + known_data * mask
