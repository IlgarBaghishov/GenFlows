"""Big reservoir generation via parallel denoising with overlap averaging.

Generates a large 3D reservoir by denoising a grid of overlapping 50x50x50
blocks simultaneously (MultiDiffusion-style). At each denoising step, all
blocks predict velocities independently (with well-only conditioning), and
overlapping predictions are blended with smooth weights. Wells are re-injected
at each step at the correct noise level.

Data axis convention (from empirical analysis of lobe data):
    Axis 0 = X (horizontal), Axis 1 = Y (horizontal), Axis 2 = Z (depth/vertical)
Blocks tile axes 0 and 1 (horizontal plane). Axis 2 stays at block_size.
"""

import time
import numpy as np
import torch

from genflows.utils.masking_lobes import apply_inpaint_output


# ---------------------------------------------------------------------------
# Conditioning map
# ---------------------------------------------------------------------------

def compute_conditioning_map(grid_shape, height_range=(0.8, 0.2),
                             radius_range=(0.8, 0.2), ar_range=(0.8, 0.2),
                             azimuth_range=(0.1, 0.9), ntg=0.5):
    """Compute normalized conditioning (5D) for each block position.

    Height, radius, aspect_ratio interpolated linearly along Y (i axis).
    Azimuth interpolated linearly along X (j axis). NTG fixed.

    Returns:
        (ny, nx, 5) float32 — [height, radius, aspect_ratio, azimuth, ntg]
    """
    ny, nx = grid_shape
    cond_map = np.zeros((ny, nx, 5), dtype=np.float32)
    for i in range(ny):
        for j in range(nx):
            x_frac = j / max(nx - 1, 1)
            y_frac = i / max(ny - 1, 1)
            cond_map[i, j] = [
                height_range[0] + (height_range[1] - height_range[0]) * y_frac,
                radius_range[0] + (radius_range[1] - radius_range[0]) * y_frac,
                ar_range[0] + (ar_range[1] - ar_range[0]) * y_frac,
                azimuth_range[0] + (azimuth_range[1] - azimuth_range[0]) * x_frac,
                ntg,
            ]
    return cond_map


# ---------------------------------------------------------------------------
# Well generation
# ---------------------------------------------------------------------------

def _generate_well_path(volume_shape, min_horiz_len, max_horiz_len,
                        occupied_voxels, rng):
    """Generate a single L-shaped well path (vertical + horizontal leg).

    Returns list of (z, y, x) coordinates, or None if placement failed.
    """
    D, H, W = volume_shape

    for _ in range(100):
        x = rng.integers(5, W - 5)
        y = rng.integers(5, H - 5)
        z_start = rng.integers(0, max(1, D // 4))
        z_kick = rng.integers(z_start + 5, D - 2)

        dy, dx = [(0, 1), (0, -1), (1, 0), (-1, 0)][rng.integers(4)]
        horiz_len = rng.integers(min_horiz_len, max_horiz_len + 1)

        path = [(z, y, x) for z in range(z_start, z_kick + 1)]

        cy, cx = y, x
        for _ in range(horiz_len):
            cy += dy
            cx += dx
            if 0 <= cy < H and 0 <= cx < W:
                path.append((z_kick, cy, cx))
            else:
                break

        if not set(path) & occupied_voxels:
            return path

    return None


def _assign_well_facies(path, height_norm, radius_norm, rng):
    """Assign alternating sand(1)/shale(-1) facies along a well path.

    Vertical segment length ~ height, horizontal segment length ~ radius.
    """
    facies = np.zeros(len(path), dtype=np.float32)

    # Find kickoff index (vertical → horizontal transition)
    kick_idx = len(path)
    for k in range(1, len(path)):
        if path[k][0] == path[k - 1][0]:
            kick_idx = k
            break

    current_val = 1.0 if rng.random() > 0.5 else -1.0

    # Vertical section
    v_base = max(3, int(height_norm * 20 + 3))
    seg_remaining = max(3, v_base + int(rng.integers(-2, 3)))
    for k in range(kick_idx):
        facies[k] = current_val
        seg_remaining -= 1
        if seg_remaining <= 0:
            current_val *= -1
            seg_remaining = max(3, v_base + int(rng.integers(-2, 3)))

    # Horizontal section
    h_base = max(3, int(radius_norm * 20 + 3))
    seg_remaining = max(3, h_base + int(rng.integers(-2, 3)))
    for k in range(kick_idx, len(path)):
        facies[k] = current_val
        seg_remaining -= 1
        if seg_remaining <= 0:
            current_val *= -1
            seg_remaining = max(3, h_base + int(rng.integers(-2, 3)))

    return facies


def generate_wells_for_block(volume_shape, height_norm, radius_norm,
                             n_wells=2, min_horiz_len=15, max_horiz_len=40,
                             rng=None):
    """Generate well mask and known_data for one block.

    Creates L-shaped wells with alternating sand/shale facies whose segment
    lengths correlate with lobe height (vertical) and radius (horizontal).

    Returns:
        (well_mask, well_known_data) each (1, 1, D, H, W) tensors
    """
    if rng is None:
        rng = np.random.default_rng()

    D, H, W = volume_shape
    mask = torch.zeros(1, 1, D, H, W)
    known = torch.zeros(1, 1, D, H, W)
    occupied = set()

    for _ in range(n_wells):
        path = _generate_well_path(volume_shape, min_horiz_len, max_horiz_len,
                                   occupied, rng)
        if path is None:
            continue
        facies = _assign_well_facies(path, height_norm, radius_norm, rng)
        occupied |= set(path)
        for (z, y, x), val in zip(path, facies):
            mask[0, 0, z, y, x] = 1.0
            known[0, 0, z, y, x] = float(val)

    return mask, known


def generate_all_wells(grid_shape, cond_map, block_size=50,
                       n_wells_per_block=2, min_horiz_len=15,
                       max_horiz_len=40, seed=42):
    """Pre-generate wells for all blocks.

    Returns:
        (well_masks, well_data) — dicts keyed by (i, j) tuples
    """
    rng = np.random.default_rng(seed)
    ny, nx = grid_shape
    vol = (block_size, block_size, block_size)
    well_masks, well_data = {}, {}

    for i in range(ny):
        for j in range(nx):
            m, d = generate_wells_for_block(
                vol, cond_map[i, j, 0], cond_map[i, j, 1],
                n_wells=n_wells_per_block,
                min_horiz_len=min_horiz_len, max_horiz_len=max_horiz_len,
                rng=rng,
            )
            well_masks[(i, j)] = m
            well_data[(i, j)] = d

    return well_masks, well_data


# ---------------------------------------------------------------------------
# Blending weights
# ---------------------------------------------------------------------------

def _compute_blend_weights(grid_shape, block_size, overlap, device):
    """Precompute per-block blending weights for overlap averaging.

    Interior voxels get weight 1. Overlap faces get a linear ramp from
    ~0 (at the edge shared with neighbor) to ~1 (at the interior boundary).
    Ramps from neighboring blocks sum to 1 everywhere (including corners).
    """
    ny, nx = grid_shape
    weights = {}
    # Ramps that sum to 1 with their complement
    ramp_up = torch.linspace(0, 1, overlap + 2)[1:-1]    # (overlap,)
    ramp_down = torch.linspace(1, 0, overlap + 2)[1:-1]  # (overlap,)

    for i in range(ny):
        for j in range(nx):
            w = torch.ones(block_size, block_size, block_size)
            if j > 0:
                w[:overlap, :, :] *= ramp_up.view(-1, 1, 1)
            if j < nx - 1:
                w[-overlap:, :, :] *= ramp_down.view(-1, 1, 1)
            if i > 0:
                w[:, :overlap, :] *= ramp_up.view(1, -1, 1)
            if i < ny - 1:
                w[:, -overlap:, :] *= ramp_down.view(1, -1, 1)
            weights[(i, j)] = w.to(device)

    return weights


# ---------------------------------------------------------------------------
# Parallel generation (MultiDiffusion-style)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_big_reservoir(method, grid_shape, block_size, overlap, cond_map,
                           well_masks=None, well_data=None,
                           n_steps=50, cfg_scale=3.0, max_batch=10,
                           device='cuda'):
    """Generate a big reservoir via parallel denoising with overlap averaging.

    All blocks are denoised simultaneously. At each step, each block predicts
    a velocity (with well-only inpaint conditioning), and overlapping
    predictions are blended with smooth weights. Wells are re-injected at
    every step at the OT-path noise level.

    Args:
        method: loaded generative method (FlowMatching)
        grid_shape: (ny, nx) number of blocks
        block_size: voxels per block dimension (50)
        overlap: overlap voxels between neighbors (e.g. 10)
        cond_map: (ny, nx, 5) normalized conditioning per block
        well_masks: dict (i,j) -> (1,1,S,S,S) or None
        well_data: dict (i,j) -> (1,1,S,S,S) or None
        n_steps: sampling steps
        cfg_scale: classifier-free guidance scale
        max_batch: max blocks per GPU call
        device: torch device

    Returns:
        generated_blocks: dict (i,j) -> (1,1,S,S,S) tensor (CPU, binary {-1,1})
        step_times: list of (step, n_blocks, elapsed_seconds)
    """
    ny, nx = grid_shape
    S = block_size
    stride = S - overlap
    total_x = S + (nx - 1) * stride
    total_y = S + (ny - 1) * stride
    has_wells = well_masks is not None
    dt = 1.0 / n_steps
    positions = [(i, j) for i in range(ny) for j in range(nx)]

    def block_origin(i, j):
        return (j * stride, i * stride)

    # 1. Initialize global volume with noise
    x_global = torch.randn(total_x, total_y, S, device=device)

    # 2. Precompute blending weights
    blend_w = _compute_blend_weights(grid_shape, S, overlap, device)

    # 3. Move per-block well tensors to device once
    dev_wmasks, dev_wdata = {}, {}
    if has_wells:
        for i, j in positions:
            dev_wmasks[(i, j)] = well_masks[(i, j)].to(device)
            dev_wdata[(i, j)] = well_data[(i, j)].to(device)

    # 4. Denoising loop
    step_times = []

    for step in range(n_steps):
        t_val = step * dt
        t0 = time.time()

        v_accum = torch.zeros_like(x_global)
        w_accum = torch.zeros_like(x_global)

        # Process all blocks in sub-batches
        for b_start in range(0, len(positions), max_batch):
            batch_pos = positions[b_start:b_start + max_batch]
            bs = len(batch_pos)

            # Extract blocks from global volume
            blocks = torch.zeros(bs, 1, S, S, S, device=device)
            mask_batch = torch.zeros(bs, 1, S, S, S, device=device)
            known_batch = torch.zeros(bs, 1, S, S, S, device=device)
            cond_batch = torch.zeros(bs, 5, device=device)

            for idx, (i, j) in enumerate(batch_pos):
                xs, ys = block_origin(i, j)
                blocks[idx, 0] = x_global[xs:xs + S, ys:ys + S, :]
                if has_wells:
                    mask_batch[idx] = dev_wmasks[(i, j)]
                    known_batch[idx] = dev_wdata[(i, j)]
                cond_batch[idx] = torch.tensor(cond_map[i, j], device=device)

            # Predict velocity with well-only inpaint context
            method.model.set_inpaint_context(mask_batch, known_batch)
            t_tensor = torch.full((bs,), t_val, device=device)
            t_emb = t_tensor * 1000

            if cfg_scale > 0:
                v_cond = method.model(blocks, t_emb, cond_batch)
                v_uncond = method.model(blocks, t_emb)
                v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v_pred = method.model(blocks, t_emb)

            method.model.clear_inpaint_context()

            # Scatter weighted velocities to global accumulators
            for idx, (i, j) in enumerate(batch_pos):
                xs, ys = block_origin(i, j)
                w = blend_w[(i, j)]
                v_accum[xs:xs + S, ys:ys + S, :] += w * v_pred[idx, 0]
                w_accum[xs:xs + S, ys:ys + S, :] += w

        # Average and step
        x_global = x_global + (v_accum / w_accum.clamp(min=1e-8)) * dt

        elapsed = time.time() - t0
        step_times.append((step, len(positions), elapsed))
        if step % 10 == 0 or step == n_steps - 1:
            print(f"  Step {step}/{n_steps}: {elapsed:.1f}s")

    # 5. Extract final blocks, hard-replace wells, binarize
    generated_blocks = {}
    for i, j in positions:
        xs, ys = block_origin(i, j)
        block = x_global[xs:xs + S, ys:ys + S, :].clone()
        if has_wells:
            wm = dev_wmasks[(i, j)][0, 0]
            wk = dev_wdata[(i, j)][0, 0]
            block = block * (1 - wm) + wk * wm
        block = (block > 0).float() * 2 - 1
        generated_blocks[(i, j)] = block.unsqueeze(0).unsqueeze(0).cpu()

    return generated_blocks, step_times


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble_reservoir(generated_blocks, grid_shape, block_size, overlap):
    """Stitch generated blocks into a full volume.

    Axes 0 (X) and 1 (Y) are stitched. Axis 2 (Z, depth) stays at block_size.

    Returns:
        (X_total, Y_total, Z_depth) int8 array with values {0, 1}
    """
    ny, nx = grid_shape
    stride = block_size - overlap
    total_x = block_size + (nx - 1) * stride
    total_y = block_size + (ny - 1) * stride

    result = np.zeros((total_x, total_y, block_size), dtype=np.float32)
    for i in range(ny):
        for j in range(nx):
            block = generated_blocks[(i, j)].numpy()[0, 0]
            x_start = j * stride
            y_start = i * stride
            result[x_start:x_start + block_size,
                   y_start:y_start + block_size, :] = block

    return (result > 0).astype(np.int8)


def assemble_well_mask(well_masks, grid_shape, block_size, overlap):
    """Stitch per-block well masks into a full-volume mask.

    Returns:
        (X_total, Y_total, Z_depth) float32 array
    """
    ny, nx = grid_shape
    stride = block_size - overlap
    total_x = block_size + (nx - 1) * stride
    total_y = block_size + (ny - 1) * stride

    result = np.zeros((total_x, total_y, block_size), dtype=np.float32)
    for i in range(ny):
        for j in range(nx):
            block = well_masks[(i, j)].numpy()[0, 0]
            x_start = j * stride
            y_start = i * stride
            region = result[x_start:x_start + block_size,
                            y_start:y_start + block_size, :]
            np.maximum(region, block, out=region)

    return result
