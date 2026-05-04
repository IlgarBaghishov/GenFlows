"""Generate a big reservoir via parallel denoising with overlap averaging.

Uses the trained inpainting model from examples/lobes/inpainting/checkpoints/.
All blocks are denoised simultaneously (MultiDiffusion-style) — no autoregressive
ordering. Wells are enforced via channel concatenation + OT-path re-injection.
Smooth transitions emerge from overlap averaging.

Runs two scenarios: without wells and with wells.

Usage:
    cd examples/lobes/big_reservoir
    python generate.py
"""

import os
import time
import numpy as np
import torch

from resflow.models.unet3d import UNet3D
from resflow.methods.flow_matching import FlowMatching
from resflow.assembly import (
    compute_conditioning_map,
    generate_all_wells,
    generate_big_reservoir,
    assemble_reservoir,
    assemble_well_mask,
)

# ---- Configuration --------------------------------------------------------

GRID_SHAPE = (10, 10)       # (ny, nx) blocks
BLOCK_SIZE = 50
OVERLAP = 20                 # overlap voxels for smooth blending between neighbors
N_STEPS = 50
CFG_SCALE = 3.0
SEED = 42

# Paths (relative to this script's directory)
CHECKPOINT_PATH = '../inpainting/checkpoints/flow_matching.pt'
COND_STATS_PATH = '../inpainting/checkpoints/cond_stats.npz'

# Property gradient (normalized to [0, 1])
# Height, radius, aspect_ratio all vary together along Y (i axis)
# Azimuth varies along X (j axis)
HEIGHT_RANGE = (0.8, 0.2)       # along Y
RADIUS_RANGE = (0.8, 0.2)       # along Y
ASPECT_RATIO_RANGE = (0.8, 0.2) # along Y (same direction as height/radius)
AZIMUTH_RANGE = (0.1, 0.9)      # along X
NTG = 0.7                       # fixed

# Wells
N_WELLS_PER_BLOCK = 2
WELL_HORIZ_LEN = (15, 40)
WELL_SEED = 123


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Grid: {GRID_SHAPE[0]}x{GRID_SHAPE[1]} blocks of {BLOCK_SIZE}^3")

    stride = BLOCK_SIZE - OVERLAP
    total_x = BLOCK_SIZE + (GRID_SHAPE[1] - 1) * stride
    total_y = BLOCK_SIZE + (GRID_SHAPE[0] - 1) * stride
    print(f"Output volume: {total_x} x {total_y} x {BLOCK_SIZE} (X x Y x Z_depth)")

    os.makedirs('results', exist_ok=True)

    # ---- Load model -------------------------------------------------------
    print("\nLoading model...")
    model = UNet3D(in_channels=3, out_channels=1, num_time_embs=1).to(device)
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.eval()
    method = FlowMatching(model)

    # ---- Conditioning map -------------------------------------------------
    cond_map = compute_conditioning_map(
        GRID_SHAPE,
        height_range=HEIGHT_RANGE,
        radius_range=RADIUS_RANGE,
        ar_range=ASPECT_RATIO_RANGE,
        azimuth_range=AZIMUTH_RANGE,
        ntg=NTG,
    )
    np.save('results/conditioning_map.npy', cond_map)
    print(f"\nConditioning map: {cond_map.shape}")
    print(f"  Height:  {cond_map[0, 0, 0]:.2f} -> {cond_map[-1, 0, 0]:.2f} (along Y)")
    print(f"  Radius:  {cond_map[0, 0, 1]:.2f} -> {cond_map[-1, 0, 1]:.2f} (along Y)")
    print(f"  AR:      {cond_map[0, 0, 2]:.2f} -> {cond_map[-1, 0, 2]:.2f} (along Y)")
    print(f"  Azimuth: {cond_map[0, 0, 3]:.2f} -> {cond_map[0, -1, 3]:.2f} (along X)")
    print(f"  NTG:     {cond_map[0, 0, 4]:.2f} (fixed)")

    # ---- Pre-generate wells -----------------------------------------------
    print(f"\nGenerating wells ({N_WELLS_PER_BLOCK} per block, "
          f"horiz length {WELL_HORIZ_LEN[0]}-{WELL_HORIZ_LEN[1]})...")
    well_masks, well_data = generate_all_wells(
        GRID_SHAPE, cond_map, block_size=BLOCK_SIZE,
        n_wells_per_block=N_WELLS_PER_BLOCK,
        min_horiz_len=WELL_HORIZ_LEN[0],
        max_horiz_len=WELL_HORIZ_LEN[1],
        seed=WELL_SEED,
    )
    well_mask_full = assemble_well_mask(
        well_masks, GRID_SHAPE, BLOCK_SIZE, OVERLAP)
    np.save('results/well_mask_assembled.npy', well_mask_full)
    total_well_voxels = sum(m.sum().item() for m in well_masks.values())
    print(f"  Total well voxels: {total_well_voxels:.0f}")

    # ---- Scenario 1: No wells ---------------------------------------------
    print("\n" + "=" * 60)
    print("SCENARIO 1: No wells (boundary conditioning only)")
    print("=" * 60)
    torch.manual_seed(SEED)
    t_start = time.time()
    blocks_no_wells, times_no_wells = generate_big_reservoir(
        method, GRID_SHAPE, BLOCK_SIZE, OVERLAP, cond_map,
        well_masks=None, well_data=None,
        n_steps=N_STEPS, cfg_scale=CFG_SCALE, device=device,
    )
    total_no_wells = time.time() - t_start
    print(f"\nTotal time (no wells): {total_no_wells:.1f}s")

    reservoir_no_wells = assemble_reservoir(
        blocks_no_wells, GRID_SHAPE, BLOCK_SIZE, OVERLAP)
    np.save('results/big_reservoir_no_wells.npy', reservoir_no_wells)
    print(f"Saved: {reservoir_no_wells.shape}, NTG={reservoir_no_wells.mean():.3f}")

    # ---- Scenario 2: With wells -------------------------------------------
    print("\n" + "=" * 60)
    print("SCENARIO 2: With wells (boundary + well conditioning)")
    print("=" * 60)
    torch.manual_seed(SEED)
    t_start = time.time()
    blocks_with_wells, times_with_wells = generate_big_reservoir(
        method, GRID_SHAPE, BLOCK_SIZE, OVERLAP, cond_map,
        well_masks=well_masks, well_data=well_data,
        n_steps=N_STEPS, cfg_scale=CFG_SCALE, device=device,
    )
    total_with_wells = time.time() - t_start
    print(f"\nTotal time (with wells): {total_with_wells:.1f}s")

    reservoir_with_wells = assemble_reservoir(
        blocks_with_wells, GRID_SHAPE, BLOCK_SIZE, OVERLAP)
    np.save('results/big_reservoir_with_wells.npy', reservoir_with_wells)
    print(f"Saved: {reservoir_with_wells.shape}, NTG={reservoir_with_wells.mean():.3f}")

    # ---- Save timing ------------------------------------------------------
    np.savez('results/timing.npz',
             times_no_wells=np.array(times_no_wells),
             times_with_wells=np.array(times_with_wells),
             total_no_wells=total_no_wells,
             total_with_wells=total_with_wells)

    print(f"\n{'=' * 60}")
    print(f"Done! Results saved to results/")
    print(f"  No wells:   {total_no_wells:.1f}s total")
    print(f"  With wells: {total_with_wells:.1f}s total")
    print(f"  Volume size: {reservoir_no_wells.shape}")


if __name__ == '__main__':
    main()
