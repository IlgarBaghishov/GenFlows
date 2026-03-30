"""Conditioning sensitivity: vary one parameter from the same noise.

Generates 50x50x50 voxels with identical initial noise but different
height (or radius) values to see how the model responds to small
conditioning changes.

Usage:
    cd examples/lobes/cond_sensitivity
    python generate.py
"""

import os
import time
import numpy as np
import torch

from genflows.models.unet3d import UNet3D
from genflows.methods.flow_matching import FlowMatching

# ---- Configuration --------------------------------------------------------

CHECKPOINT = '../standard/checkpoints/flow_matching.pt'
COND_STATS = '../standard/checkpoints/cond_stats.npz'

SEED = 42
N_STEPS = 50
CFG_SCALE = 3.0

# Height sweep: centered on 25, dense near center, spreading outward
HEIGHT_VALUES = [5, 10, 15, 20, 23, 24, 24.5, 25, 25.5, 26, 27, 30, 35, 40, 45]
HEIGHT_FIXED = dict(radius=16.5, aspect_ratio=2.0, azimuth=90.0, ntg=0.5)

# Radius sweep: centered on 16.5, dense near center, spreading outward
RADIUS_VALUES = [3, 6, 10, 13, 15, 16, 16.5, 17, 18, 20, 23, 27, 30]
RADIUS_FIXED = dict(height=25.0, aspect_ratio=2.0, azimuth=90.0, ntg=0.5)


def normalize(raw, cond_min, cond_max):
    return (raw - cond_min) / (cond_max - cond_min + 1e-8)


def run_sweep(method, sweep_values, param_idx, fixed_raw, cond_min, cond_max,
              device):
    """Generate one sample per sweep value, all from the same noise."""
    samples = []
    for val in sweep_values:
        raw = fixed_raw.copy()
        raw[param_idx] = val
        cond_norm = normalize(raw, cond_min, cond_max)
        cond_tensor = torch.from_numpy(cond_norm).unsqueeze(0).to(device)

        torch.manual_seed(SEED)
        x = method.sample((1, 1, 50, 50, 50), device, cond=cond_tensor,
                          cfg_scale=CFG_SCALE, n_steps=N_STEPS)
        binary = (x > 0).float().cpu().numpy()[0, 0]  # (50,50,50)
        samples.append(binary)
    return np.stack(samples)  # (N, 50, 50, 50)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs('results', exist_ok=True)

    # Load model
    model = UNet3D(in_channels=1, num_time_embs=1).to(device)
    model.load_state_dict(
        torch.load(CHECKPOINT, map_location=device, weights_only=True))
    model.eval()
    method = FlowMatching(model)

    # Load normalization stats
    stats = np.load(COND_STATS)
    cond_min, cond_max = stats['cond_min'], stats['cond_max']
    param_names = ['height', 'radius', 'aspect_ratio', 'azimuth', 'ntg']
    print("Conditioning ranges:")
    for i, name in enumerate(param_names):
        print(f"  {name}: [{cond_min[i]:.2f}, {cond_max[i]:.2f}]")

    # Height sweep
    print(f"\nHeight sweep: {HEIGHT_VALUES}")
    fixed_raw = np.array([
        HEIGHT_FIXED.get('height', 25.0),
        HEIGHT_FIXED['radius'],
        HEIGHT_FIXED['aspect_ratio'],
        HEIGHT_FIXED['azimuth'],
        HEIGHT_FIXED['ntg'],
    ], dtype=np.float32)

    t0 = time.time()
    height_samples = run_sweep(method, HEIGHT_VALUES, 0, fixed_raw,
                               cond_min, cond_max, device)
    print(f"  Done in {time.time() - t0:.1f}s — {height_samples.shape}")
    np.savez('results/height_sweep.npz',
             samples=height_samples,
             values=np.array(HEIGHT_VALUES, dtype=np.float32))

    # Radius sweep
    print(f"\nRadius sweep: {RADIUS_VALUES}")
    fixed_raw = np.array([
        RADIUS_FIXED['height'],
        RADIUS_FIXED.get('radius', 16.5),
        RADIUS_FIXED['aspect_ratio'],
        RADIUS_FIXED['azimuth'],
        RADIUS_FIXED['ntg'],
    ], dtype=np.float32)

    t0 = time.time()
    radius_samples = run_sweep(method, RADIUS_VALUES, 1, fixed_raw,
                               cond_min, cond_max, device)
    print(f"  Done in {time.time() - t0:.1f}s — {radius_samples.shape}")
    np.savez('results/radius_sweep.npz',
             samples=radius_samples,
             values=np.array(RADIUS_VALUES, dtype=np.float32))

    print("\nDone! Results saved to results/")


if __name__ == '__main__':
    main()
