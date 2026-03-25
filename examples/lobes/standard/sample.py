import os
import time
import numpy as np
import torch

from genflows.models.unet3d import UNet3D
from genflows.methods.diffusion import Diffusion
from genflows.methods.flow_matching import FlowMatching
from genflows.methods.meanflow import MeanFlow
from genflows.methods.rectified_flow import RectifiedFlow


def load_cond_stats(path='checkpoints/cond_stats.npz'):
    """Load conditioning normalization stats saved during training."""
    stats = np.load(path)
    return stats['cond_min'], stats['cond_max']


def normalize_cond(raw_cond, cond_min, cond_max):
    """Normalize raw conditioning values to [0, 1].

    Args:
        raw_cond: array of shape (N, 5) — [height, radius, aspect_ratio, azimuth_deg, ntg]
        cond_min, cond_max: arrays of shape (5,) from training dataset
    """
    return (raw_cond - cond_min) / (cond_max - cond_min + 1e-8)


def load_method(name, ckpt_path, device):
    """Load a method wrapper with pretrained weights."""
    if name == 'diffusion':
        model = UNet3D(in_channels=1, num_time_embs=1).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        return Diffusion(model, n_steps=1000)
    elif name == 'flow_matching':
        model = UNet3D(in_channels=1, num_time_embs=1).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        return FlowMatching(model)
    elif name in ('rectified_flow', 'rectified_flow_bwd', 'rectified_flow_bidir', 'rectified_flow_rand'):
        model = UNet3D(in_channels=1, num_time_embs=1).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        return RectifiedFlow(model)
    elif name == 'meanflow_std':
        model = UNet3D(in_channels=1, num_time_embs=2).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        return MeanFlow(model, cfg_mode='standard')
    elif name == 'meanflow_embed':
        model = UNet3D(in_channels=1, num_time_embs=2).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        return MeanFlow(model, cfg_mode='embedded', omega=3.0, kappa=0.0)
    else:
        raise ValueError(f"Unknown method: {name}")


def samples_to_binary(samples):
    """Convert model output from [-1, 1] to binary {0, 1}."""
    return (samples > 0).float()


def make_cond_grid(cond_min, cond_max, n_samples=8):
    """Create a grid of conditioning values spanning the training range.

    Returns conditioning for n_samples, linearly spaced between min and max
    for each parameter. Values are in raw (unnormalized) units.
    """
    raw = np.zeros((n_samples, 5), dtype=np.float32)
    for i in range(5):
        raw[:, i] = np.linspace(cond_min[i], cond_max[i], n_samples)
    return raw


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs("results", exist_ok=True)

    # Load normalization stats
    cond_min, cond_max = load_cond_stats()
    param_names = ['height', 'radius', 'aspect_ratio', 'azimuth', 'ntg']
    print("Conditioning ranges (from training):")
    for i, name in enumerate(param_names):
        print(f"  {name}: [{cond_min[i]:.2f}, {cond_max[i]:.2f}]")

    # Load all available methods
    methods = {}
    checkpoints = {
        # 'diffusion': 'checkpoints/diffusion.pt',
        'flow_matching': 'checkpoints/flow_matching.pt',
        # 'rectified_flow': 'checkpoints/rectified_flow.pt',
        # 'rectified_flow_bwd': 'checkpoints/rectified_flow_bwd.pt',
        # 'rectified_flow_bidir': 'checkpoints/rectified_flow_bidir.pt',
        # 'rectified_flow_rand': 'checkpoints/rectified_flow_rand.pt',
        # 'meanflow_std': 'checkpoints/meanflow_std.pt',
        # 'meanflow_embed': 'checkpoints/meanflow_embed.pt',
    }

    for name, path in checkpoints.items():
        if os.path.exists(path):
            print(f"Loading {name} from {path}")
            methods[name] = load_method(name, path, device)
        else:
            print(f"WARNING: {path} not found, skipping {name}")

    if not methods:
        print("No checkpoints found in 'checkpoints/'. Run train.py first.")
        return

    # Create conditioning: grid spanning the parameter space
    n_samples = 8
    raw_cond = make_cond_grid(cond_min, cond_max, n_samples)
    cond_norm = normalize_cond(raw_cond, cond_min, cond_max)
    cond_tensor = torch.from_numpy(cond_norm).to(device)

    shape = (n_samples, 1, 50, 50, 50)
    step_counts = [1, 5, 10, 50, 100, 500, 1000]

    print(f"\nGenerating {n_samples} samples per method per step count...")
    print(f"Conditioning values (raw):")
    for i in range(n_samples):
        vals = ', '.join(f"{param_names[j]}={raw_cond[i, j]:.2f}" for j in range(5))
        print(f"  Sample {i}: {vals}")

    for n_steps in step_counts:
        print(f"\n--- {n_steps} steps ---")

        if 'diffusion' in methods:
            for sampler in ['ddpm', 'ddim']:
                t0 = time.time()
                samples = methods['diffusion'].sample(
                    shape, device, cond=cond_tensor, cfg_scale=3.0,
                    n_steps=n_steps, sampler=sampler
                )
                elapsed = time.time() - t0
                binary = samples_to_binary(samples)
                tag = sampler.upper()
                print(f"  {tag}  {n_steps:>4} steps: {elapsed:.2f}s")
                np.save(f"results/samples_{sampler}_{n_steps}steps.npy", binary.cpu().numpy())

        for name in ['flow_matching', 'rectified_flow', 'rectified_flow_bwd',
                      'rectified_flow_bidir', 'rectified_flow_rand',
                      'meanflow_std', 'meanflow_embed']:
            if name not in methods:
                continue
            t0 = time.time()
            samples = methods[name].sample(
                shape, device, cond=cond_tensor, cfg_scale=3.0, n_steps=n_steps
            )
            elapsed = time.time() - t0
            binary = samples_to_binary(samples)
            short = name.upper().replace('_', '-')
            print(f"  {short:<15} {n_steps:>4} steps: {elapsed:.2f}s")
            np.save(f"results/samples_{name}_{n_steps}steps.npy", binary.cpu().numpy())

    print("\nDone! Samples saved to 'results/' as .npy files.")
    print("Each file has shape (N, 1, 50, 50, 50) with binary {0, 1} values.")


if __name__ == "__main__":
    main()
