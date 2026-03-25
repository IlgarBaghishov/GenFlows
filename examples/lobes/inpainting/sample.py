import os
import time
import numpy as np
import torch

from genflows.models.unet3d import UNet3D
from genflows.methods.diffusion import Diffusion
from genflows.methods.flow_matching import FlowMatching
from genflows.methods.meanflow import MeanFlow
from genflows.methods.rectified_flow import RectifiedFlow
from genflows.utils.masking_lobes import (
    generate_well_mask, generate_boundary_mask, generate_cross_section_mask,
    generate_combination_mask, apply_inpaint_output,
)
from genflows.utils.data_lobes import LobeDataset


def load_cond_stats(path='checkpoints/cond_stats.npz'):
    stats = np.load(path)
    return stats['cond_min'], stats['cond_max']


def normalize_cond(raw_cond, cond_min, cond_max):
    return (raw_cond - cond_min) / (cond_max - cond_min + 1e-8)


def load_method(name, ckpt_path, device):
    """Load an inpainting method with in_channels=3."""
    if name == 'diffusion':
        model = UNet3D(in_channels=3, out_channels=1, num_time_embs=1).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        return Diffusion(model, n_steps=1000)
    elif name == 'flow_matching':
        model = UNet3D(in_channels=3, out_channels=1, num_time_embs=1).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        return FlowMatching(model)
    elif name == 'rectified_flow':
        model = UNet3D(in_channels=3, out_channels=1, num_time_embs=1).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        return RectifiedFlow(model)
    elif name == 'meanflow_std':
        model = UNet3D(in_channels=3, out_channels=1, num_time_embs=2).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        return MeanFlow(model, cfg_mode='standard')
    elif name == 'meanflow_embed':
        model = UNet3D(in_channels=3, out_channels=1, num_time_embs=2).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        return MeanFlow(model, cfg_mode='embedded', omega=3.0, kappa=0.0)
    else:
        raise ValueError(f"Unknown method: {name}")


def sample_inpaint(method, shape, device, cond, mask, known_data, **sample_kwargs):
    """Run inpainting: set context, sample, hard-replace known regions.

    Args:
        method: generative method (Diffusion, FlowMatching, etc.)
        shape: (B, 1, D, H, W)
        device: torch device
        cond: (B, 5) normalized conditioning
        mask: (B, 1, D, H, W) binary, 1=known
        known_data: (B, 1, D, H, W) clean values where mask=1
        **sample_kwargs: passed to method.sample (cfg_scale, n_steps, etc.)

    Returns:
        result: (B, 1, D, H, W) with known regions hard-replaced
    """
    method.model.set_inpaint_context(mask, known_data)
    samples = method.sample(shape, device, cond=cond, **sample_kwargs)
    result = apply_inpaint_output(samples, mask, known_data)
    method.model.clear_inpaint_context()
    return result


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

    # Load test data to use as ground truth for mask creation
    dataset = LobeDataset(data_dir='../data')
    n_samples = 4
    print(f"\nLoading {n_samples} test volumes as ground truth for inpainting...")
    gt_facies = []
    gt_cond = []
    # Use samples from the end of the dataset (likely in the test split)
    for i in range(len(dataset) - n_samples, len(dataset)):
        f, c = dataset[i]
        gt_facies.append(f)
        gt_cond.append(c)
    gt_facies = torch.stack(gt_facies).to(device)  # (N, 1, 50, 50, 50)
    gt_cond = torch.stack(gt_cond).to(device)      # (N, 5)

    shape = (n_samples, 1, 50, 50, 50)

    # Load all available methods
    methods = {}
    checkpoints = {
        # 'diffusion': 'checkpoints/diffusion.pt',
        'flow_matching': 'checkpoints/flow_matching.pt',
        # 'rectified_flow': 'checkpoints/rectified_flow.pt',
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
        print("No checkpoints found. Run train.py first.")
        return

    # Define mask scenarios to test
    mask_scenarios = {
        'wells': lambda: torch.stack([generate_well_mask() for _ in range(n_samples)]).to(device),
        'boundaries': lambda: torch.stack([generate_boundary_mask() for _ in range(n_samples)]).to(device),
        'cross_section': lambda: torch.stack([generate_cross_section_mask() for _ in range(n_samples)]).to(device),
        'combo_wells_bounds': lambda: torch.stack([
            generate_combination_mask((50, 50, 50), ['wells', 'boundaries'])
            for _ in range(n_samples)
        ]).to(device),
        'combo_all': lambda: torch.stack([
            generate_combination_mask((50, 50, 50), ['wells', 'boundaries', 'cross_sections'])
            for _ in range(n_samples)
        ]).to(device),
    }

    n_steps = 50
    cfg_scale = 3.0

    for scenario_name, make_mask in mask_scenarios.items():
        mask = make_mask()
        known_data = gt_facies * mask
        mask_frac = mask.sum() / mask.numel()
        print(f"\n--- Mask: {scenario_name} (known: {mask_frac:.1%}) ---")

        for method_name, method in methods.items():
            t0 = time.time()
            result = sample_inpaint(
                method, shape, device, gt_cond, mask, known_data,
                cfg_scale=cfg_scale, n_steps=n_steps,
            )
            elapsed = time.time() - t0
            binary = (result > 0).float()
            print(f"  {method_name}: {elapsed:.2f}s")

            # Save results
            fname = f"results/{method_name}_{scenario_name}_{n_steps}steps.npy"
            np.save(fname, binary.cpu().numpy())

        # Also save the mask and ground truth for comparison
        np.save(f"results/mask_{scenario_name}.npy", mask.cpu().numpy())

    # Save ground truth
    gt_binary = (gt_facies > 0).float()
    np.save("results/ground_truth.npy", gt_binary.cpu().numpy())
    np.save("results/ground_truth_cond.npy", gt_cond.cpu().numpy())

    print("\nDone! Results saved to 'results/'.")


if __name__ == "__main__":
    main()
