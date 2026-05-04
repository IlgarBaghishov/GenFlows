import argparse
import glob
import os
import re
import time
import numpy as np
import torch

from resflow.models.unet3d import UNet3D
from resflow.methods.flow_matching import FlowMatching
from resflow.utils.masking_lobes import (
    generate_well_mask, generate_boundary_mask, generate_cross_section_mask,
    generate_combination_mask, apply_inpaint_output,
)
from resflow.utils.data_lobes import LobeDataset


def load_cond_stats(path='checkpoints/cond_stats.npz'):
    stats = np.load(path)
    return stats['cond_min'], stats['cond_max']


def load_method(ckpt_path, device):
    """Load Flow Matching inpainting method."""
    model = UNet3D(in_channels=3, out_channels=1, num_time_embs=1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    return FlowMatching(model)


def sample_inpaint(method, shape, device, cond, mask, known_data, **sample_kwargs):
    """Run inpainting: set context, sample, hard-replace known regions."""
    method.model.set_inpaint_context(mask, known_data)
    samples = method.sample(shape, device, cond=cond, **sample_kwargs)
    result = apply_inpaint_output(samples, mask, known_data)
    method.model.clear_inpaint_context()
    return result


def find_latest_epoch(checkpoint_dir='checkpoints'):
    """Find the latest epoch from available flow_matching_epoch*.pt files."""
    pattern = os.path.join(checkpoint_dir, 'flow_matching_epoch*.pt')
    files = glob.glob(pattern)
    if not files:
        return None
    epochs = []
    for f in files:
        m = re.search(r'epoch(\d+)\.pt$', f)
        if m:
            epochs.append(int(m.group(1)))
    return max(epochs) if epochs else None


def main():
    parser = argparse.ArgumentParser(description='Sample from inpainting checkpoints')
    parser.add_argument('--epoch', type=int, default=None,
                        help='Epoch checkpoint to sample from (default: latest)')
    parser.add_argument('--checkpoints-dir', default='checkpoints')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Determine which epoch to sample from
    epoch = args.epoch
    if epoch is None:
        epoch = find_latest_epoch(args.checkpoints_dir)
        if epoch is None:
            print("No checkpoints found. Run train.py first.")
            return
    print(f"Sampling from epoch {epoch}")

    ckpt_path = os.path.join(args.checkpoints_dir, f'flow_matching_epoch{epoch:03d}.pt')
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return

    # Create epoch-specific results directory
    results_dir = f'results/epoch_{epoch:03d}'
    os.makedirs(results_dir, exist_ok=True)

    # Load model
    method = load_method(ckpt_path, device)
    print(f"Loaded checkpoint: {ckpt_path}")

    # Load normalization stats
    cond_min, cond_max = load_cond_stats(os.path.join(args.checkpoints_dir, 'cond_stats.npz'))

    # Load test data as ground truth
    dataset = LobeDataset(data_dir='../data')
    n_samples = 4
    print(f"Loading {n_samples} test volumes as ground truth...")
    gt_facies = []
    gt_cond = []
    for i in range(len(dataset) - n_samples, len(dataset)):
        f, c = dataset[i]
        gt_facies.append(f)
        gt_cond.append(c)
    gt_facies = torch.stack(gt_facies).to(device)
    gt_cond = torch.stack(gt_cond).to(device)

    shape = (n_samples, 1, 50, 50, 50)

    # Fixed seed for consistent masks across epochs
    torch.manual_seed(42)
    np.random.seed(42)

    # Define mask scenarios
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

        t0 = time.time()
        result = sample_inpaint(
            method, shape, device, gt_cond, mask, known_data,
            cfg_scale=cfg_scale, n_steps=n_steps,
        )
        elapsed = time.time() - t0
        binary = (result > 0).float()
        print(f"  flow_matching: {elapsed:.2f}s")

        # Save results
        np.save(os.path.join(results_dir, f'flow_matching_{scenario_name}_{n_steps}steps.npy'),
                binary.cpu().numpy())
        np.save(os.path.join(results_dir, f'mask_{scenario_name}.npy'),
                mask.cpu().numpy())

    # Save ground truth
    gt_binary = (gt_facies > 0).float()
    np.save(os.path.join(results_dir, 'ground_truth.npy'), gt_binary.cpu().numpy())
    np.save(os.path.join(results_dir, 'ground_truth_cond.npy'), gt_cond.cpu().numpy())

    print(f"\nDone! Results saved to '{results_dir}/'.")


if __name__ == "__main__":
    main()
