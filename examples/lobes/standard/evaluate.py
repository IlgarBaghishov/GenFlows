"""Evaluate generative models by round-trip consistency with a pretrained CNN.

For each test sample: use its conditioning to generate a structure, then run
the CNN to predict properties back. Compare predicted vs. original conditioning.
"""

import argparse
import os
import time
import numpy as np
import torch

from genflows.models.unet3d import UNet3D
from genflows.methods.diffusion import Diffusion
from genflows.methods.flow_matching import FlowMatching
from genflows.methods.meanflow import MeanFlow
from genflows.methods.rectified_flow import RectifiedFlow
from genflows.utils.data_lobes import get_lobe_loaders
from genflows.utils.evaluation import (
    LobePropertyPredictor, compute_rmse, compute_r2, resolve_angle_ambiguity, plot_parity,
)


def load_cond_stats(path='checkpoints/cond_stats.npz'):
    stats = np.load(path)
    return stats['cond_min'], stats['cond_max']


def load_method(name, ckpt_path, device):
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


def main():
    parser = argparse.ArgumentParser(description='Evaluate generative models via CNN round-trip')
    parser.add_argument('--n_samples', type=int, default=500, help='Number of test samples to evaluate')
    parser.add_argument('--n_steps', type=int, default=50, help='Sampling steps')
    parser.add_argument('--cfg_scale', type=float, default=3.0, help='CFG scale')
    parser.add_argument('--batch_size', type=int, default=32, help='Generation batch size')
    parser.add_argument('--cnn_weights', type=str, default='../CNN/cnn3d_property_predictor.pth')
    parser.add_argument('--data_dir', type=str, default='../data')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load test split (same seed as training)
    _, _, test_loader, dataset = get_lobe_loaders(
        data_dir=args.data_dir, batch_size=args.batch_size, seed=42
    )

    # Collect test conditioning (denormalized to physical units)
    all_cond_norm = []
    for _, cond in test_loader:
        all_cond_norm.append(cond)
        if sum(c.shape[0] for c in all_cond_norm) >= args.n_samples:
            break
    all_cond_norm = torch.cat(all_cond_norm)[:args.n_samples]
    all_cond_raw = dataset.denormalize_cond(all_cond_norm).numpy()

    # Target properties in physical units
    targets = {
        'height': all_cond_raw[:, 0],
        'radius': all_cond_raw[:, 1],
        'aspect_ratio': all_cond_raw[:, 2],
        'angle_deg': all_cond_raw[:, 3],
        'ntg': all_cond_raw[:, 4],
    }

    n_samples = len(all_cond_norm)
    print(f"Evaluating on {n_samples} test samples, {args.n_steps} steps, cfg={args.cfg_scale}")

    # Load CNN evaluator
    predictor = LobePropertyPredictor(args.cnn_weights, args.data_dir, device)
    print("CNN property predictor loaded")

    # Load generative methods
    checkpoints = {
        'diffusion': 'checkpoints/diffusion.pt',
        'flow_matching': 'checkpoints/flow_matching.pt',
        'rectified_flow': 'checkpoints/rectified_flow.pt',
        'rectified_flow_bwd': 'checkpoints/rectified_flow_bwd.pt',
        'rectified_flow_bidir': 'checkpoints/rectified_flow_bidir.pt',
        'rectified_flow_rand': 'checkpoints/rectified_flow_rand.pt',
        'meanflow_std': 'checkpoints/meanflow_std.pt',
        'meanflow_embed': 'checkpoints/meanflow_embed.pt',
    }

    methods = {}
    for name, path in checkpoints.items():
        if os.path.exists(path):
            print(f"Loading {name}")
            methods[name] = load_method(name, path, device)

    if not methods:
        print("No checkpoints found in 'checkpoints/'. Run train.py first.")
        return

    os.makedirs("results", exist_ok=True)

    # Evaluate each method
    for method_name, method in methods.items():
        print(f"\n=== {method_name} ===")
        t0 = time.time()

        all_pred = {k: [] for k in ['height', 'radius', 'aspect_ratio', 'angle_deg']}
        all_ntg = []

        for i in range(0, n_samples, args.batch_size):
            batch_end = min(i + args.batch_size, n_samples)
            batch_cond = all_cond_norm[i:batch_end].to(device)
            batch_size = batch_cond.shape[0]
            shape = (batch_size, 1, 50, 50, 50)

            # Generate
            samples = method.sample(shape, device, cond=batch_cond,
                                    cfg_scale=args.cfg_scale, n_steps=args.n_steps)
            binary = (samples > 0).float()

            # Predict properties
            pred = predictor.predict(binary)
            for k in all_pred:
                all_pred[k].append(pred[k])
            all_ntg.append(predictor.compute_ntg(binary))

            print(f"  Batch {i//args.batch_size + 1}/{(n_samples + args.batch_size - 1)//args.batch_size}")

        elapsed = time.time() - t0

        # Concatenate
        for k in all_pred:
            all_pred[k] = np.concatenate(all_pred[k])
        all_ntg = np.concatenate(all_ntg)

        # Resolve angle ambiguity
        all_pred['angle_deg'] = resolve_angle_ambiguity(all_pred['angle_deg'], targets['angle_deg'])

        # Compute metrics
        print(f"\nResults ({elapsed:.1f}s):")
        print(f"  {'Property':<15} {'RMSE':>10} {'R²':>10}")
        print(f"  {'-'*37}")
        for key in ['height', 'radius', 'aspect_ratio', 'angle_deg']:
            rmse = compute_rmse(all_pred[key], targets[key])
            r2 = compute_r2(all_pred[key], targets[key])
            print(f"  {key:<15} {rmse:>10.4f} {r2:>10.4f}")

        ntg_rmse = compute_rmse(all_ntg, targets['ntg'])
        ntg_r2 = compute_r2(all_ntg, targets['ntg'])
        print(f"  {'ntg (voxel)':<15} {ntg_rmse:>10.4f} {ntg_r2:>10.4f}")

        # Save data
        np.savez(
            f"results/eval_{method_name}_{args.n_steps}steps.npz",
            pred_height=all_pred['height'],
            pred_radius=all_pred['radius'],
            pred_aspect_ratio=all_pred['aspect_ratio'],
            pred_angle_deg=all_pred['angle_deg'],
            pred_ntg=all_ntg,
            target_height=targets['height'],
            target_radius=targets['radius'],
            target_aspect_ratio=targets['aspect_ratio'],
            target_angle_deg=targets['angle_deg'],
            target_ntg=targets['ntg'],
        )

        # Parity plot
        plot_predictions = {**all_pred, 'ntg': all_ntg}
        plot_parity(
            plot_predictions, targets,
            save_path=f"results/parity_{method_name}_{args.n_steps}steps.png",
            title_prefix=method_name,
        )

    print("\nDone! Results saved to 'results/'.")


if __name__ == "__main__":
    main()
