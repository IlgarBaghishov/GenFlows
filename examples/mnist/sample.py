import os
import time
import torch

from genflows.models.unet import UNet
from genflows.methods.diffusion import Diffusion
from genflows.methods.flow_matching import FlowMatching
from genflows.methods.meanflow import MeanFlow
from genflows.methods.rectified_flow import RectifiedFlow
from genflows.utils.plotting import plot_samples


def make_digit_labels(n_per_digit=10, device='cpu'):
    """Create labels for a 10x10 grid: each row is one digit (0-9), 10 samples per digit."""
    return torch.arange(10, device=device).repeat_interleave(n_per_digit)


def load_method(name, ckpt_path, device):
    """Load a method wrapper with pretrained weights."""
    if name == 'diffusion':
        model = UNet(in_channels=1, num_time_embs=1).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        return Diffusion(model, n_steps=1000)
    elif name == 'flow_matching':
        model = UNet(in_channels=1, num_time_embs=1).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        return FlowMatching(model)
    elif name in ('rectified_flow', 'rectified_flow_bwd', 'rectified_flow_bidir', 'rectified_flow_rand'):
        model = UNet(in_channels=1, num_time_embs=1).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        return RectifiedFlow(model)
    elif name == 'meanflow_std':
        model = UNet(in_channels=1, num_time_embs=2).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        return MeanFlow(model, cfg_mode='standard')
    elif name == 'meanflow_embed':
        model = UNet(in_channels=1, num_time_embs=2).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        return MeanFlow(model, cfg_mode='embedded', omega=3.0, kappa=0.0)
    else:
        raise ValueError(f"Unknown method: {name}")


def timed_sample(method, shape, device, labels, n_steps, **kwargs):
    """Sample and return (samples, elapsed_seconds)."""
    t0 = time.time()
    samples = method.sample(shape, device, cond=labels, n_steps=n_steps, **kwargs)
    elapsed = time.time() - t0
    return samples, elapsed


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs("results", exist_ok=True)

    # Load all methods
    methods = {}
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

    for name, path in checkpoints.items():
        if os.path.exists(path):
            print(f"Loading {name} from {path}")
            methods[name] = load_method(name, path, device)
        else:
            print(f"WARNING: {path} not found, skipping {name}")

    if not methods:
        print("No checkpoints found in 'checkpoints/'. Run train.py first.")
        return

    # Sampling config
    n_per_digit = 10
    n_samples = 10 * n_per_digit
    shape = (n_samples, 1, 32, 32)
    labels = make_digit_labels(n_per_digit, device=device)
    step_counts = [1, 5, 10, 50, 100, 500, 1000]

    for n_steps in step_counts:
        if 'diffusion' in methods:
            samples, t = timed_sample(methods['diffusion'], shape, device, labels, n_steps, sampler='ddpm')
            print(f"DDPM  {n_steps:>4} steps: {t:.2f}s")
            plot_samples(samples, f"DDPM Samples ({n_steps} steps)", f"results/samples_ddpm_{n_steps}steps.png", nrow=n_per_digit)

            samples, t = timed_sample(methods['diffusion'], shape, device, labels, n_steps, sampler='ddim', eta=0.0)
            print(f"DDIM  {n_steps:>4} steps: {t:.2f}s")
            plot_samples(samples, f"DDIM Samples ({n_steps} steps)", f"results/samples_ddim_{n_steps}steps.png", nrow=n_per_digit)

        if 'flow_matching' in methods:
            samples, t = timed_sample(methods['flow_matching'], shape, device, labels, n_steps)
            print(f"FM    {n_steps:>4} steps: {t:.2f}s")
            plot_samples(samples, f"Flow Matching Samples ({n_steps} steps)", f"results/samples_flow_matching_{n_steps}steps.png", nrow=n_per_digit)

        if 'rectified_flow' in methods:
            samples, t = timed_sample(methods['rectified_flow'], shape, device, labels, n_steps)
            print(f"RF    {n_steps:>4} steps: {t:.2f}s")
            plot_samples(samples, f"Rectified Flow Samples ({n_steps} steps)", f"results/samples_rectified_flow_{n_steps}steps.png", nrow=n_per_digit)

        if 'rectified_flow_bwd' in methods:
            samples, t = timed_sample(methods['rectified_flow_bwd'], shape, device, labels, n_steps)
            print(f"RF-B  {n_steps:>4} steps: {t:.2f}s")
            plot_samples(samples, f"Rectified Flow Backward ({n_steps} steps)", f"results/samples_rectified_flow_bwd_{n_steps}steps.png", nrow=n_per_digit)

        if 'rectified_flow_bidir' in methods:
            samples, t = timed_sample(methods['rectified_flow_bidir'], shape, device, labels, n_steps)
            print(f"RF-BD {n_steps:>4} steps: {t:.2f}s")
            plot_samples(samples, f"Rectified Flow Bidir ({n_steps} steps)", f"results/samples_rectified_flow_bidir_{n_steps}steps.png", nrow=n_per_digit)

        if 'rectified_flow_rand' in methods:
            samples, t = timed_sample(methods['rectified_flow_rand'], shape, device, labels, n_steps)
            print(f"RF-R  {n_steps:>4} steps: {t:.2f}s")
            plot_samples(samples, f"Rectified Flow Rand ({n_steps} steps)", f"results/samples_rectified_flow_rand_{n_steps}steps.png", nrow=n_per_digit)

        if 'meanflow_std' in methods:
            samples, t = timed_sample(methods['meanflow_std'], shape, device, labels, n_steps)
            print(f"MF-S  {n_steps:>4} steps: {t:.2f}s")
            plot_samples(samples, f"MeanFlow Std CFG ({n_steps} steps)", f"results/samples_meanflow_{n_steps}steps.png", nrow=n_per_digit)

        if 'meanflow_embed' in methods:
            samples, t = timed_sample(methods['meanflow_embed'], shape, device, labels, n_steps)
            print(f"MF-E  {n_steps:>4} steps: {t:.2f}s")
            plot_samples(samples, f"MeanFlow Embed CFG ({n_steps} steps)", f"results/samples_meanflow_embedded_cfg_{n_steps}steps.png", nrow=n_per_digit)

        print()

    print("Done! Check the 'results' directory for output images.")


if __name__ == "__main__":
    main()
