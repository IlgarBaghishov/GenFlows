import sys
import os
import torch

# Add package root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genflows.models.unet import UNet
from genflows.methods.diffusion import Diffusion
from genflows.methods.flow_matching import FlowMatching
from genflows.methods.meanflow import MeanFlow
from genflows.methods.rectified_flow import RectifiedFlow
from genflows.utils.data import get_mnist_loaders
from genflows.utils.plotting import plot_samples, plot_loss
from genflows.utils.training import train_model, train_reflow


def make_digit_labels(n_per_digit=10, device='cpu'):
    """Create labels for a 10x10 grid: each row is one digit (0-9), 10 samples per digit."""
    labels = torch.arange(10, device=device).repeat_interleave(n_per_digit)
    return labels


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, _ = get_mnist_loaders(batch_size=128)

    # Per-method epoch counts tuned for MNIST
    epochs_ddpm = 400   # Slow converger — learns noise prediction across 1000 timesteps
    epochs_fm = 300     # Simpler velocity objective; also serves as base for reflow pairs
    epochs_rf = 100     # Least needed — learning from coupled pairs is easier
    epochs_mf = 600     # JVP-based training converges slower, needs more epochs

    # --- 1. Diffusion (DDPM) ---
    print("\n--- Training Diffusion (DDPM) ---")
    model_diff = UNet(in_channels=1, num_time_embs=1).to(device)
    method_diff = Diffusion(model_diff, n_steps=1000)
    loss_diff = train_model(method_diff, train_loader, epochs=epochs_ddpm, device=device)

    # --- 2. Flow Matching ---
    print("\n--- Training Flow Matching ---")
    model_fm = UNet(in_channels=1, num_time_embs=1).to(device)
    method_fm = FlowMatching(model_fm)
    loss_fm = train_model(method_fm, train_loader, epochs=epochs_fm, device=device)

    # --- 3. Rectified Flow (2-Rectified Flow) ---
    print("\n--- Generating Reflow Pairs from Flow Matching model ---")
    # Use a RectifiedFlow wrapper around the trained FM model to generate pairs
    reflow_generator = RectifiedFlow(model_fm)
    paired_dataset = reflow_generator.generate_reflow_pairs(train_loader, device, n_steps=100)
    print("\n--- Training Rectified Flow (2-Rectified Flow) ---")
    model_rf = UNet(in_channels=1, num_time_embs=1).to(device)
    method_rf = RectifiedFlow(model_rf)
    loss_rf = train_reflow(method_rf, paired_dataset, epochs=epochs_rf, device=device)

    # --- 4. MeanFlow (Standard CFG) ---
    print("\n--- Training MeanFlow (Standard CFG) ---")
    model_mf = UNet(in_channels=1, num_time_embs=2).to(device)
    method_mf = MeanFlow(model_mf, cfg_mode='standard')
    loss_mf = train_model(method_mf, train_loader, epochs=epochs_mf, device=device)

    # --- 5. MeanFlow (Embedded CFG) ---
    print("\n--- Training MeanFlow (Embedded CFG) ---")
    model_mf_cfg = UNet(in_channels=1, num_time_embs=2).to(device)
    method_mf_cfg = MeanFlow(model_mf_cfg, cfg_mode='embedded', omega=3.0, kappa=0.0)
    loss_mf_cfg = train_model(method_mf_cfg, train_loader, epochs=epochs_mf, device=device)

    # --- Generate Plots & Comparisons ---
    print("\n--- Generating Plots ---")
    plot_loss(loss_diff, "Diffusion Training Loss", "results/loss_diffusion.png")
    plot_loss(loss_fm, "Flow Matching Training Loss", "results/loss_flow_matching.png")
    plot_loss(loss_rf, "Rectified Flow Training Loss", "results/loss_rectified_flow.png")
    plot_loss(loss_mf, "MeanFlow (Std CFG) Training Loss", "results/loss_meanflow.png")
    plot_loss(loss_mf_cfg, "MeanFlow (Embed CFG) Training Loss", "results/loss_meanflow_embedded_cfg.png")

    # 10x10 grid: rows are digits 0-9, 10 samples per digit
    n_per_digit = 10
    n_samples = 10 * n_per_digit  # 100
    shape = (n_samples, 1, 32, 32)
    labels = make_digit_labels(n_per_digit, device=device)

    for n_steps in [1, 5, 10, 50, 100, 500, 1000]:
        print(f"Sampling DDPM ({n_steps} steps)...")
        samples_diff = method_diff.sample(shape, device, labels=labels, n_steps=n_steps, sampler='ddpm')
        plot_samples(samples_diff, f"DDPM Samples ({n_steps} steps)", f"results/samples_ddpm_{n_steps}steps.png", nrow=n_per_digit)

        print(f"Sampling DDIM ({n_steps} steps)...")
        samples_ddim = method_diff.sample(shape, device, labels=labels, n_steps=n_steps, sampler='ddim', eta=0.0)
        plot_samples(samples_ddim, f"DDIM Samples ({n_steps} steps)", f"results/samples_ddim_{n_steps}steps.png", nrow=n_per_digit)

        print(f"Sampling Flow Matching ({n_steps} steps)...")
        samples_fm = method_fm.sample(shape, device, labels=labels, n_steps=n_steps)
        plot_samples(samples_fm, f"Flow Matching Samples ({n_steps} steps)", f"results/samples_flow_matching_{n_steps}steps.png", nrow=n_per_digit)

        print(f"Sampling Rectified Flow ({n_steps} steps)...")
        samples_rf = method_rf.sample(shape, device, labels=labels, n_steps=n_steps)
        plot_samples(samples_rf, f"Rectified Flow Samples ({n_steps} steps)", f"results/samples_rectified_flow_{n_steps}steps.png", nrow=n_per_digit)

        print(f"Sampling MeanFlow Std CFG ({n_steps} steps)...")
        samples_mf = method_mf.sample(shape, device, labels=labels, n_steps=n_steps)
        plot_samples(samples_mf, f"MeanFlow Std CFG ({n_steps} steps)", f"results/samples_meanflow_{n_steps}steps.png", nrow=n_per_digit)

        print(f"Sampling MeanFlow Embed CFG ({n_steps} steps)...")
        samples_mf_cfg = method_mf_cfg.sample(shape, device, labels=labels, n_steps=n_steps)
        plot_samples(samples_mf_cfg, f"MeanFlow Embed CFG ({n_steps} steps)", f"results/samples_meanflow_embedded_cfg_{n_steps}steps.png", nrow=n_per_digit)

    print("Done! Check the 'results' directory for output images.")

if __name__ == "__main__":
    main()
