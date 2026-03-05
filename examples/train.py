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
from genflows.utils.plotting import plot_loss
from genflows.utils.training import train_model, train_reflow


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, _ = get_mnist_loaders(batch_size=128)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Per-method epoch counts tuned for MNIST
    epochs_ddpm = 400
    epochs_fm = 300
    epochs_rf = 100
    epochs_mf = 600

    # --- 1. Diffusion (DDPM) ---
    print("\n--- Training Diffusion (DDPM) ---")
    model_diff = UNet(in_channels=1, num_time_embs=1).to(device)
    method_diff = Diffusion(model_diff, n_steps=1000)
    loss_diff = train_model(method_diff, train_loader, epochs=epochs_ddpm, device=device)
    torch.save(model_diff.state_dict(), "checkpoints/diffusion.pt")
    plot_loss(loss_diff, "Diffusion Training Loss", "results/loss_diffusion.png")

    # --- 2. Flow Matching ---
    print("\n--- Training Flow Matching ---")
    model_fm = UNet(in_channels=1, num_time_embs=1).to(device)
    method_fm = FlowMatching(model_fm)
    loss_fm = train_model(method_fm, train_loader, epochs=epochs_fm, device=device)
    torch.save(model_fm.state_dict(), "checkpoints/flow_matching.pt")
    plot_loss(loss_fm, "Flow Matching Training Loss", "results/loss_flow_matching.png")

    # --- 3. Rectified Flow (2-Rectified Flow) ---
    print("\n--- Generating Reflow Pairs from Flow Matching model ---")
    reflow_generator = RectifiedFlow(model_fm)
    paired_dataset = reflow_generator.generate_reflow_pairs(train_loader, device, n_steps=100)
    print("\n--- Training Rectified Flow (2-Rectified Flow) ---")
    model_rf = UNet(in_channels=1, num_time_embs=1).to(device)
    method_rf = RectifiedFlow(model_rf)
    loss_rf = train_reflow(method_rf, paired_dataset, epochs=epochs_rf, device=device)
    torch.save(model_rf.state_dict(), "checkpoints/rectified_flow.pt")
    plot_loss(loss_rf, "Rectified Flow Training Loss", "results/loss_rectified_flow.png")

    # --- 4. MeanFlow (Standard CFG) ---
    print("\n--- Training MeanFlow (Standard CFG) ---")
    model_mf = UNet(in_channels=1, num_time_embs=2).to(device)
    method_mf = MeanFlow(model_mf, cfg_mode='standard')
    loss_mf = train_model(method_mf, train_loader, epochs=epochs_mf, device=device)
    torch.save(model_mf.state_dict(), "checkpoints/meanflow_std.pt")
    plot_loss(loss_mf, "MeanFlow (Std CFG) Training Loss", "results/loss_meanflow.png")

    # --- 5. MeanFlow (Embedded CFG) ---
    print("\n--- Training MeanFlow (Embedded CFG) ---")
    model_mf_cfg = UNet(in_channels=1, num_time_embs=2).to(device)
    method_mf_cfg = MeanFlow(model_mf_cfg, cfg_mode='embedded', omega=3.0, kappa=0.0)
    loss_mf_cfg = train_model(method_mf_cfg, train_loader, epochs=epochs_mf, device=device)
    torch.save(model_mf_cfg.state_dict(), "checkpoints/meanflow_embed.pt")
    plot_loss(loss_mf_cfg, "MeanFlow (Embed CFG) Training Loss", "results/loss_meanflow_embedded_cfg.png")

    print("\nDone! Checkpoints saved to 'checkpoints/', loss curves to 'results/'.")


if __name__ == "__main__":
    main()
