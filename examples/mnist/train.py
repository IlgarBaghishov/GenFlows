import sys
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator

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
    accelerator = Accelerator()
    device = accelerator.device
    accelerator.print(f"Using device: {device}")

    # Download data on main process first to avoid race conditions
    if accelerator.is_main_process:
        get_mnist_loaders(batch_size=128)
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("results", exist_ok=True)
    accelerator.wait_for_everyone()

    train_loader, _ = get_mnist_loaders(batch_size=128)

    # Per-method epoch counts tuned for MNIST
    epochs_ddpm = 350
    epochs_fm = 350
    epochs_rf = 200
    epochs_mf = 800

    # --- 1. Diffusion (DDPM) ---
    accelerator.print("\n--- Training Diffusion (DDPM) ---")
    model_diff = UNet(in_channels=1, num_time_embs=1).to(device)
    method_diff = Diffusion(model_diff, n_steps=1000)
    loss_diff = train_model(method_diff, train_loader, epochs=epochs_ddpm, accelerator=accelerator)
    if accelerator.is_main_process:
        torch.save(method_diff.model.state_dict(), "checkpoints/diffusion.pt")
        plot_loss(loss_diff, "Diffusion Training Loss", "results/loss_diffusion.png")

    # --- 2. Flow Matching ---
    accelerator.print("\n--- Training Flow Matching ---")
    model_fm = UNet(in_channels=1, num_time_embs=1).to(device)
    method_fm = FlowMatching(model_fm)
    loss_fm = train_model(method_fm, train_loader, epochs=epochs_fm, accelerator=accelerator)
    if accelerator.is_main_process:
        torch.save(method_fm.model.state_dict(), "checkpoints/flow_matching.pt")
        plot_loss(loss_fm, "Flow Matching Training Loss", "results/loss_flow_matching.png")

    # --- 3. Generate Reflow Pairs (distributed across all GPUs) ---
    reflow_generator = RectifiedFlow(method_fm.model)
    reflow_generator.model.to(device)
    pair_loader = accelerator.prepare(
        DataLoader(train_loader.dataset, batch_size=128, shuffle=False)
    )
    silent = not accelerator.is_main_process

    accelerator.print("\n--- Generating Forward Reflow Pairs from Flow Matching model ---")
    local_fwd = reflow_generator.generate_reflow_pairs(pair_loader, device, n_steps=100, silent=silent)
    accelerator.wait_for_everyone()
    accelerator.print("\n--- Generating Backward Reflow Pairs from Flow Matching model ---")
    local_bwd = reflow_generator.generate_reflow_pairs_backward(pair_loader, device, n_steps=100, silent=silent)

    # Gather pairs from all ranks
    def gather_pairs(local_dataset):
        return TensorDataset(*[accelerator.gather(t.to(device)).cpu() for t in local_dataset.tensors])

    paired_fwd = gather_pairs(local_fwd)
    paired_bwd = gather_pairs(local_bwd)
    paired_bidir = TensorDataset(
        torch.cat([paired_fwd.tensors[0], paired_bwd.tensors[0]]),
        torch.cat([paired_fwd.tensors[1], paired_bwd.tensors[1]]),
        torch.cat([paired_fwd.tensors[2], paired_bwd.tensors[2]]),
    )
    accelerator.print(f"  Forward: {len(paired_fwd)}, Backward: {len(paired_bwd)}, Total: {len(paired_bidir)}")

    fm_weights = method_fm.model.state_dict()

    # --- 3a. Rectified Flow Forward (random init, 300 epochs) ---
    accelerator.print("\n--- Training Rectified Flow Forward - Random Init (2-Rectified Flow) ---")
    model_rf_rand = UNet(in_channels=1, num_time_embs=1).to(device)
    method_rf_rand = RectifiedFlow(model_rf_rand)
    loss_rf_rand = train_reflow(method_rf_rand, paired_fwd, epochs=epochs_fm, accelerator=accelerator)
    if accelerator.is_main_process:
        torch.save(method_rf_rand.model.state_dict(), "checkpoints/rectified_flow_rand.pt")
        plot_loss(loss_rf_rand, "Rectified Flow Fwd (Random Init) Training Loss", "results/loss_rectified_flow_rand.png")

    # --- 3b. Rectified Flow Forward (warm-started from FM) ---
    accelerator.print("\n--- Training Rectified Flow Forward - Warm Start (2-Rectified Flow) ---")
    model_rf = UNet(in_channels=1, num_time_embs=1).to(device)
    model_rf.load_state_dict(fm_weights)
    method_rf = RectifiedFlow(model_rf)
    loss_rf = train_reflow(method_rf, paired_fwd, epochs=epochs_rf, accelerator=accelerator)
    if accelerator.is_main_process:
        torch.save(method_rf.model.state_dict(), "checkpoints/rectified_flow.pt")
        plot_loss(loss_rf, "Rectified Flow Fwd (Warm Start) Training Loss", "results/loss_rectified_flow.png")

    # --- 3c. Rectified Flow Backward (warm-started from FM) ---
    accelerator.print("\n--- Training Rectified Flow Backward - Warm Start (2-Rectified Flow) ---")
    model_rf_bwd = UNet(in_channels=1, num_time_embs=1).to(device)
    model_rf_bwd.load_state_dict(fm_weights)
    method_rf_bwd = RectifiedFlow(model_rf_bwd)
    loss_rf_bwd = train_reflow(method_rf_bwd, paired_bwd, epochs=epochs_rf, accelerator=accelerator)
    if accelerator.is_main_process:
        torch.save(method_rf_bwd.model.state_dict(), "checkpoints/rectified_flow_bwd.pt")
        plot_loss(loss_rf_bwd, "Rectified Flow Bwd (Warm Start) Training Loss", "results/loss_rectified_flow_bwd.png")

    # --- 3d. Rectified Flow Bidirectional (warm-started from FM) ---
    accelerator.print("\n--- Training Rectified Flow Bidirectional - Warm Start (2-Rectified Flow) ---")
    model_rf_bidir = UNet(in_channels=1, num_time_embs=1).to(device)
    model_rf_bidir.load_state_dict(fm_weights)
    method_rf_bidir = RectifiedFlow(model_rf_bidir)
    loss_rf_bidir = train_reflow(method_rf_bidir, paired_bidir, epochs=epochs_rf, accelerator=accelerator)
    if accelerator.is_main_process:
        torch.save(method_rf_bidir.model.state_dict(), "checkpoints/rectified_flow_bidir.pt")
        plot_loss(loss_rf_bidir, "Rectified Flow Bidir (Warm Start) Training Loss", "results/loss_rectified_flow_bidir.png")

    # --- 4. MeanFlow (Standard CFG) ---
    accelerator.print("\n--- Training MeanFlow (Standard CFG) ---")
    model_mf = UNet(in_channels=1, num_time_embs=2).to(device)
    method_mf = MeanFlow(model_mf, cfg_mode='standard')
    loss_mf = train_model(method_mf, train_loader, epochs=epochs_mf, accelerator=accelerator)
    if accelerator.is_main_process:
        torch.save(method_mf.model.state_dict(), "checkpoints/meanflow_std.pt")
        plot_loss(loss_mf, "MeanFlow (Std CFG) Training Loss", "results/loss_meanflow.png")

    # --- 5. MeanFlow (Embedded CFG) ---
    accelerator.print("\n--- Training MeanFlow (Embedded CFG) ---")
    model_mf_cfg = UNet(in_channels=1, num_time_embs=2).to(device)
    method_mf_cfg = MeanFlow(model_mf_cfg, cfg_mode='embedded', omega=3.0, kappa=0.0)
    loss_mf_cfg = train_model(method_mf_cfg, train_loader, epochs=epochs_mf, accelerator=accelerator)
    if accelerator.is_main_process:
        torch.save(method_mf_cfg.model.state_dict(), "checkpoints/meanflow_embed.pt")
        plot_loss(loss_mf_cfg, "MeanFlow (Embed CFG) Training Loss", "results/loss_meanflow_embedded_cfg.png")

    accelerator.print("\nDone! Checkpoints saved to 'checkpoints/', loss curves to 'results/'.")
    accelerator.wait_for_everyone()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
