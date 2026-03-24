import os
import torch
from accelerate import Accelerator

from genflows.models.unet3d import UNet3D
from genflows.methods.diffusion import Diffusion
from genflows.methods.flow_matching import FlowMatching
from genflows.methods.meanflow import MeanFlow
from genflows.methods.rectified_flow import RectifiedFlow
from genflows.utils.data_lobes import get_lobe_inpaint_loaders
from genflows.utils.plotting import plot_loss
from genflows.utils.training import train_model_inpaint


def main():
    accelerator = Accelerator()
    device = accelerator.device
    accelerator.print(f"Using device: {device}")

    # Load data on main process first (computes NTG, filters)
    if accelerator.is_main_process:
        get_lobe_inpaint_loaders(data_dir='data', batch_size=32)
        os.makedirs("checkpoints_inpaint", exist_ok=True)
        os.makedirs("results_inpaint", exist_ok=True)
    accelerator.wait_for_everyone()

    train_loader, val_loader, test_loader, dataset = get_lobe_inpaint_loaders(
        data_dir='data', batch_size=32
    )
    accelerator.print(f"Dataset: {len(dataset)} samples after filtering")
    accelerator.print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    accelerator.print(f"  Cond min: {dataset.cond_min}")
    accelerator.print(f"  Cond max: {dataset.cond_max}")

    # Save dataset normalization stats for sampling
    if accelerator.is_main_process:
        import numpy as np
        np.savez('checkpoints_inpaint/cond_stats.npz',
                 cond_min=dataset.cond_min, cond_max=dataset.cond_max)

    # Per-method epoch counts
    epochs_ddpm = 350
    epochs_fm = 350
    epochs_rf = 200
    epochs_mf = 800

    # # --- 1. Diffusion (DDPM) ---
    # accelerator.print("\n--- Training Diffusion Inpaint (DDPM) ---")
    # model_diff = UNet3D(in_channels=3, out_channels=1, num_time_embs=1).to(device)
    # method_diff = Diffusion(model_diff, n_steps=1000)
    # loss_diff = train_model_inpaint(method_diff, train_loader, epochs=epochs_ddpm, accelerator=accelerator)
    # if accelerator.is_main_process:
    #     torch.save(method_diff.model.state_dict(), "checkpoints_inpaint/diffusion_inpaint.pt")
    #     plot_loss(loss_diff, "Diffusion Inpaint Training Loss", "results_inpaint/loss_diffusion_inpaint.png")

    # --- 2. Flow Matching ---
    accelerator.print("\n--- Training Flow Matching Inpaint ---")
    model_fm = UNet3D(in_channels=3, out_channels=1, num_time_embs=1).to(device)
    method_fm = FlowMatching(model_fm)
    loss_fm = train_model_inpaint(method_fm, train_loader, epochs=epochs_fm, accelerator=accelerator)
    if accelerator.is_main_process:
        torch.save(method_fm.model.state_dict(), "checkpoints_inpaint/flow_matching_inpaint.pt")
        plot_loss(loss_fm, "Flow Matching Inpaint Training Loss", "results_inpaint/loss_flow_matching_inpaint.png")

    # # --- 3. Rectified Flow (round 1 only — reflow pairs with inpaint deferred) ---
    # accelerator.print("\n--- Training Rectified Flow Inpaint ---")
    # model_rf = UNet3D(in_channels=3, out_channels=1, num_time_embs=1).to(device)
    # method_rf = RectifiedFlow(model_rf)
    # loss_rf = train_model_inpaint(method_rf, train_loader, epochs=epochs_fm, accelerator=accelerator)
    # if accelerator.is_main_process:
    #     torch.save(method_rf.model.state_dict(), "checkpoints_inpaint/rectified_flow_inpaint.pt")
    #     plot_loss(loss_rf, "Rectified Flow Inpaint Training Loss", "results_inpaint/loss_rectified_flow_inpaint.png")

    # # --- 4. MeanFlow (Standard CFG) ---
    # accelerator.print("\n--- Training MeanFlow Inpaint (Standard CFG) ---")
    # model_mf = UNet3D(in_channels=3, out_channels=1, num_time_embs=2).to(device)
    # method_mf = MeanFlow(model_mf, cfg_mode='standard')
    # loss_mf = train_model_inpaint(method_mf, train_loader, epochs=epochs_mf, accelerator=accelerator)
    # if accelerator.is_main_process:
    #     torch.save(method_mf.model.state_dict(), "checkpoints_inpaint/meanflow_std_inpaint.pt")
    #     plot_loss(loss_mf, "MeanFlow Inpaint (Std CFG) Training Loss", "results_inpaint/loss_meanflow_inpaint.png")

    # # --- 5. MeanFlow (Embedded CFG) ---
    # accelerator.print("\n--- Training MeanFlow Inpaint (Embedded CFG) ---")
    # model_mf_cfg = UNet3D(in_channels=3, out_channels=1, num_time_embs=2).to(device)
    # method_mf_cfg = MeanFlow(model_mf_cfg, cfg_mode='embedded', omega=3.0, kappa=0.0)
    # loss_mf_cfg = train_model_inpaint(method_mf_cfg, train_loader, epochs=epochs_mf, accelerator=accelerator)
    # if accelerator.is_main_process:
    #     torch.save(method_mf_cfg.model.state_dict(), "checkpoints_inpaint/meanflow_embed_inpaint.pt")
    #     plot_loss(loss_mf_cfg, "MeanFlow Inpaint (Embed CFG) Training Loss", "results_inpaint/loss_meanflow_embedded_cfg_inpaint.png")

    accelerator.print("\nDone! Checkpoints saved to 'checkpoints_inpaint/', loss curves to 'results_inpaint/'.")
    accelerator.wait_for_everyone()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
