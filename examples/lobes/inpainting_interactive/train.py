import os
import shutil
import numpy as np
import torch
from accelerate import Accelerator

from resflow.models.unet3d import UNet3D
from resflow.methods.flow_matching import FlowMatching
from resflow.utils.data_lobes import get_lobe_inpaint_loaders
from resflow.utils.plotting import plot_loss
from resflow.utils.training import train_model_inpaint

# --- Configuration ---
EPOCHS_PER_RUN = 25
TOTAL_EPOCHS = 350  # LR scheduler horizon (warmup 17 epochs + cosine decay 333 epochs)
SAVE_EVERY = 25
CHECKPOINT_DIR = 'checkpoints'
RESULTS_DIR = 'results'


def main():
    accelerator = Accelerator()
    device = accelerator.device
    accelerator.print(f"Using device: {device}")

    # Load data on main process first (computes NTG, filters)
    if accelerator.is_main_process:
        get_lobe_inpaint_loaders(data_dir='../data', batch_size=32)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
    accelerator.wait_for_everyone()

    train_loader, val_loader, test_loader, dataset = get_lobe_inpaint_loaders(
        data_dir='../data', batch_size=32
    )
    accelerator.print(f"Dataset: {len(dataset)} samples after filtering")
    accelerator.print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Save dataset normalization stats for sampling
    if accelerator.is_main_process:
        np.savez(os.path.join(CHECKPOINT_DIR, 'cond_stats.npz'),
                 cond_min=dataset.cond_min, cond_max=dataset.cond_max)

    # --- Flow Matching Inpainting ---
    accelerator.print(f"\n--- Training Flow Matching Inpaint ({EPOCHS_PER_RUN} epochs) ---")
    model_fm = UNet3D(in_channels=3, out_channels=1, num_time_embs=1).to(device)
    method_fm = FlowMatching(model_fm)

    loss_history = train_model_inpaint(
        method_fm, train_loader, epochs=EPOCHS_PER_RUN,
        checkpoint_dir=CHECKPOINT_DIR, save_every=SAVE_EVERY,
        total_epochs=TOTAL_EPOCHS, accelerator=accelerator,
    )

    total_epochs = len(loss_history)
    accelerator.print(f"\nTraining complete. Total epochs: {total_epochs}, "
                      f"final loss: {loss_history[-1]:.4f}")

    if accelerator.is_main_process:
        # Save full loss history
        np.save(os.path.join(CHECKPOINT_DIR, 'loss_history.npy'), np.array(loss_history))

        # Copy inference checkpoint with method name for sample.py
        src = os.path.join(CHECKPOINT_DIR, f'inference_epoch{total_epochs:03d}.pt')
        dst = os.path.join(CHECKPOINT_DIR, f'flow_matching_epoch{total_epochs:03d}.pt')
        if os.path.exists(src):
            shutil.copy2(src, dst)

        # Also save as the "latest" checkpoint for convenience
        shutil.copy2(src, os.path.join(CHECKPOINT_DIR, 'flow_matching.pt'))

        # Plot full loss curve
        plot_loss(loss_history, "Flow Matching Inpaint Training Loss",
                  os.path.join(RESULTS_DIR, 'loss_curve.png'))
        accelerator.print(f"Loss curve saved to {RESULTS_DIR}/loss_curve.png")

    accelerator.wait_for_everyone()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
