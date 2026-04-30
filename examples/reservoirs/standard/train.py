"""Train conditional Flow Matching on SciLM-ai/SiliciclasticReservoirs (1M cubes).

Mirrors examples/lobes/standard/train.py. Uses the binary facies version of the
HuggingFace dataset, with conditioning built from params_slim.parquet (caption /
poro_ave / perm_ave excluded — see data_reservoirs.py).

Default DATA_DIR points to $SCRATCH because the dataset is large; override via
the RESERVOIR_DATA_DIR env var.
"""
import os

import numpy as np
import torch
from accelerate import Accelerator

from genflows.models.unet3d import UNet3D
from genflows.methods.flow_matching import FlowMatching
from genflows.utils.data_reservoirs import (
    get_reservoir_loaders, COND_DIM, LAYER_TYPES,
)
from genflows.utils.plotting import plot_loss
from genflows.utils.training import train_model


# Dataset lives on $SCRATCH because of its size (~131 GB after our subset filter).
DEFAULT_DATA_DIR = os.environ.get(
    'RESERVOIR_DATA_DIR',
    '/scratch/11316/rustamzade17/SiliciclasticReservoirs',
)


def main():
    accelerator = Accelerator()
    device = accelerator.device
    accelerator.print(f"Using device: {device}")
    accelerator.print(f"Data dir: {DEFAULT_DATA_DIR}")

    # Build the cond cache on the main process before workers fork.
    if accelerator.is_main_process:
        get_reservoir_loaders(data_dir=DEFAULT_DATA_DIR, batch_size=32, num_workers=0)
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("results", exist_ok=True)
    accelerator.wait_for_everyone()

    train_loader, val_loader, test_loader, dataset = get_reservoir_loaders(
        data_dir=DEFAULT_DATA_DIR, batch_size=32, num_workers=4,
    )
    accelerator.print(f"Dataset: {len(dataset)} train / "
                      f"{len(val_loader.dataset)} val / "
                      f"{len(test_loader.dataset)} test")
    accelerator.print(f"Cond dim: {COND_DIM}  Layer types: {LAYER_TYPES}")
    accelerator.print(f"  cont_min: {dataset.cont_min}")
    accelerator.print(f"  cont_max: {dataset.cont_max}")

    # Save normalization stats for sampling (mirrors lobes/standard/train.py).
    if accelerator.is_main_process:
        np.savez(
            'checkpoints/cond_stats.npz',
            cont_min=dataset.cont_min,
            cont_max=dataset.cont_max,
            layer_types=np.array(LAYER_TYPES, dtype=object),
        )

    epochs_fm = 40  # ~12.5 h on 8 nodes; matches lobes' samples-seen with margin.
    save_every = 5  # checkpoint every 5 epochs (~1.5 h on 8-node Vista).

    accelerator.print("\n--- Training Flow Matching ---")
    model_fm = UNet3D(in_channels=1, num_cond=COND_DIM,
                      num_time_embs=1, expand_angle_idx=None).to(device)
    method_fm = FlowMatching(model_fm)
    # Resumable training: rerun the same `sbatch run_vista.sh` and it picks up
    # from checkpoints/training_state.pt. To reset, delete that file.
    loss_fm = train_model(
        method_fm, train_loader, epochs=epochs_fm,
        accelerator=accelerator,
        checkpoint_dir='checkpoints', save_every=save_every,
        total_epochs=epochs_fm,
    )
    if accelerator.is_main_process:
        # Final EMA-applied weights — same name as lobes' final checkpoint.
        torch.save(method_fm.model.state_dict(), "checkpoints/flow_matching.pt")
        plot_loss(loss_fm, "Flow Matching Training Loss",
                  "results/loss_flow_matching.png")

    accelerator.print("\nDone! Checkpoints saved to 'checkpoints/'.")
    accelerator.wait_for_everyone()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
