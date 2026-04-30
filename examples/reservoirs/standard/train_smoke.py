"""Tiny end-to-end smoke training for the reservoir Flow Matching pipeline.

Trains for a handful of epochs on a 2K-sample subset of train, saves a
checkpoint, plots the loss, then reloads and runs one forward pass to confirm
the saved weights deserialize. About 1-2 minutes on a single GH200.
"""
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from accelerate import Accelerator

from genflows.models.unet3d import UNet3D
from genflows.methods.flow_matching import FlowMatching
from genflows.utils.data_reservoirs import (
    ReservoirDataset, COND_DIM, LAYER_TYPES,
)
from genflows.utils.plotting import plot_loss
from genflows.utils.training import train_model


DATA_DIR = os.environ.get(
    'RESERVOIR_DATA_DIR',
    '/scratch/11316/rustamzade17/SiliciclasticReservoirs',
)
N_SUBSET = 2000
BATCH = 32
EPOCHS = 3
NUM_WORKERS = 4
OUT_DIR = 'checkpoints_smoke'
RES_DIR = 'results_smoke'


def main():
    accelerator = Accelerator()
    device = accelerator.device
    accelerator.print(f"Device: {device}")

    full = ReservoirDataset(DATA_DIR, split='train')
    accelerator.print(f"Full train: {len(full)}; using random subset of {N_SUBSET}")
    rng = np.random.default_rng(0)
    subset_idx = rng.choice(len(full), size=N_SUBSET, replace=False).tolist()
    train_set = Subset(full, subset_idx)
    train_loader = DataLoader(
        train_set, batch_size=BATCH, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    if accelerator.is_main_process:
        os.makedirs(OUT_DIR, exist_ok=True)
        os.makedirs(RES_DIR, exist_ok=True)
        np.savez(os.path.join(OUT_DIR, 'cond_stats.npz'),
                 cont_min=full.cont_min, cont_max=full.cont_max,
                 layer_types=np.array(LAYER_TYPES, dtype=object))

    model = UNet3D(in_channels=1, num_cond=COND_DIM,
                   num_time_embs=1, expand_angle_idx=None).to(device)
    method = FlowMatching(model)

    accelerator.print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    accelerator.print(f"Steps/epoch: {len(train_loader)}")

    t0 = time.time()
    losses = train_model(method, train_loader, epochs=EPOCHS, accelerator=accelerator)
    accelerator.print(f"Train done in {time.time()-t0:.1f}s; final loss: {losses[-1]:.4f}")

    if accelerator.is_main_process:
        ckpt = os.path.join(OUT_DIR, 'flow_matching_smoke.pt')
        torch.save(method.model.state_dict(), ckpt)
        plot_loss(losses, "FM smoke (2K subset)", os.path.join(RES_DIR, 'loss_smoke.png'))

        # Reload + sanity forward pass.
        m2 = UNet3D(in_channels=1, num_cond=COND_DIM,
                    num_time_embs=1, expand_angle_idx=None).to(device)
        m2.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        m2.eval()
        with torch.no_grad():
            x = torch.randn(2, 1, 64, 64, 32, device=device)
            t = torch.rand(2, device=device)
            c = torch.randn(2, COND_DIM, device=device)
            out = m2(x, t, c)
        accelerator.print(f"Reloaded checkpoint; forward output: {tuple(out.shape)}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
