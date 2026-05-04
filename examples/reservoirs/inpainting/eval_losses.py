"""Evaluate train/val/test FM-inpaint loss on the trained checkpoint.

Loads the EMA inference checkpoint and computes the same MSE used during
training (FM velocity loss with random t and random noise, with
drop_prob=0.1 CFG dropping) on:
  - 50000-sample subset of train  (matches val/test size for fair comparison)
  - 50000-sample val split        (full)
  - 50000-sample test split       (full)

To reduce per-cube stochasticity from t and the random initial noise,
each cube is evaluated K times (default K=4) and the per-cube losses are
averaged before reducing across ranks.

Run with the same launcher as train.py:
    torchrun --nnodes=2 --nproc_per_node=3 eval_losses.py
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from genflows.models.unet3d import UNet3D
from genflows.utils.data_reservoirs import (
    COND_DIM, ReservoirDataset, VOLUME_SHAPE,
)
from genflows.utils.masking import InpaintDataset


def setup_dist():
    if not dist.is_available() or not dist.is_initialized():
        if 'LOCAL_RANK' in os.environ:
            local_rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend='nccl')
            return dist.get_rank(), dist.get_world_size(), local_rank
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    return 0, 1, 0


def all_reduce_sum(t):
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


@torch.no_grad()
def eval_loader(model, loader, device, k_passes, drop_prob, label, rank, log_every=50):
    """Compute mean FM loss over loader, K random (t, noise) per cube."""
    model.eval()
    sum_loss = torch.zeros(1, device=device, dtype=torch.float64)
    n_seen = torch.zeros(1, device=device, dtype=torch.float64)

    t0 = time.time()
    for bi, (x, cond, mask) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        cond = cond.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        # Set inpaint context (unwrap if DDP-wrapped — we don't wrap here)
        inpaint_data = x * mask
        model.set_inpaint_context(mask, inpaint_data)

        B = x.shape[0]
        per_cube_loss = torch.zeros(B, device=device, dtype=torch.float64)

        for _ in range(k_passes):
            x0 = torch.randn_like(x)
            t = torch.rand((B,), device=device)
            t_expand = t.view(-1, *([1] * (x.ndim - 1)))
            xt = (1 - t_expand) * x0 + t_expand * x
            v_target = x - x0
            drop_mask = torch.rand(B, device=device) < drop_prob
            v_pred = model(xt, t * 1000, cond, drop_mask=drop_mask)
            # Per-cube MSE (mean over channel/spatial dims).
            per_cube = F.mse_loss(v_pred, v_target, reduction='none')
            per_cube = per_cube.flatten(1).mean(dim=1)
            per_cube_loss += per_cube.detach().to(torch.float64)

        per_cube_loss /= k_passes
        sum_loss += per_cube_loss.sum()
        n_seen += B

        if rank == 0 and (bi + 1) % log_every == 0:
            elapsed = time.time() - t0
            running = (sum_loss / n_seen.clamp(min=1)).item()
            print(f"[{label}] batch {bi+1}/{len(loader)}  "
                  f"running={running:.6f}  elapsed={elapsed:.1f}s", flush=True)

    all_reduce_sum(sum_loss)
    all_reduce_sum(n_seen)
    return (sum_loss / n_seen.clamp(min=1)).item(), int(n_seen.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        default=os.path.join(
                            os.environ.get('SCRATCH', '.'),
                            'genflows_runs/reservoirs_inpainting/checkpoints/'
                            'inference_epoch040.pt'))
    parser.add_argument('--data-dir', type=str,
                        default=os.environ.get(
                            'RESERVOIR_DATA_DIR',
                            os.path.join(os.environ.get('SCRATCH', '.'),
                                         'SiliciclasticReservoirs')))
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--k-passes', type=int, default=4,
                        help='Random (t, noise) draws per cube to reduce variance')
    parser.add_argument('--train-subset', type=int, default=50000)
    parser.add_argument('--drop-prob', type=float, default=0.1,
                        help='CFG drop probability — match training default')
    parser.add_argument('--out', type=str, default='eval_losses_epoch040.npz')
    args = parser.parse_args()

    rank, world_size, local_rank = setup_dist()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    if rank == 0:
        print(f"World size: {world_size}", flush=True)
        print(f"Checkpoint: {args.ckpt}", flush=True)
        print(f"Data dir:   {args.data_dir}", flush=True)
        print(f"K passes:   {args.k_passes}  drop_prob: {args.drop_prob}", flush=True)

    # Datasets (rank 0 may need to materialize cond cache; others wait)
    if rank == 0:
        train_set = ReservoirDataset(args.data_dir, split='train')
        val_set = ReservoirDataset(args.data_dir, split='val',
                                   cont_min=train_set.cont_min,
                                   cont_max=train_set.cont_max)
        test_set = ReservoirDataset(args.data_dir, split='test',
                                    cont_min=train_set.cont_min,
                                    cont_max=train_set.cont_max)
    if dist.is_initialized():
        dist.barrier()
    if rank != 0:
        train_set = ReservoirDataset(args.data_dir, split='train')
        val_set = ReservoirDataset(args.data_dir, split='val',
                                   cont_min=train_set.cont_min,
                                   cont_max=train_set.cont_max)
        test_set = ReservoirDataset(args.data_dir, split='test',
                                    cont_min=train_set.cont_min,
                                    cont_max=train_set.cont_max)
    if dist.is_initialized():
        dist.barrier()

    # Deterministic 50k subset of train (first N indices so all ranks agree).
    train_subset = Subset(train_set, list(range(min(args.train_subset, len(train_set)))))

    train_inp = InpaintDataset(train_subset, volume_shape=VOLUME_SHAPE)
    val_inp = InpaintDataset(val_set, volume_shape=VOLUME_SHAPE)
    test_inp = InpaintDataset(test_set, volume_shape=VOLUME_SHAPE)

    # Distributed sampler so each rank takes a disjoint slice
    def make_loader(ds):
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank,
                                     shuffle=False, drop_last=False) \
            if world_size > 1 else None
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          sampler=sampler, num_workers=args.num_workers,
                          pin_memory=True,
                          persistent_workers=args.num_workers > 0)

    train_loader = make_loader(train_inp)
    val_loader = make_loader(val_inp)
    test_loader = make_loader(test_inp)

    if rank == 0:
        print(f"Sizes  train_subset={len(train_inp)}  val={len(val_inp)}  "
              f"test={len(test_inp)}", flush=True)
        print(f"Per-rank batches  train={len(train_loader)}  "
              f"val={len(val_loader)}  test={len(test_loader)}", flush=True)

    # Model + EMA weights
    model = UNet3D(in_channels=3, out_channels=1, num_cond=COND_DIM,
                   num_time_embs=1, expand_angle_idx=None).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    # Strip 'module.' prefix if present
    state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
    model.load_state_dict(state)

    if rank == 0:
        print("\n=== Evaluating ===", flush=True)

    results = {}
    for label, loader in [('train_subset', train_loader),
                          ('val', val_loader),
                          ('test', test_loader)]:
        if rank == 0:
            print(f"\n--- {label} ---", flush=True)
        loss, n = eval_loader(model, loader, device, args.k_passes,
                              args.drop_prob, label, rank)
        results[label] = (loss, n)
        if rank == 0:
            print(f"[{label}] FINAL mean MSE = {loss:.6f}  (n={n})", flush=True)

    if rank == 0:
        print("\n========== SUMMARY ==========", flush=True)
        for label, (loss, n) in results.items():
            print(f"  {label:13s}: {loss:.6f}  (n={n})", flush=True)
        np.savez(args.out,
                 train_loss=results['train_subset'][0],
                 val_loss=results['val'][0],
                 test_loss=results['test'][0],
                 train_n=results['train_subset'][1],
                 val_n=results['val'][1],
                 test_n=results['test'][1],
                 k_passes=args.k_passes,
                 ckpt=args.ckpt)
        print(f"\nSaved: {args.out}", flush=True)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
