"""Well-conditioned big lobe reservoir + per-voxel Bernoulli entropy map.

Workflow:
  1. Load the no-wells base reservoir produced by generate.py.
  2. Carve a 1-voxel vertical well at the centre of every block. The well
     "data" (clean facies values) is taken from the base reservoir so the
     inpaint context is self-consistent.
  3. Generate ONE with-wells realization (seed 0). Plot the same 4-figure
     suite as visualize.py (headline, xy_slices, xz_rows, yz_cols) with
     the wells overlaid in red.
  4. Generate N additional realizations with different noise seeds across
     3 GPUs in parallel (subprocess workers).
  5. Compute per-voxel Bernoulli entropy
        H(p) = -p log2(p) - (1-p) log2(1-p)
     and render the same 4-figure suite (binary cmap replaced with viridis).

Usage:
    python wells_entropy.py                       # orchestrator (3-GPU parallel)
    python wells_entropy.py --worker --seed-start 10 --n-real 30  # one worker
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import torch

from resflow.models.unet3d import UNet3D
from resflow.methods.flow_matching import FlowMatching
from resflow.assembly import (
    BlockSpec, COND_DIM, LAYER_TYPE_TO_IDX,
    build_cond_vector,
)
from resflow.assembly.big_reservoir_multi import _compute_blend_weights

from generate import (
    CKPT, COND_STATS, GRID_SHAPE, BLOCK_SHAPE,
    N_STEPS, CFG_SCALE, build_grid_specs,
)
from visualize import (
    load_run, well_x_positions, well_y_positions,
    make_headline, make_xy_grid, make_xz_rows, make_yz_cols,
    save_fig,
)


RESULTS_DIR = 'results'
FIG_DIR = 'figures'
N_REALIZATIONS = 50
DEFAULT_OVERLAP = 16


# --------------------------------------------------------------------------
# Wells
# --------------------------------------------------------------------------

def carve_center_wells(grid_shape, block_shape, overlap, base_volume_binary):
    """Carve a vertical 1-voxel well at block-centre of every block.

    Returns (well_masks, well_data, global_mask).
    well_data is the clean facies value at well voxels mapped to {-1, +1},
    sourced from base_volume_binary.
    """
    ny, nx = grid_shape
    Sx, Sy, Sz = block_shape
    Tx, Ty, _ = base_volume_binary.shape
    stride_x = Sx - overlap
    stride_y = Sy - overlap
    cx, cy = Sx // 2, Sy // 2
    well_masks, well_data = {}, {}
    global_mask = np.zeros((Tx, Ty, Sz), dtype=np.uint8)
    base_pm1 = base_volume_binary.astype(np.float32) * 2.0 - 1.0
    for i in range(ny):
        for j in range(nx):
            xs = j * stride_x
            ys = i * stride_y
            m = torch.zeros(1, Sx, Sy, Sz)
            d = torch.zeros(1, Sx, Sy, Sz)
            m[0, cx, cy, :] = 1.0
            d[0, cx, cy, :] = torch.from_numpy(
                base_pm1[xs + cx, ys + cy, :])
            well_masks[(i, j)] = m
            well_data[(i, j)] = d
            global_mask[xs + cx, ys + cy, :] = 1
    return well_masks, well_data, global_mask


# --------------------------------------------------------------------------
# Parallel-block generator that respects per-block wells
# --------------------------------------------------------------------------

@torch.no_grad()
def generate_with_wells(method, grid_specs, well_masks, well_data,
                        cont_min, cont_max,
                        block_shape, overlap_xy, n_steps, cfg_scale,
                        max_batch, device, verbose=False):
    ny = len(grid_specs); nx = len(grid_specs[0])
    Sx, Sy, Sz = block_shape
    stride_x = Sx - overlap_xy
    stride_y = Sy - overlap_xy
    Tx = Sx + (nx - 1) * stride_x
    Ty = Sy + (ny - 1) * stride_y

    positions = [(i, j) for i in range(ny) for j in range(nx)]
    cond_per, uncond_flag = {}, {}
    for i, j in positions:
        v = build_cond_vector(grid_specs[i][j], cont_min, cont_max)
        if v is None:
            cond_per[(i, j)] = None
            uncond_flag[(i, j)] = True
        else:
            cond_per[(i, j)] = torch.from_numpy(v).to(device)
            uncond_flag[(i, j)] = False

    blend_w = _compute_blend_weights((ny, nx), block_shape, overlap_xy, device)

    x_global = torch.randn(Tx, Ty, Sz, device=device)
    dt = 1.0 / n_steps

    def block_origin(i, j):
        return j * stride_x, i * stride_y

    for step in range(n_steps):
        t_val = step * dt
        t0 = time.time()
        v_accum = torch.zeros_like(x_global)
        w_accum = torch.zeros_like(x_global)

        cond_pos = [p for p in positions if not uncond_flag[p]]
        uncond_pos = [p for p in positions if uncond_flag[p]]

        for group_pos, group_uncond in [(cond_pos, False), (uncond_pos, True)]:
            for b_start in range(0, len(group_pos), max_batch):
                batch_pos = group_pos[b_start:b_start + max_batch]
                bs = len(batch_pos)
                if bs == 0:
                    continue
                masks_b = torch.stack([well_masks[p] for p in batch_pos]).to(device)
                data_b  = torch.stack([well_data[p]  for p in batch_pos]).to(device)
                method.model.set_inpaint_context(masks_b, data_b)

                blocks = torch.zeros(bs, 1, Sx, Sy, Sz, device=device)
                for idx, (i, j) in enumerate(batch_pos):
                    xs, ys = block_origin(i, j)
                    blocks[idx, 0] = x_global[xs:xs + Sx, ys:ys + Sy, :]

                t_tensor = torch.full((bs,), t_val, device=device)
                t_emb_in = t_tensor * 1000

                if group_uncond:
                    v_pred = method.model(blocks, t_emb_in)
                else:
                    cond_batch = torch.stack(
                        [cond_per[p] for p in batch_pos], dim=0)
                    if cfg_scale > 0:
                        v_cond = method.model(blocks, t_emb_in, cond_batch)
                        v_uncond = method.model(blocks, t_emb_in)
                        v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
                    else:
                        v_pred = method.model(blocks, t_emb_in, cond_batch)

                for idx, (i, j) in enumerate(batch_pos):
                    xs, ys = block_origin(i, j)
                    w = blend_w[(i, j)]
                    v_accum[xs:xs + Sx, ys:ys + Sy, :] += w * v_pred[idx, 0]
                    w_accum[xs:xs + Sx, ys:ys + Sy, :] += w

        x_global = x_global + (v_accum / w_accum.clamp(min=1e-8)) * dt
        if verbose and (step % 10 == 0 or step == n_steps - 1):
            print(f"    step {step:3d}/{n_steps}  {time.time()-t0:.2f}s",
                  flush=True)

    method.model.clear_inpaint_context()
    return x_global.cpu()


# --------------------------------------------------------------------------
# Entropy
# --------------------------------------------------------------------------

def bernoulli_entropy(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


# --------------------------------------------------------------------------
# Worker: sample N realizations on whichever GPU CUDA_VISIBLE_DEVICES picks
# --------------------------------------------------------------------------

def _load_method(device):
    model = UNet3D(in_channels=3, out_channels=1, num_cond=COND_DIM,
                   num_time_embs=1, expand_angle_idx=None).to(device)
    model.load_state_dict(
        torch.load(CKPT, map_location=device, weights_only=True))
    model.eval()
    return FlowMatching(model)


def run_worker(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[worker {args.seed_start}-{args.seed_start + args.n_real - 1}] "
          f"device={device}", flush=True)

    stats = np.load(COND_STATS, allow_pickle=True)
    cont_min, cont_max = stats['cont_min'], stats['cont_max']
    base_npz = np.load(
        os.path.join(RESULTS_DIR, f'reservoir_hard_ov{args.overlap:02d}.npz'),
        allow_pickle=True)
    base_binary = base_npz['binary']

    grid = build_grid_specs(cont_min, cont_max)
    well_masks, well_data, _ = carve_center_wells(
        GRID_SHAPE, BLOCK_SHAPE, args.overlap, base_binary)

    method = _load_method(device)

    real_dir = os.path.join(RESULTS_DIR, f'realizations_ov{args.overlap:02d}')
    os.makedirs(real_dir, exist_ok=True)

    for k in range(args.n_real):
        seed = args.seed_start + k
        out = os.path.join(real_dir, f'sample_{seed:04d}.npy')
        if os.path.exists(out):
            print(f"[worker] seed {seed} exists, skipping", flush=True)
            continue
        torch.manual_seed(seed)
        t0 = time.time()
        x = generate_with_wells(
            method, grid, well_masks, well_data,
            cont_min, cont_max,
            block_shape=BLOCK_SHAPE, overlap_xy=args.overlap,
            n_steps=N_STEPS, cfg_scale=CFG_SCALE,
            max_batch=24, device=device, verbose=False,
        )
        binary = (x.numpy() > 0).astype(np.int8)
        np.save(out, binary)
        print(f"[worker] seed {seed:4d}: NTG={binary.mean():.3f}  "
              f"{time.time()-t0:.1f}s", flush=True)


# --------------------------------------------------------------------------
# Plot bundles using the visualize.py helpers
# --------------------------------------------------------------------------

def _make_run_for_array(arr, overlap):
    """Wrap an arbitrary (Tx, Ty, Sz) array as a run-dict for the plotters."""
    Tx, Ty, Sz = arr.shape
    return {
        'binary': arr,
        'mode': 'hard',
        'overlap': overlap,
        'block_shape': BLOCK_SHAPE,
        'ny': GRID_SHAPE[0],
        'nx': GRID_SHAPE[1],
    }


def render_with_wells_suite(seed0_binary, well_xs, well_ys, base_binary,
                            overlap, fig_dir):
    """Same 4-figure suite as visualize.py but with wells overlaid (cells
    coloured by the conditioning facies, red bordered)."""
    run = _make_run_for_array(seed0_binary, overlap)
    os.makedirs(fig_dir, exist_ok=True)
    make_headline(run, os.path.join(fig_dir, 'headline.png'),
                  well_xs=well_xs, well_ys=well_ys, base_binary=base_binary)
    make_xy_grid(run, os.path.join(fig_dir, 'xy_slices.png'),
                 well_xs=well_xs, well_ys=well_ys, base_binary=base_binary,
                 title_prefix='With wells — XY slices')
    make_xz_rows(run, os.path.join(fig_dir, 'xz_rows.png'),
                 well_xs=well_xs, well_ys=well_ys, base_binary=base_binary,
                 title_prefix='With wells — XZ rows (at well y)')
    make_yz_cols(run, os.path.join(fig_dir, 'yz_cols.png'),
                 well_xs=well_xs, well_ys=well_ys, base_binary=base_binary,
                 title_prefix='With wells — YZ cols (at well x)')


def render_entropy_suite(entropy, well_xs, well_ys, base_binary,
                         overlap, n_real, fig_dir):
    """Same 4-figure layout but coloured by entropy in [0, 1] bits."""
    run = _make_run_for_array(entropy, overlap)
    os.makedirs(fig_dir, exist_ok=True)
    Tx, Ty, Sz = entropy.shape
    z_mid = Sz // 2

    make_xy_grid(run, os.path.join(fig_dir, 'xy_slices.png'),
                 well_xs=well_xs, well_ys=well_ys, base_binary=base_binary,
                 volume=entropy, cmap='viridis', vmin=0, vmax=1,
                 cbar_label='entropy (bits)',
                 title_prefix=f'Entropy XY slices ({n_real} reals)')
    make_xz_rows(run, os.path.join(fig_dir, 'xz_rows.png'),
                 well_xs=well_xs, well_ys=well_ys, base_binary=base_binary,
                 volume=entropy, cmap='viridis', vmin=0, vmax=1,
                 cbar_label='entropy (bits)',
                 title_prefix=f'Entropy XZ rows at well y ({n_real} reals)')
    make_yz_cols(run, os.path.join(fig_dir, 'yz_cols.png'),
                 well_xs=well_xs, well_ys=well_ys, base_binary=base_binary,
                 volume=entropy, cmap='viridis', vmin=0, vmax=1,
                 cbar_label='entropy (bits)',
                 title_prefix=f'Entropy YZ cols at well x ({n_real} reals)')

    # A small headline-style entropy figure: top-down XY at z_mid + long XZ at y_mid
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(0.022 * Tx + 4, 0.022 * Ty + 4 + 0.10 * Sz + 1.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[Ty + 30, Sz + 4], hspace=0.12)
    ax_top = fig.add_subplot(gs[0])
    from visualize import shade_overlaps, draw_borders, overlay_wells_xy, \
        overlay_wells_in_xz, _pcm
    im = _pcm(ax_top, entropy[:, :, z_mid].T, [0, Tx, 0, Ty], 'viridis', 0, 1)
    shade_overlaps(ax_top, run, 'x', 0, Ty)
    shade_overlaps(ax_top, run, 'y', 0, Tx)
    draw_borders(ax_top, run)
    overlay_wells_xy(ax_top, well_xs, well_ys, base_binary, z_mid)
    ax_top.set_title(f'Entropy top-down XY @ z={z_mid}  '
                     f'({n_real} reals, overlap={overlap})',
                     fontsize=11, fontweight='bold')
    ax_top.set_xlabel('X'); ax_top.set_ylabel('Y')

    ax_long = fig.add_subplot(gs[1])
    y_mid = Ty // 2
    _pcm(ax_long, entropy[:, y_mid, :].T, [0, Tx, 0, Sz], 'viridis', 0, 1)
    Sx_ = run['block_shape'][0]
    from visualize import block_origins
    for x0 in block_origins(run['nx'], Sx_, run['overlap']):
        ax_long.axvline(x0, color='red', lw=0.4, alpha=0.4)
        ax_long.axvline(x0 + Sx_, color='red', lw=0.4, alpha=0.4)
    if y_mid in well_ys:
        overlay_wells_in_xz(ax_long, well_xs, base_binary, y_mid, Sz)
    ax_long.set_title(f'Entropy long-section XZ @ y={y_mid}', fontsize=10)
    ax_long.set_xlabel('X'); ax_long.set_ylabel('Z')
    cax = fig.add_axes([0.93, 0.10, 0.012, 0.80])
    fig.colorbar(im, cax=cax, label='entropy (bits)')
    save_fig(fig, os.path.join(fig_dir, 'headline.png'), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------

def run_orchestrator(args):
    overlap = args.overlap
    n_total = args.n_real

    # Load base, carve wells, get global mask + well coordinate lists
    stats = np.load(COND_STATS, allow_pickle=True)
    cont_min, cont_max = stats['cont_min'], stats['cont_max']
    base_path = os.path.join(RESULTS_DIR, f'reservoir_hard_ov{overlap:02d}.npz')
    base_npz = np.load(base_path, allow_pickle=True)
    base_binary = base_npz['binary']
    _, _, global_mask = carve_center_wells(
        GRID_SHAPE, BLOCK_SHAPE, overlap, base_binary)
    base_run = load_run(base_path)
    well_xs = well_x_positions(base_run)
    well_ys = well_y_positions(base_run)
    print(f"Wells: {len(well_xs)}x{len(well_ys)} columns = "
          f"{len(well_xs)*len(well_ys)} wells", flush=True)
    print(f"  well x positions: {well_xs}", flush=True)
    print(f"  well y positions: {well_ys}", flush=True)

    if not args.plots_only:
        # Launch 3 GPU workers
        n_gpus = 3
        chunks = [n_total // n_gpus] * n_gpus
        for r in range(n_total - sum(chunks)):
            chunks[r] += 1
        starts = [0]
        for c in chunks[:-1]:
            starts.append(starts[-1] + c)

        procs = []
        for gpu_id, (sstart, nr) in enumerate(zip(starts, chunks)):
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            log_path = os.path.join(RESULTS_DIR, f'worker_gpu{gpu_id}.log')
            os.makedirs(RESULTS_DIR, exist_ok=True)
            log_fh = open(log_path, 'w')
            p = subprocess.Popen(
                [sys.executable, __file__, '--worker',
                 '--seed-start', str(sstart),
                 '--n-real', str(nr),
                 '--overlap', str(overlap)],
                env=env, stdout=log_fh, stderr=subprocess.STDOUT)
            procs.append((p, log_fh, gpu_id, sstart, nr))
            print(f"  launched worker gpu={gpu_id}: seeds {sstart}..{sstart+nr-1}",
                  flush=True)
        print(f"\nWaiting for {n_gpus} workers ({n_total} realizations)...",
              flush=True)
        for p, fh, gpu_id, sstart, nr in procs:
            rc = p.wait()
            fh.close()
            if rc != 0:
                print(f"  WORKER gpu={gpu_id} FAILED (rc={rc})  see "
                      f"{RESULTS_DIR}/worker_gpu{gpu_id}.log", flush=True)
                sys.exit(1)
            print(f"  worker gpu={gpu_id} done", flush=True)
    else:
        print("plots-only: skipping sampling, re-rendering from saved data",
              flush=True)

    # Load all realizations
    real_dir = os.path.join(RESULTS_DIR, f'realizations_ov{overlap:02d}')
    files = sorted(f for f in os.listdir(real_dir) if f.startswith('sample_'))
    print(f"\nAggregating {len(files)} realizations from {real_dir}...",
          flush=True)
    samples = np.stack([np.load(os.path.join(real_dir, f)) for f in files],
                       axis=0)
    print(f"  stack shape: {samples.shape}  dtype={samples.dtype}", flush=True)

    # Plot the seed-0 realization with the full 4-figure suite
    wells_fig_dir = os.path.join(FIG_DIR, f'with_wells_ov{overlap:02d}')
    render_with_wells_suite(samples[0], well_xs, well_ys, base_binary,
                            overlap, wells_fig_dir)
    print(f"  with-wells plots -> {wells_fig_dir}/", flush=True)

    # Per-voxel mean -> entropy, plot 4-figure suite
    p_mean = samples.mean(axis=0).astype(np.float32)
    entropy = bernoulli_entropy(p_mean).astype(np.float32)
    np.savez(os.path.join(RESULTS_DIR, f'entropy_ov{overlap:02d}.npz'),
             p=p_mean, entropy=entropy, n_real=len(files))
    ent_fig_dir = os.path.join(FIG_DIR, f'entropy_ov{overlap:02d}')
    render_entropy_suite(entropy, well_xs, well_ys, base_binary,
                         overlap, len(files), ent_fig_dir)
    print(f"  entropy plots -> {ent_fig_dir}/", flush=True)
    print(f"  entropy stats: min={entropy.min():.3f}  max={entropy.max():.3f}  "
          f"mean={entropy.mean():.3f}", flush=True)


# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--worker', action='store_true')
    p.add_argument('--seed-start', type=int, default=0)
    p.add_argument('--n-real', type=int, default=N_REALIZATIONS)
    p.add_argument('--overlap', type=int, default=DEFAULT_OVERLAP)
    p.add_argument('--plots-only', action='store_true',
                   help='Skip sampling; just re-render plots from saved npy/npz.')
    return p.parse_args()


def main():
    args = parse_args()
    if args.worker:
        run_worker(args)
    else:
        run_orchestrator(args)


if __name__ == '__main__':
    main()
