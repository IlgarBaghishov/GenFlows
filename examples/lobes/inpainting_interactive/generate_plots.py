#!/usr/bin/env python
"""Generate all inpainting plots from saved .npy results.

Uses plotting functions from resflow.utils.plotting_lobes.
"""
import glob
import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from resflow.utils.plotting_lobes import (
    custom_cmap, draw_mask_boundary, plot_inpaint_comparison,
)

RESULTS_DIR = 'results'
MASK_NAMES = ['wells', 'boundaries', 'cross_section', 'combo_wells_bounds', 'combo_all']


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Discover epochs (directories only)
    epoch_dirs = sorted([d for d in glob.glob(f'{RESULTS_DIR}/epoch_*') if os.path.isdir(d)])
    available_epochs = []
    for d in epoch_dirs:
        m = re.search(r'epoch_(\d+)', d)
        if m:
            available_epochs.append(int(m.group(1)))
    print(f"Available epochs: {available_epochs}")

    if not available_epochs:
        print("No epoch results found.")
        return

    # --- Loss curve ---
    loss_path = 'checkpoints/loss_history.npy'
    if os.path.exists(loss_path):
        losses = np.load(loss_path)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(1, len(losses) + 1), losses, 'b-', linewidth=1)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.set_title('Flow Matching Inpaint — Training Loss')
        ax.grid(True, alpha=0.3)
        for e in range(25, len(losses) + 1, 25):
            ax.axvline(e, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
            ax.text(e, ax.get_ylim()[1], f'  {e}', fontsize=8, va='top', color='gray')
        plt.tight_layout()
        fig.savefig(f'{RESULTS_DIR}/loss_curve.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {RESULTS_DIR}/loss_curve.png")

    # --- Epoch comparison plots ---
    if len(available_epochs) >= 2:
        for scenario in ['wells', 'boundaries', 'combo_all']:
            sample_idx = 0
            n_epochs = len(available_epochs)
            fig, axes = plt.subplots(2, n_epochs + 1, figsize=(4 * (n_epochs + 1), 8))
            fig.suptitle(f'{scenario.title()} — Sample {sample_idx} across epochs', fontsize=14, fontweight='bold')

            gt = np.load(os.path.join(epoch_dirs[0], 'ground_truth.npy'))
            mask = np.load(os.path.join(epoch_dirs[0], f'mask_{scenario}.npy'))
            z_mid = 25

            axes[0, 0].imshow(gt[sample_idx, 0, :, :, z_mid].T, cmap=custom_cmap, vmin=0, vmax=1, origin='lower')
            axes[0, 0].set_title('Ground Truth')
            axes[1, 0].imshow(mask[sample_idx, 0, :, :, z_mid].T, cmap='gray', vmin=0, vmax=1, origin='lower')
            axes[1, 0].set_title('Mask')

            for col, epoch in enumerate(available_epochs, 1):
                epoch_dir = f'{RESULTS_DIR}/epoch_{epoch:03d}'
                res = np.load(os.path.join(epoch_dir, f'flow_matching_{scenario}_50steps.npy'))
                m_arr = np.load(os.path.join(epoch_dir, f'mask_{scenario}.npy'))

                axes[0, col].imshow(res[sample_idx, 0, :, :, z_mid].T, cmap=custom_cmap, vmin=0, vmax=1, origin='lower')
                axes[0, col].set_title(f'Epoch {epoch}')

                axes[1, col].imshow(res[sample_idx, 0, :, 25, :].T, cmap=custom_cmap, vmin=0, vmax=1, origin='lower')
                m_slice = m_arr[sample_idx, 0, :, 25, :].T
                draw_mask_boundary(axes[1, col], m_slice, color='red', linewidth=1)
                axes[1, col].set_title(f'Epoch {epoch} (XZ)')

            plt.tight_layout()
            save_path = f'{RESULTS_DIR}/epoch_comparison_{scenario}.png'
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {save_path}")

    # --- NTG convergence ---
    if len(available_epochs) >= 2:
        gt_ntg = np.load(os.path.join(epoch_dirs[-1], 'ground_truth.npy')).mean()
        fig, ax = plt.subplots(figsize=(10, 4))
        for name in MASK_NAMES:
            ntgs = []
            for epoch in available_epochs:
                res_path = f'{RESULTS_DIR}/epoch_{epoch:03d}/flow_matching_{name}_50steps.npy'
                if os.path.exists(res_path):
                    ntgs.append(np.load(res_path).mean())
                else:
                    ntgs.append(np.nan)
            ax.plot(available_epochs, ntgs, 'o-', label=name)
        ax.axhline(gt_ntg, color='black', linestyle='--', label=f'GT NTG ({gt_ntg:.3f})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('NTG')
        ax.set_title('NTG Convergence Across Training')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(f'{RESULTS_DIR}/ntg_convergence.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {RESULTS_DIR}/ntg_convergence.png")

    # --- Per-scenario plots for latest epoch ---
    latest_epoch = available_epochs[-1]
    epoch_dir = f'{RESULTS_DIR}/epoch_{latest_epoch:03d}'
    print(f"\nGenerating per-scenario plots for epoch {latest_epoch}...")

    gt = np.load(os.path.join(epoch_dir, 'ground_truth.npy'))
    n_samples = gt.shape[0]

    for name in MASK_NAMES:
        masks = np.load(os.path.join(epoch_dir, f'mask_{name}.npy'))
        results = np.load(os.path.join(epoch_dir, f'flow_matching_{name}_50steps.npy'))
        for i in range(n_samples):
            save_path = os.path.join(epoch_dir, f'plot_{name}_sample{i}.png')
            plot_inpaint_comparison(
                gt[i, 0], results[i, 0], masks[i, 0],
                f'Epoch {latest_epoch} — {name} (Sample {i})',
                save_path=save_path
            )

    print("\nDone! All plots regenerated.")


if __name__ == '__main__':
    main()
