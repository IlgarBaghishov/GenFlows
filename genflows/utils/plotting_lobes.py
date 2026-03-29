"""Shared plotting utilities for inpainting visualization.

Used by both sample_and_plot.ipynb and generate_plots.py.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection

custom_cmap = LinearSegmentedColormap.from_list('grey_yellow', ['grey', 'yellow'])


def draw_mask_boundary(ax, mask_2d, color='red', linewidth=1.5):
    """Draw pixel-aligned boundary around mask regions.

    Unlike contour (which interpolates diagonally and creates diamond shapes
    around single pixels), this draws line segments exactly at pixel edges
    where the mask transitions between 0 and 1.
    """
    binary = (mask_2d > 0.5).astype(np.int8)
    ny, nx = binary.shape
    segments = []

    # Horizontal boundaries (between rows i and i+1)
    hdiff = np.diff(binary, axis=0)
    for i, j in zip(*np.nonzero(hdiff)):
        segments.append([(j - 0.5, i + 0.5), (j + 0.5, i + 0.5)])

    # Vertical boundaries (between cols j and j+1)
    vdiff = np.diff(binary, axis=1)
    for i, j in zip(*np.nonzero(vdiff)):
        segments.append([(j + 0.5, i - 0.5), (j + 0.5, i + 0.5)])

    # Edge boundaries (mask=1 touching image border)
    for j in np.nonzero(binary[0, :])[0]:
        segments.append([(j - 0.5, -0.5), (j + 0.5, -0.5)])
    for j in np.nonzero(binary[-1, :])[0]:
        segments.append([(j - 0.5, ny - 0.5), (j + 0.5, ny - 0.5)])
    for i in np.nonzero(binary[:, 0])[0]:
        segments.append([(-0.5, i - 0.5), (-0.5, i + 0.5)])
    for i in np.nonzero(binary[:, -1])[0]:
        segments.append([(nx - 0.5, i - 0.5), (nx - 0.5, i + 0.5)])

    if segments:
        lc = LineCollection(segments, colors=color, linewidths=linewidth)
        ax.add_collection(lc)


def find_transition_slices(mask_3d):
    """Find slice indices that best show the transition between known and unknown.

    For each axis, finds the slice with the most 0<->1 boundary pixels,
    rejecting slices that are all-known or all-unknown.
    """
    best = []
    for axis in range(3):
        max_score = -1
        best_idx = mask_3d.shape[axis] // 2
        for i in range(mask_3d.shape[axis]):
            slc = np.take(mask_3d, i, axis=axis)
            frac = slc.mean()
            if frac == 0.0 or frac == 1.0:
                continue
            boundary = (np.abs(np.diff(slc, axis=0)).sum() +
                       np.abs(np.diff(slc, axis=1)).sum())
            if boundary > max_score:
                max_score = boundary
                best_idx = i
        best.append(best_idx)
    return best


def plot_inpaint_comparison(ground_truth, result, mask, title, slices=None, save_path=None):
    """Plot ground truth vs inpainted result with mask boundary overlay.

    Slices are auto-selected to show the transition between known and
    unknown regions (not the known region itself).

    Args:
        ground_truth: (50, 50, 50) binary array
        result: (50, 50, 50) binary array
        mask: (50, 50, 50) binary array, 1=known
        title: figure title
        slices: (x, y, z) slice indices, or None for auto
        save_path: if provided, save figure to this path
    """
    if slices is None:
        slices = find_transition_slices(mask)
    x_sl, y_sl, z_sl = slices

    slice_defs = [
        (lambda d: d[:, :, z_sl].T, f'XY plane (z={z_sl})', 'X', 'Y'),
        (lambda d: d[:, y_sl, :].T, f'XZ plane (y={y_sl})', 'X', 'Z'),
        (lambda d: d[x_sl, :, :].T, f'YZ plane (x={x_sl})', 'Y', 'Z'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    for col, (slice_fn, col_title, xlabel, ylabel) in enumerate(slice_defs):
        gt_slice = slice_fn(ground_truth)
        res_slice = slice_fn(result)
        mask_slice = slice_fn(mask)

        # Top row: ground truth
        axes[0, col].imshow(gt_slice, cmap=custom_cmap, vmin=0, vmax=1, origin='lower')
        axes[0, col].set_title(col_title)
        axes[0, col].set_xlabel(xlabel)
        axes[0, col].set_ylabel(ylabel)

        # Bottom row: inpainted result with mask overlay
        axes[1, col].imshow(res_slice, cmap=custom_cmap, vmin=0, vmax=1, origin='lower')
        # Tint known region with semi-transparent red
        red_overlay = np.zeros((*mask_slice.shape, 4))
        red_overlay[mask_slice > 0.5] = [1, 0, 0, 0.12]
        axes[1, col].imshow(red_overlay, origin='lower')
        # Draw pixel-aligned mask boundary (not contour — avoids diamond artifacts)
        draw_mask_boundary(axes[1, col], mask_slice, color='red', linewidth=1.5)
        axes[1, col].set_xlabel(xlabel)
        axes[1, col].set_ylabel(ylabel)

    axes[0, 0].set_ylabel('Ground Truth\nY')
    axes[1, 0].set_ylabel('Inpainted (red=known)\nY')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
