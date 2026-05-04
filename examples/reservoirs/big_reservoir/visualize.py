"""Render figures for each multi-type big-reservoir run.

For each NPZ produced by ``generate.py``, the per-run output dir holds:

    headline.png            top-down XY at z=mid + long-section XZ at y=mid;
                            X-block boundaries, Y-row overlap shading, labels.
    xy_slices.png           5 XY (top-down) slices at z fractions
                            {0.1, 0.3, 0.5, 0.7, 0.9}, no whitespace.
    xz_slices/row<i>.png    one figure per Y-row, 5 panels (fractions
                            {0.1, .., 0.9}) stacked vertically.
    yz_slices/block<j>_<name>.png
                            one figure per X-block, 5 panels (block-local
                            x fractions {0.1, .., 0.9}) stacked vertically.

Plus a top-level ``compare_topdown.png`` aggregating the 6 runs.
"""
import os
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np


CMAP = LinearSegmentedColormap.from_list('grey_yellow', ['grey', 'yellow'])
RESULTS_DIR = os.environ.get('RESERVOIR_OUT_DIR', 'results')
FIG_DIR = os.environ.get('RESERVOIR_FIG_DIR', 'figures')
SLICE_FRACS = [0.1, 0.3, 0.5, 0.7, 0.9]


def short_layer(name):
    return name.split(':', 1)[1] if ':' in name else name


def load_run(path):
    d = np.load(path, allow_pickle=True)
    return {
        'binary': d['binary'],
        'volume': d['volume'],
        'mode': str(d['mode']),
        'overlap': int(d['overlap']),
        'block_shape': tuple(int(s) for s in d['block_shape']),
        'ny': int(d['ny']),
        'nx': int(d['nx']),
        'pure_x_indices': [int(i) for i in d['pure_x_indices']],
        'pure_x_layer': [str(s) for s in d['pure_x_layer']],
        'trans_x_indices': [int(i) for i in d['trans_x_indices']],
        'trans_kind': str(d['trans_kind']),
        'row_azimuths_deg': d['row_azimuths_deg'],
    }


def block_x_origins(nx, Sx, overlap):
    stride = Sx - overlap
    return [j * stride for j in range(nx)]


def block_y_origins(ny, Sy, overlap):
    stride = Sy - overlap
    return [i * stride for i in range(ny)]


def shade_overlaps_x(ax, run, y0, y1, Tx):
    """Shade the X-overlap zones between adjacent X-blocks (vertical bands)."""
    Sx = run['block_shape'][0]
    origins = block_x_origins(run['nx'], Sx, run['overlap'])
    overlap = run['overlap']
    if overlap <= 0:
        return
    for j in range(run['nx'] - 1):
        x_lo = origins[j + 1]              # start of next block
        x_hi = origins[j] + Sx             # end of current block
        ax.add_patch(Rectangle((x_lo, y0), x_hi - x_lo, y1 - y0,
                               facecolor='red', alpha=0.05, zorder=3,
                               edgecolor='none'))


def shade_overlaps_y(ax, run, x0, x1, Ty):
    """Shade the Y-overlap zones between adjacent Y-rows (horizontal bands)."""
    Sy = run['block_shape'][1]
    origins = block_y_origins(run['ny'], Sy, run['overlap'])
    overlap = run['overlap']
    if overlap <= 0:
        return
    for i in range(run['ny'] - 1):
        y_lo = origins[i + 1]
        y_hi = origins[i] + Sy
        ax.add_patch(Rectangle((x0, y_lo), x1 - x0, y_hi - y_lo,
                               facecolor='cyan', alpha=0.10, zorder=3,
                               edgecolor='none'))


def draw_block_borders(ax, run, Tx, Ty, draw_y=True):
    """Solid lines at every block leading + trailing edge in X and (optional) Y."""
    Sx, Sy, _ = run['block_shape']
    for x0 in block_x_origins(run['nx'], Sx, run['overlap']):
        ax.axvline(x0, color='red', lw=0.4, alpha=0.6, zorder=4)
        ax.axvline(x0 + Sx, color='red', lw=0.4, alpha=0.6, zorder=4)
    if draw_y:
        for y0 in block_y_origins(run['ny'], Sy, run['overlap']):
            ax.axhline(y0, color='blue', lw=0.4, alpha=0.6, zorder=4)
            ax.axhline(y0 + Sy, color='blue', lw=0.4, alpha=0.6, zorder=4)


def label_x_blocks(ax, run, label_y):
    Sx = run['block_shape'][0]
    origins = block_x_origins(run['nx'], Sx, run['overlap'])
    pure_pos_index = {j: k for k, j in enumerate(run['pure_x_indices'])}
    for j in range(run['nx']):
        x_mid = origins[j] + Sx // 2
        if j in pure_pos_index:
            label = short_layer(run['pure_x_layer'][pure_pos_index[j]])
            color = 'black'; size = 8
        else:
            label = '[' + run['trans_kind'] + ']'
            color = 'magenta'; size = 7
        ax.text(x_mid, label_y, label, ha='center', va='bottom',
                fontsize=size, rotation=30, color=color)


def make_headline(run, out_path):
    binary = run['binary']
    Tx, Ty, Sz = binary.shape

    fig = plt.figure(figsize=(0.026 * Tx + 4, 0.045 * Ty + 4 + 0.13 * Sz + 1.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[Ty + 30, Sz + 4],
                          hspace=0.10)

    ax_top = fig.add_subplot(gs[0])
    # Pick the z-slice with highest sand fraction so the headline shows the
    # most informative section (channels and lobes simultaneously).
    z_per_slice_ntg = binary.reshape(-1, Sz).mean(axis=0)
    z_best = int(np.argmax(z_per_slice_ntg))
    img_xy = binary[:, :, z_best].T
    ax_top.imshow(img_xy, cmap=CMAP, vmin=0, vmax=1, origin='lower',
                  extent=[0, Tx, 0, Ty], aspect='equal')
    az = run['row_azimuths_deg']
    ax_top.set_title(f'Top-down XY @ z={z_best}  '
                     f'(mode={run["mode"]}, overlap={run["overlap"]}, '
                     f'rows az=[{", ".join(f"{a:.0f}°" for a in az)}])',
                     fontsize=11, fontweight='bold')
    ax_top.set_xlabel('X')
    ax_top.set_ylabel('Y')
    shade_overlaps_x(ax_top, run, 0, Ty, Tx)
    shade_overlaps_y(ax_top, run, 0, Tx, Ty)
    draw_block_borders(ax_top, run, Tx, Ty, draw_y=True)
    label_x_blocks(ax_top, run, label_y=Ty + 4)

    ax_long = fig.add_subplot(gs[1])
    y_mid = Ty // 2
    img_xz = binary[:, y_mid, :].T
    ax_long.imshow(img_xz, cmap=CMAP, vmin=0, vmax=1, origin='lower',
                   extent=[0, Tx, 0, Sz], aspect='equal')
    ax_long.set_title(f'Long-section XZ @ y={y_mid}', fontsize=10)
    ax_long.set_xlabel('X')
    ax_long.set_ylabel('Z')
    shade_overlaps_x(ax_long, run, 0, Sz, Tx)
    Sx = run['block_shape'][0]
    for x0 in block_x_origins(run['nx'], Sx, run['overlap']):
        ax_long.axvline(x0, color='red', lw=0.4, alpha=0.6, zorder=4)
        ax_long.axvline(x0 + Sx, color='red', lw=0.4, alpha=0.6, zorder=4)

    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def make_xy_slices(run, out_path):
    """5 top-down XY slices stacked vertically — preserves data aspect, no gaps."""
    binary = run['binary']
    Tx, Ty, Sz = binary.shape
    zs = [int(round(f * (Sz - 1))) for f in SLICE_FRACS]

    fig_w = 12.0
    img_h = fig_w * Ty / Tx
    panel_h = img_h + 0.32   # title + small label
    n = len(zs)
    fig_h = n * panel_h + 0.5
    fig, axes = plt.subplots(n, 1, figsize=(fig_w, fig_h), sharex=True)
    for ax, z in zip(axes, zs):
        ax.imshow(binary[:, :, z].T, cmap=CMAP, vmin=0, vmax=1, origin='lower',
                  extent=[0, Tx, 0, Ty], aspect='equal')
        ax.set_title(f'z = {z}  ({z/(Sz-1):.2f} of Z)', fontsize=9, pad=1)
        ax.set_ylabel('Y', fontsize=8)
        shade_overlaps_x(ax, run, 0, Ty, Tx)
        shade_overlaps_y(ax, run, 0, Tx, Ty)
        draw_block_borders(ax, run, Tx, Ty, draw_y=True)
    axes[-1].set_xlabel('X')
    fig.suptitle(f'XY slices  (mode={run["mode"]}, overlap={run["overlap"]})',
                 fontweight='bold', fontsize=11, y=0.995)
    plt.subplots_adjust(hspace=0.04, top=1 - 0.6/fig_h, bottom=0.03,
                        left=0.04, right=0.99)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def make_xz_per_row(run, out_dir):
    """One figure per Y-row, 5 panels stacked (one per fraction)."""
    os.makedirs(out_dir, exist_ok=True)
    binary = run['binary']
    Tx, Ty, Sz = binary.shape
    Sy = run['block_shape'][1]
    stride_y = Sy - run['overlap']

    fig_w = 14.0
    img_h = fig_w * Sz / Tx
    panel_h = img_h + 0.34
    n = len(SLICE_FRACS)
    fig_h = n * panel_h + 0.5

    for i in range(run['ny']):
        y_start = i * stride_y
        ys = [int(round(y_start + f * (Sy - 1))) for f in SLICE_FRACS]
        ys = [min(y, Ty - 1) for y in ys]

        fig, axes = plt.subplots(n, 1, figsize=(fig_w, fig_h), sharex=True)
        for ax, frac, y in zip(axes, SLICE_FRACS, ys):
            ax.imshow(binary[:, y, :].T, cmap=CMAP, vmin=0, vmax=1,
                      origin='lower', extent=[0, Tx, 0, Sz], aspect='equal')
            ax.set_title(f'frac={frac:.1f}  (y={y})', fontsize=9, pad=1)
            ax.set_ylabel('Z', fontsize=8)
            shade_overlaps_x(ax, run, 0, Sz, Tx)
            Sx = run['block_shape'][0]
            for x0 in block_x_origins(run['nx'], Sx, run['overlap']):
                ax.axvline(x0, color='red', lw=0.4, alpha=0.6)
                ax.axvline(x0 + Sx, color='red', lw=0.4, alpha=0.6)
        axes[-1].set_xlabel('X')
        az = run['row_azimuths_deg'][i]
        fig.suptitle(f'XZ slices — row {i}  (azimuth={az:.0f}°,  mode={run["mode"]}, '
                     f'overlap={run["overlap"]})',
                     fontweight='bold', fontsize=11, y=0.995)
        plt.subplots_adjust(hspace=0.05, top=1 - 0.6/fig_h, bottom=0.04,
                            left=0.04, right=0.99)
        out = os.path.join(out_dir, f'row{i}.png')
        fig.savefig(out, dpi=120, bbox_inches='tight')
        plt.close(fig)


def make_yz_per_block(run, out_dir):
    """One figure per X-block; 5 panels stacked (5 block-local x fractions)."""
    os.makedirs(out_dir, exist_ok=True)
    binary = run['binary']
    Tx, Ty, Sz = binary.shape
    Sx, Sy, _ = run['block_shape']
    origins = block_x_origins(run['nx'], Sx, run['overlap'])

    pure_pos_index = {j: k for k, j in enumerate(run['pure_x_indices'])}

    for j in range(run['nx']):
        x_origin = origins[j]
        if j in pure_pos_index:
            short = short_layer(run['pure_x_layer'][pure_pos_index[j]])
            label = short
            color_label = 'black'
        else:
            short = run['trans_kind']
            label = f'[{short}]'
            color_label = 'magenta'

        xs = [int(round(x_origin + f * (Sx - 1))) for f in SLICE_FRACS]
        xs = [min(x, Tx - 1) for x in xs]

        fig_w = 10.0
        img_h = fig_w * Sz / Ty
        panel_h = img_h + 0.34
        n = len(SLICE_FRACS)
        fig_h = n * panel_h + 0.5
        fig, axes = plt.subplots(n, 1, figsize=(fig_w, fig_h), sharex=True)
        for ax, frac, x in zip(axes, SLICE_FRACS, xs):
            ax.imshow(binary[x, :, :].T, cmap=CMAP, vmin=0, vmax=1,
                      origin='lower', extent=[0, Ty, 0, Sz], aspect='equal')
            ax.set_title(f'frac={frac:.1f}  (x={x})', fontsize=9, pad=1)
            ax.set_ylabel('Z', fontsize=8)
            for y0 in block_y_origins(run['ny'], Sy, run['overlap']):
                ax.axvline(y0, color='blue', lw=0.4, alpha=0.6)
                ax.axvline(y0 + Sy, color='blue', lw=0.4, alpha=0.6)
            shade_overlaps_y(ax, run, 0, Sz, Ty)
        axes[-1].set_xlabel('Y')
        fig.suptitle(f'YZ slices — block {j} ({label})  (mode={run["mode"]}, '
                     f'overlap={run["overlap"]})',
                     fontweight='bold', fontsize=11, color=color_label, y=0.995)
        plt.subplots_adjust(hspace=0.05, top=1 - 0.7/fig_h, bottom=0.05,
                            left=0.07, right=0.99)
        out_name = f'block{j:02d}_{short}.png'
        fig.savefig(os.path.join(out_dir, out_name), dpi=120, bbox_inches='tight')
        plt.close(fig)


def make_compare_topdown(runs, out_path):
    # Pick only modes/overlaps that actually have a run on disk so the
    # compare figure stays compact when the scenario is restricted.
    all_modes = ['hard', 'soft_nobuffer', 'soft', 'buffer']
    all_overlaps = [32, 24, 16, 12, 8]
    present_modes = [m for m in all_modes if any((m, o) in runs for o in all_overlaps)]
    present_overlaps = [o for o in all_overlaps if any((m, o) in runs for m in present_modes)]
    if not present_modes or not present_overlaps:
        return
    modes = present_modes
    overlaps = present_overlaps
    fig, axes = plt.subplots(len(modes), len(overlaps),
                             figsize=(len(overlaps) * 8, len(modes) * 3.0),
                             squeeze=False)
    for mi, mode in enumerate(modes):
        for oi, ov in enumerate(overlaps):
            run = runs.get((mode, ov))
            ax = axes[mi, oi]
            if run is None:
                ax.set_axis_off()
                continue
            binary = run['binary']
            Tx, Ty, Sz = binary.shape
            z_mid = Sz // 2
            ax.imshow(binary[:, :, z_mid].T, cmap=CMAP, vmin=0, vmax=1,
                      origin='lower', extent=[0, Tx, 0, Ty], aspect='equal')
            shade_overlaps_x(ax, run, 0, Ty, Tx)
            shade_overlaps_y(ax, run, 0, Tx, Ty)
            Sx = run['block_shape'][0]
            for x0 in block_x_origins(run['nx'], Sx, run['overlap']):
                ax.axvline(x0, color='red', lw=0.3, alpha=0.6)
            ax.set_title(f'{mode}  overlap={ov}  '
                         f'NTG={binary.mean():.3f}  shape={Tx}x{Ty}',
                         fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Top-down XY (z=mid) — comparison across runs',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, 'reservoir_*.npz')))
    print(f"Found {len(paths)} runs.")

    runs = {}
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        run = load_run(p)
        runs[(run['mode'], run['overlap'])] = run

        sub = os.path.join(FIG_DIR, name)
        os.makedirs(sub, exist_ok=True)
        print(f"\n{name}  shape={run['binary'].shape}  NTG={run['binary'].mean():.3f}")

        make_headline(run, os.path.join(sub, 'headline.png'))
        print(f"  headline.png")
        make_xy_slices(run, os.path.join(sub, 'xy_slices.png'))
        print(f"  xy_slices.png")
        make_xz_per_row(run, os.path.join(sub, 'xz_slices'))
        print(f"  xz_slices/  ({run['ny']} files)")
        make_yz_per_block(run, os.path.join(sub, 'yz_slices'))
        print(f"  yz_slices/  ({run['nx']} files)")

    if runs:
        make_compare_topdown(runs, os.path.join(FIG_DIR, 'compare_topdown.png'))
        print(f"\ncompare_topdown.png")


if __name__ == '__main__':
    main()
