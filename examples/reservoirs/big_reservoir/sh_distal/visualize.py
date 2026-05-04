"""Render figures for each lobe-only big-reservoir run.

Per-run output dir holds:
    headline.png          top-down XY at the highest-NTG z-slice + long-section
                           XZ at y=middle. Block borders + overlap shading.
    xy_slices.png         5 XY (top-down) slices at z fractions 0.1..0.9 in a
                           2-row x 3-col grid.
    xz_rows.png           10 XZ panels, ONE per Y-row, taken at the well y
                           position (block-centre) so vertical wells are
                           always intersected when wells are overlaid later.
    yz_cols.png           10 YZ panels, ONE per X-column, at the well x
                           position (block-centre).

Plus a top-level compare_topdown.png aggregating the runs by overlap.

Helpers in this file are reused by wells_entropy.py to keep with-well and
entropy plots in the exact same layout.
"""
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np


CMAP = LinearSegmentedColormap.from_list('grey_yellow', ['grey', 'yellow'])
WELL_COLOR = 'red'
RESULTS_DIR = os.environ.get('RESERVOIR_OUT_DIR', 'results')
FIG_DIR = os.environ.get('RESERVOIR_FIG_DIR', 'figures')
SLICE_FRACS = [0.1, 0.3, 0.5, 0.7, 0.9]

# When RESERVOIR_FAST=1 (default) the pcolormesh background is rendered as
# a single rasterised image embedded in the PDF (DPI = SAVE_DPI below).
# Wells and all other artists stay vector, so text + well outlines are
# crisp at any zoom but the reservoir background pixelates past the
# embedded DPI.  Set RESERVOIR_FAST=0 for true-vector pcolormesh (slow,
# infinite zoom on every voxel) — use this for the final paper figures.
FAST = os.environ.get('RESERVOIR_FAST', '1') != '0'
SAVE_DPI = int(os.environ.get('RESERVOIR_DPI', '200' if FAST else '300'))


def save_fig(fig, out_path, dpi=None):
    """Save the figure as a PDF.  When ``FAST`` is true the pcolormesh
    backgrounds are already flagged ``rasterized=True`` (see ``_pcm``) so
    the PDF embeds them as a hi-DPI image while keeping wells/text/axes
    vector.  When ``FAST`` is false the backgrounds are pure vector.
    """
    pdf_path = os.path.splitext(out_path)[0] + '.pdf'
    fig.savefig(pdf_path, dpi=dpi or SAVE_DPI, bbox_inches='tight')


def _pcm(ax, arr_2d, extent, cmap, vmin, vmax):
    """Background renderer.

    With ``FAST=True`` (default) the pcolormesh artist is flagged
    ``rasterized=True`` so the PDF embeds it as a single bitmap at
    SAVE_DPI — fast to render, fast to write, hybrid quality (vector
    wells + raster reservoir).  With ``FAST=False`` the cells are pure
    vector (true infinite-zoom, slow).
    """
    ny, nx = arr_2d.shape
    x_lo, x_hi, y_lo, y_hi = extent
    xs = np.linspace(x_lo, x_hi, nx + 1)
    ys = np.linspace(y_lo, y_hi, ny + 1)
    pcm = ax.pcolormesh(xs, ys, arr_2d, cmap=cmap, vmin=vmin, vmax=vmax,
                        shading='flat', edgecolors='none', linewidth=0,
                        antialiased=False, rasterized=FAST)
    # Pin the limits to the data extent BEFORE setting aspect='equal'.
    # Without this, sharex=True on stacked panels (e.g. yz_cols where the
    # data is 496-wide x 32-tall) causes matplotlib to expand ylim to
    # maintain the aspect, leaving all data crammed at the bottom.
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect('equal')
    return pcm


# --------------------------------------------------------------------------
# Run loading + geometry helpers
# --------------------------------------------------------------------------

def load_run(path):
    d = np.load(path, allow_pickle=True)
    return {
        'binary': d['binary'],
        'mode': str(d['mode']),
        'overlap': int(d['overlap']),
        'block_shape': tuple(int(s) for s in d['block_shape']),
        'ny': int(d['ny']),
        'nx': int(d['nx']),
    }


def block_origins(n, S, overlap):
    stride = S - overlap
    return [k * stride for k in range(n)]


def well_y_positions(run):
    """Global y coords of row-wise wells (one per Y-row, at block centre)."""
    Sy = run['block_shape'][1]
    cy = Sy // 2
    return [origin + cy for origin in
            block_origins(run['ny'], Sy, run['overlap'])]


def well_x_positions(run):
    Sx = run['block_shape'][0]
    cx = Sx // 2
    return [origin + cx for origin in
            block_origins(run['nx'], Sx, run['overlap'])]


# --------------------------------------------------------------------------
# Borders, overlap shading, well overlays
# --------------------------------------------------------------------------

def shade_overlaps(ax, run, axis, lo, hi):
    Sx, Sy, _ = run['block_shape']
    n = run['nx'] if axis == 'x' else run['ny']
    S = Sx if axis == 'x' else Sy
    overlap = run['overlap']
    if overlap <= 0:
        return
    origins = block_origins(n, S, overlap)
    color = 'red' if axis == 'x' else 'cyan'
    alpha = 0.05 if axis == 'x' else 0.10
    for k in range(n - 1):
        a_lo = origins[k + 1]
        a_hi = origins[k] + S
        if axis == 'x':
            ax.add_patch(Rectangle((a_lo, lo), a_hi - a_lo, hi - lo,
                                    facecolor=color, alpha=alpha,
                                    zorder=3, edgecolor='none'))
        else:
            ax.add_patch(Rectangle((lo, a_lo), hi - lo, a_hi - a_lo,
                                    facecolor=color, alpha=alpha,
                                    zorder=3, edgecolor='none'))


def draw_borders(ax, run, draw_x=True, draw_y=True):
    Sx, Sy, _ = run['block_shape']
    if draw_x:
        for x0 in block_origins(run['nx'], Sx, run['overlap']):
            ax.axvline(x0, color='red', lw=0.4, alpha=0.4, zorder=4)
            ax.axvline(x0 + Sx, color='red', lw=0.4, alpha=0.4, zorder=4)
    if draw_y:
        for y0 in block_origins(run['ny'], Sy, run['overlap']):
            ax.axhline(y0, color='blue', lw=0.4, alpha=0.4, zorder=4)
            ax.axhline(y0 + Sy, color='blue', lw=0.4, alpha=0.4, zorder=4)


def _facies_color(sand):
    """Same palette as the binary CMAP: yellow for sand (=1), grey for mud."""
    return 'yellow' if sand else 'grey'


def overlay_wells_xy(ax, well_xs, well_ys, base_binary=None, z_slice=None):
    """At every (well_x, well_y) draw a 1x1 cell.  If base_binary + z_slice
    are provided, fill with the conditioning facies colour (grey/yellow);
    otherwise leave unfilled.  Always outline in red so the well voxel is
    visible against any background."""
    for y in well_ys:
        for x in well_xs:
            if base_binary is not None and z_slice is not None:
                fc = _facies_color(bool(base_binary[x, y, z_slice]))
                ax.add_patch(Rectangle((x, y), 1, 1, facecolor=fc,
                                        edgecolor=WELL_COLOR, linewidth=0.6,
                                        zorder=6))
            else:
                ax.add_patch(Rectangle((x, y), 1, 1, fill=False,
                                        edgecolor=WELL_COLOR, linewidth=0.6,
                                        zorder=6))


def overlay_wells_in_xz(ax, well_xs, base_binary, y_slice, z_max):
    """For an XZ panel taken at y=y_slice, draw each well as a 1xz_max
    column: per-voxel facies-coloured fill (no per-cell border), with ONE
    red rectangle outline around the whole column."""
    for x in well_xs:
        col = base_binary[x, y_slice, :z_max]
        for z, v in enumerate(col):
            ax.add_patch(Rectangle((x, z), 1, 1,
                                    facecolor=_facies_color(bool(v)),
                                    edgecolor='none', linewidth=0,
                                    zorder=6))
        ax.add_patch(Rectangle((x, 0), 1, z_max, fill=False,
                                edgecolor=WELL_COLOR, linewidth=0.6,
                                zorder=7))


def overlay_wells_in_yz(ax, well_ys, base_binary, x_slice, z_max):
    """Same as overlay_wells_in_xz but for a YZ panel at x=x_slice."""
    for y in well_ys:
        col = base_binary[x_slice, y, :z_max]
        for z, v in enumerate(col):
            ax.add_patch(Rectangle((y, z), 1, 1,
                                    facecolor=_facies_color(bool(v)),
                                    edgecolor='none', linewidth=0,
                                    zorder=6))
        ax.add_patch(Rectangle((y, 0), 1, z_max, fill=False,
                                edgecolor=WELL_COLOR, linewidth=0.6,
                                zorder=7))


# Backward-compatible alias used elsewhere; if base_binary is given the new
# coloured-cell version is used, otherwise it falls back to outline-only.
def overlay_well_columns(ax, positions, z_max, base_binary=None,
                          slice_idx=None, axis='y'):
    if base_binary is not None and slice_idx is not None:
        if axis == 'y':
            overlay_wells_in_xz(ax, positions, base_binary, slice_idx, z_max)
        else:
            overlay_wells_in_yz(ax, positions, base_binary, slice_idx, z_max)
        return
    for p in positions:
        ax.add_patch(Rectangle((p, 0), 1, z_max, fill=False,
                                edgecolor=WELL_COLOR, linewidth=0.6,
                                zorder=6))


# --------------------------------------------------------------------------
# Headline (top-down + long-section)
# --------------------------------------------------------------------------

def make_headline(run, out_path, well_xs=None, well_ys=None,
                  base_binary=None):
    binary = run['binary']
    Tx, Ty, Sz = binary.shape
    fig = plt.figure(figsize=(0.022 * Tx + 4, 0.022 * Ty + 4 + 0.10 * Sz + 1.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[Ty + 30, Sz + 4], hspace=0.12)

    z_per_ntg = binary.reshape(-1, Sz).mean(axis=0)
    z_best = int(np.argmax(z_per_ntg))

    ax_top = fig.add_subplot(gs[0])
    _pcm(ax_top, binary[:, :, z_best].T, [0, Tx, 0, Ty], CMAP, 0, 1)
    ax_top.set_title(f'Top-down XY @ z={z_best}  '
                     f'(mode={run["mode"]}, overlap={run["overlap"]})',
                     fontsize=11, fontweight='bold')
    ax_top.set_xlabel('X'); ax_top.set_ylabel('Y')
    shade_overlaps(ax_top, run, 'x', 0, Ty)
    shade_overlaps(ax_top, run, 'y', 0, Tx)
    draw_borders(ax_top, run)
    if well_xs is not None and well_ys is not None:
        overlay_wells_xy(ax_top, well_xs, well_ys, base_binary, z_best)

    ax_long = fig.add_subplot(gs[1])
    y_mid = Ty // 2
    _pcm(ax_long, binary[:, y_mid, :].T, [0, Tx, 0, Sz], CMAP, 0, 1)
    ax_long.set_title(f'Long-section XZ @ y={y_mid}', fontsize=10)
    ax_long.set_xlabel('X'); ax_long.set_ylabel('Z')
    shade_overlaps(ax_long, run, 'x', 0, Sz)
    Sx = run['block_shape'][0]
    for x0 in block_origins(run['nx'], Sx, run['overlap']):
        ax_long.axvline(x0, color='red', lw=0.4, alpha=0.4)
        ax_long.axvline(x0 + Sx, color='red', lw=0.4, alpha=0.4)
    if (well_xs is not None and well_ys is not None and y_mid in well_ys
            and base_binary is not None):
        overlay_wells_in_xz(ax_long, well_xs, base_binary, y_mid, Sz)

    save_fig(fig, out_path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# 2x3 XY-slice grid (z fractions)
# --------------------------------------------------------------------------

def make_xy_grid(run, out_path, well_xs=None, well_ys=None,
                 base_binary=None, volume=None, cmap=CMAP, vmin=0, vmax=1,
                 cbar_label=None, title_prefix='XY slices'):
    arr = volume if volume is not None else run['binary']
    Tx, Ty, Sz = arr.shape
    idxs = [int(round(f * (Sz - 1))) for f in SLICE_FRACS]
    ncols, nrows = 3, 2
    panel_w = 6.0
    panel_h = panel_w * Ty / Tx + 0.4
    fig_w = ncols * panel_w
    fig_h = nrows * panel_h + 0.6
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                             squeeze=False)
    im = None
    for k, (idx, frac) in enumerate(zip(idxs, SLICE_FRACS)):
        r, c = divmod(k, ncols)
        ax = axes[r, c]
        im = _pcm(ax, arr[:, :, idx].T, [0, Tx, 0, Ty], cmap, vmin, vmax)
        ax.set_title(f'frac={frac:.1f}  (z={idx})', fontsize=10)
        shade_overlaps(ax, run, 'x', 0, Ty)
        shade_overlaps(ax, run, 'y', 0, Tx)
        draw_borders(ax, run)
        if well_xs is not None and well_ys is not None:
            overlay_wells_xy(ax, well_xs, well_ys, base_binary, idx)
        ax.set_xlabel('X'); ax.set_ylabel('Y')
    for k in range(len(idxs), nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].axis('off')
    fig.suptitle(f'{title_prefix}  (mode={run["mode"]}, '
                 f'overlap={run["overlap"]})',
                 fontweight='bold', fontsize=12, y=0.995)
    plt.tight_layout(rect=[0, 0, 0.94 if cbar_label else 1.0, 0.97])
    if cbar_label and im is not None:
        cax = fig.add_axes([0.95, 0.08, 0.012, 0.84])
        fig.colorbar(im, cax=cax, label=cbar_label)
    save_fig(fig, out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Stacked panels along an axis at explicit indices
# --------------------------------------------------------------------------

def _make_axis_stack(run, axis, idxs, out_path, *, volume=None,
                     well_xs=None, well_ys=None, base_binary=None,
                     cmap=CMAP, vmin=0, vmax=1,
                     cbar_label=None, title_prefix=''):
    arr = volume if volume is not None else run['binary']
    Tx, Ty, Sz = arr.shape
    if axis == 'y':
        slicer = lambda i: arr[:, i, :].T
        ext = [0, Tx, 0, Sz]; xl='X'; yl='Z'; img_w=Tx; img_h=Sz
        view_label = 'XZ'
    elif axis == 'x':
        slicer = lambda i: arr[i, :, :].T
        ext = [0, Ty, 0, Sz]; xl='Y'; yl='Z'; img_w=Ty; img_h=Sz
        view_label = 'YZ'
    else:
        raise ValueError(axis)

    fig_w = 14.0
    panel_h = fig_w * img_h / img_w + 0.36
    fig_h = len(idxs) * panel_h + 0.6
    fig, axes = plt.subplots(len(idxs), 1, figsize=(fig_w, fig_h),
                             sharex=True)
    if len(idxs) == 1:
        axes = [axes]
    im = None
    for ax, idx in zip(axes, idxs):
        im = _pcm(ax, slicer(idx), ext, cmap, vmin, vmax)
        ax.set_title(f'{view_label} @ {axis}={idx}', fontsize=9, pad=1)
        ax.set_ylabel(yl, fontsize=8)
        if axis == 'y':
            shade_overlaps(ax, run, 'x', 0, Sz)
            Sx = run['block_shape'][0]
            for x0 in block_origins(run['nx'], Sx, run['overlap']):
                ax.axvline(x0, color='red', lw=0.4, alpha=0.4)
                ax.axvline(x0 + Sx, color='red', lw=0.4, alpha=0.4)
            # Each XZ panel cuts across Y at idx (a well_y row).  The wells
            # in that row appear as vertical red columns at the well x's.
            if (well_xs is not None and well_ys is not None
                    and idx in well_ys and base_binary is not None):
                overlay_wells_in_xz(ax, well_xs, base_binary, idx, Sz)
        else:
            shade_overlaps(ax, run, 'y', 0, Sz)
            Sy = run['block_shape'][1]
            for y0 in block_origins(run['ny'], Sy, run['overlap']):
                ax.axvline(y0, color='blue', lw=0.4, alpha=0.4)
                ax.axvline(y0 + Sy, color='blue', lw=0.4, alpha=0.4)
            if (well_xs is not None and well_ys is not None
                    and idx in well_xs and base_binary is not None):
                overlay_wells_in_yz(ax, well_ys, base_binary, idx, Sz)
    axes[-1].set_xlabel(xl)
    if title_prefix:
        fig.suptitle(f'{title_prefix}  (mode={run["mode"]}, '
                     f'overlap={run["overlap"]})',
                     fontweight='bold', fontsize=11, y=0.995)
    plt.subplots_adjust(hspace=0.05, top=1 - 0.6/fig_h, bottom=0.04,
                        left=0.04, right=0.92 if cbar_label else 0.99)
    if cbar_label and im is not None:
        cax = fig.add_axes([0.93, 0.06, 0.012, 0.88])
        fig.colorbar(im, cax=cax, label=cbar_label)
    save_fig(fig, out_path, dpi=120)
    plt.close(fig)


def make_xz_rows(run, out_path, **kw):
    """One XZ panel per Y-row, sliced at the well y-position of that row."""
    ys = well_y_positions(run)
    _make_axis_stack(run, 'y', ys, out_path,
                     title_prefix=kw.pop('title_prefix', 'XZ rows'), **kw)


def make_yz_cols(run, out_path, **kw):
    xs = well_x_positions(run)
    _make_axis_stack(run, 'x', xs, out_path,
                     title_prefix=kw.pop('title_prefix', 'YZ cols'), **kw)


# --------------------------------------------------------------------------
# Compare across overlaps
# --------------------------------------------------------------------------

def make_compare(runs, out_path):
    overlaps = sorted(runs.keys())
    if not overlaps:
        return
    fig, axes = plt.subplots(1, len(overlaps),
                             figsize=(len(overlaps) * 6, 6), squeeze=False)
    for oi, ov in enumerate(overlaps):
        ax = axes[0, oi]
        run = runs[ov]
        binary = run['binary']
        Tx, Ty, Sz = binary.shape
        z_per_ntg = binary.reshape(-1, Sz).mean(axis=0)
        z_best = int(np.argmax(z_per_ntg))
        _pcm(ax, binary[:, :, z_best].T, [0, Tx, 0, Ty], CMAP, 0, 1)
        shade_overlaps(ax, run, 'x', 0, Ty)
        shade_overlaps(ax, run, 'y', 0, Tx)
        draw_borders(ax, run)
        ax.set_title(f'overlap={ov}  NTG={binary.mean():.3f}  '
                     f'shape={Tx}x{Ty}', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Top-down XY (best-NTG z) — overlap comparison',
                 fontweight='bold', fontsize=12)
    plt.tight_layout()
    save_fig(fig, out_path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# Standalone driver: render no-well figures for every results/*.npz
# --------------------------------------------------------------------------

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, 'reservoir_*.npz')))
    print(f"Found {len(paths)} runs in {RESULTS_DIR}.")
    runs_by_ov = {}
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        run = load_run(p)
        runs_by_ov[run['overlap']] = run
        sub = os.path.join(FIG_DIR, name)
        os.makedirs(sub, exist_ok=True)
        print(f"\n{name}  shape={run['binary'].shape}  "
              f"NTG={run['binary'].mean():.3f}")
        make_headline(run, os.path.join(sub, 'headline.png'))
        print('  headline.png')
        make_xy_grid(run, os.path.join(sub, 'xy_slices.png'))
        print('  xy_slices.png')
        make_xz_rows(run, os.path.join(sub, 'xz_rows.png'))
        print('  xz_rows.png')
        make_yz_cols(run, os.path.join(sub, 'yz_cols.png'))
        print('  yz_cols.png')
    if runs_by_ov:
        make_compare(runs_by_ov, os.path.join(FIG_DIR, 'compare_topdown.png'))
        print('\ncompare_topdown.png')


if __name__ == '__main__':
    main()
