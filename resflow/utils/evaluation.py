import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from resflow.models.cnn3d import CNN3D


def compute_rmse(predicted, actual):
    """Root mean squared error."""
    return np.sqrt(np.mean((predicted - actual) ** 2))


def compute_r2(predicted, actual):
    """R-squared (coefficient of determination)."""
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    if ss_tot == 0:
        return float('nan')
    return 1.0 - ss_res / ss_tot


def sincos_to_angle_deg(sin_vals, cos_vals):
    """Convert sin(2*angle*pi/180) and cos(2*angle*pi/180) back to angle in degrees [0, 180]."""
    angle = np.arctan2(sin_vals, cos_vals) * 180.0 / np.pi  # [-180, 180]
    angle[angle < 0] += 360  # [0, 360]
    angle /= 2  # [0, 180]
    return angle


def resolve_angle_ambiguity(predicted_deg, actual_deg):
    """Adjust predicted angles to minimize error given 180-degree ambiguity.

    Returns corrected predicted angles (modifies a copy).
    """
    predicted = predicted_deg.copy()
    residuals = actual_deg - predicted
    predicted[residuals < -90] -= 180
    predicted[residuals > 90] += 180
    return predicted


def _build_scaler_params(data_dir):
    """Reconstruct StandardScaler mean and scale from the CNN training data.

    Loads parameters.csv, removes failed_cases, computes sin/cos angle encoding,
    and computes mean/std of [height, radius, aspect_ratio, sin, cos].
    This exactly reproduces the scaler fitted in the CNN training notebook.
    """
    params = pd.read_csv(os.path.join(data_dir, 'parameters.csv'))
    failed = np.load(os.path.join(data_dir, 'failed_cases.npy'))

    params = params[['height', 'radius', 'aspect_ratio', 'angle', 'net_to_gross']]
    sin_vals = np.sin(2 * params['angle'].values * np.pi / 180)
    cos_vals = np.cos(2 * params['angle'].values * np.pi / 180)

    targets = np.column_stack([
        params['height'].values,
        params['radius'].values,
        params['aspect_ratio'].values,
        sin_vals,
        cos_vals,
    ])

    # Remove failed cases (same as CNN notebook)
    valid_mask = np.ones(len(targets), dtype=bool)
    valid_mask[failed] = False
    targets = targets[valid_mask]

    mean = targets.mean(axis=0).astype(np.float32)
    scale = targets.std(axis=0).astype(np.float32)
    return mean, scale


class LobePropertyPredictor:
    """Pretrained CNN3D wrapper for predicting lobe properties from binary volumes.

    Bundles model loading, StandardScaler inverse-transform, and sin/cos-to-angle
    conversion into a single predict() call.

    Args:
        weights_path: path to CNN3D .pth checkpoint
        data_dir: path to data directory (for reconstructing scaler from parameters.csv)
        device: torch device
    """

    def __init__(self, weights_path, data_dir, device):
        self.device = device

        # Load CNN
        self.model = CNN3D().to(device)
        self.model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        self.model.eval()

        # Reconstruct scaler
        self.scaler_mean, self.scaler_scale = _build_scaler_params(data_dir)

    def _inverse_transform(self, normalized):
        """Convert from StandardScaler space to physical units."""
        return normalized * self.scaler_scale + self.scaler_mean

    @torch.no_grad()
    def predict(self, binary_volumes):
        """Predict lobe properties from binary {0,1} volumes.

        Args:
            binary_volumes: tensor of shape (B, 1, 50, 50, 50) with values in {0, 1}

        Returns:
            dict with keys 'height', 'radius', 'aspect_ratio', 'angle_deg',
            each a numpy array of shape (B,)
        """
        volumes = binary_volumes.to(self.device)
        raw_pred = self.model(volumes).cpu().numpy()
        pred = self._inverse_transform(raw_pred)

        angle = sincos_to_angle_deg(pred[:, 3], pred[:, 4])

        return {
            'height': pred[:, 0],
            'radius': pred[:, 1],
            'aspect_ratio': pred[:, 2],
            'angle_deg': angle,
        }

    @staticmethod
    def compute_ntg(binary_volumes):
        """Compute net-to-gross from binary volumes (fraction of 1s).

        Args:
            binary_volumes: tensor of shape (B, 1, 50, 50, 50) with values in {0, 1}

        Returns:
            numpy array of shape (B,)
        """
        return binary_volumes.float().mean(dim=(1, 2, 3, 4)).cpu().numpy()

    def evaluate(self, binary_volumes, target_properties):
        """Run full evaluation: predict properties, compute metrics.

        Args:
            binary_volumes: tensor of shape (B, 1, 50, 50, 50) with values in {0, 1}
            target_properties: dict with keys 'height', 'radius', 'aspect_ratio',
                'angle_deg', 'ntg' — each a numpy array of shape (B,)

        Returns:
            dict with per-property RMSE and R2 values, plus predicted values
        """
        pred = self.predict(binary_volumes)
        pred_ntg = self.compute_ntg(binary_volumes)

        # Resolve angle ambiguity before computing metrics
        pred['angle_deg'] = resolve_angle_ambiguity(
            pred['angle_deg'], target_properties['angle_deg']
        )

        results = {'predictions': pred, 'predictions_ntg': pred_ntg}

        for key in ['height', 'radius', 'aspect_ratio', 'angle_deg']:
            results[f'{key}_rmse'] = compute_rmse(pred[key], target_properties[key])
            results[f'{key}_r2'] = compute_r2(pred[key], target_properties[key])

        results['ntg_rmse'] = compute_rmse(pred_ntg, target_properties['ntg'])
        results['ntg_r2'] = compute_r2(pred_ntg, target_properties['ntg'])

        return results


def plot_parity(predictions, targets, save_path, title_prefix=''):
    """Create 2x3 parity plots (predicted vs actual) for all properties.

    Args:
        predictions: dict with 'height', 'radius', 'aspect_ratio', 'angle_deg' arrays
        targets: dict with same keys plus 'ntg'
        save_path: path to save the figure
        title_prefix: optional prefix for the suptitle (e.g. method name)
    """
    pred_ntg = predictions.get('ntg', None)

    properties = [
        ('height', 'Height'),
        ('radius', 'Radius'),
        ('aspect_ratio', 'Aspect Ratio'),
        ('angle_deg', 'Azimuth'),
        ('ntg', 'Net-to-Gross'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes_flat = axes.flatten()
    suptitle = f'Predicted vs. Actual — {title_prefix}' if title_prefix else 'Predicted vs. Actual'
    fig.suptitle(suptitle, fontsize=22)

    legend_params = {
        'fontsize': 12,
        'frameon': True,
        'facecolor': 'white',
        'edgecolor': 'black',
        'framealpha': 1,
    }

    for i, (key, label) in enumerate(properties):
        ax = axes_flat[i]

        if key == 'ntg':
            actual = targets['ntg']
            predicted = pred_ntg
        else:
            actual = targets[key]
            predicted = predictions[key]

        if predicted is None:
            ax.set_visible(False)
            continue

        # Scatter colored by target NTG
        sc = ax.scatter(actual, predicted, c=targets['ntg'], cmap='viridis',
                        alpha=0.6, s=15, edgecolors='none')
        plt.colorbar(sc, ax=ax, label='NTG', fraction=0.046, pad=0.04)

        # Perfect prediction line
        lo = min(actual.min(), predicted.min())
        hi = max(actual.max(), predicted.max())
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                'r--', lw=2, label='Perfect Prediction')

        # RMSE and R2 annotation
        rmse = compute_rmse(predicted, actual)
        r2 = compute_r2(predicted, actual)
        ax.text(0.05, 0.95, f'RMSE = {rmse:.3f}\nR$^2$ = {r2:.3f}',
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.9))

        ax.set_title(f'Lobe {label}', fontsize=18)
        ax.set_xlabel(f'Target {label}', fontsize=15)
        ax.set_ylabel(f'CNN Predicted {label}', fontsize=15)
        ax.legend(**legend_params)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=12)

    # Hide unused subplot
    axes_flat[5].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Parity plot saved to {save_path}")
