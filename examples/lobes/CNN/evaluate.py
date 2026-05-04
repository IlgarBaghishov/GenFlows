"""Evaluate a pretrained CNN3D property predictor.

Loads the trained model, generates predictions on train and test sets,
and produces all diagnostic plots:
  - Loss curves (from saved training history)
  - Parity plots: train (NTG-colored), test (NTG-colored), train+test overlay
  - Parity plots for sin/cos of angle (train+test overlay)
  - Error distribution histograms (height, radius, aspect ratio, azimuth)
  - RMSE and R-squared for train and test
  - PCA analysis on test residuals
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from resflow.models.cnn3d import CNN3D


# --- Configuration (must match train.py) ---
CONFIG = {
    "data_path_facies": "../data/facies_uncropped.npy",
    "data_path_properties": "../data/parameters.csv",
    "batch_size": 32,
    "test_split": 0.1,
    "validation_split": 0.1,
    "seed": 42,
}


class GeologyDataset(Dataset):
    """Custom Dataset for loading the 3D geological models."""
    def __init__(self, facies, properties):
        self.facies = torch.from_numpy(facies).float().unsqueeze(1)
        targets = properties[:, :5]
        self.scaler = StandardScaler()
        self.targets = torch.from_numpy(self.scaler.fit_transform(targets)).float()
        self.angle_ntg = properties[:, 5:]

    def __len__(self):
        return len(self.facies)

    def __getitem__(self, idx):
        return self.facies[idx], self.targets[idx], self.angle_ntg[idx]


def get_predictions(model, data_loader, device):
    """Gets model predictions for a given dataset."""
    model.eval()
    all_predictions = []
    all_targets = []
    all_angle_ntgs = []
    all_inputs = []
    with torch.no_grad():
        for inputs, targets, angl_ntg in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            angl_ntg = angl_ntg.to(device)

            outputs = model(inputs)
            all_inputs.append(inputs.cpu())
            all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())
            all_angle_ntgs.append(angl_ntg.cpu())

    inputs_tensor = torch.cat(all_inputs)
    predictions_tensor = torch.cat(all_predictions)
    targets_tensor = torch.cat(all_targets)
    angl_ntgs_tensor = torch.cat(all_angle_ntgs)
    return inputs_tensor, predictions_tensor, targets_tensor, angl_ntgs_tensor


def plot_loss_curves(history, save_path="results/loss_curves.png"):
    """Plots aesthetically enhanced training and validation loss curves."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], color='royalblue', marker='o', linestyle='-',
            linewidth=2, markersize=5, label='Training Loss')
    ax.plot(epochs, history['val_loss'], color='darkorange', marker='s', linestyle='--',
            linewidth=2, markersize=5, label='Validation Loss')

    ax.set_title('Model Loss Progression Over Epochs', fontsize=22, fontweight='bold', pad=20)
    ax.set_xlabel('Epoch', fontsize=17, fontweight='bold')
    ax.set_ylabel('Loss (Mean Squared Error)', fontsize=17, fontweight='bold')

    ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'

    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)

    min_val_loss = min(history['val_loss'])
    min_val_epoch = history['val_loss'].index(min_val_loss) + 1

    ax.annotate(f'Minimum Validation Loss:\n  Epoch: {min_val_epoch}\n  Loss: {min_val_loss:.4f}',
                xy=(min_val_epoch, min_val_loss),
                xytext=(min_val_epoch + 0.1 * len(epochs), min_val_loss + 0.1 * max(history['val_loss'])),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=15,
                bbox=dict(boxstyle="round,pad=0.5", fc="ivory", ec="gray", lw=1))

    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.legend(loc='upper right', fontsize=17, frameon=True, shadow=True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Loss curves saved to {save_path}")
    plt.close()


def plot_parity_ntg_colored(actual, predicted, ntg, title, labels, save_path):
    """2x2 parity plot colored by NTG (for train or test separately)."""
    legend_params = {
        'fontsize': 12, 'frameon': True, 'facecolor': 'white',
        'edgecolor': 'black', 'framealpha': 1,
    }
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(title, fontsize=22)

    for i, (key, label) in enumerate(labels):
        r, c = divmod(i, 2)
        sns.scatterplot(x=actual[key], y=predicted[key], ax=ax[r, c], hue=ntg)
        ax[r, c].set_title(f'Model Performance for Lobe {label}', fontsize=20)
        ax[r, c].set_xlabel(f'Actual {label}', fontsize=17)
        ax[r, c].set_ylabel(f'Predicted {label}', fontsize=17)
        lo = min(actual[key].min(), predicted[key].min())
        hi = max(actual[key].max(), predicted[key].max())
        ax[r, c].plot([lo, hi], [lo, hi], 'r--', lw=2, label='Perfect Prediction')
        ax[r, c].legend(**legend_params)
        ax[r, c].grid(True)
        ax[r, c].tick_params(axis='both', labelsize=14)
        if label == 'Azimuth':
            ax[r, c].set_xlim(-15, 200)
            ax[r, c].set_ylim(-15, 200)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Parity plot saved to {save_path}")
    plt.close()


def plot_parity_train_test_overlay(actual_train, predicted_train, actual_test, predicted_test, save_path):
    """2x2 parity plot with train (alpha=0.3) and test (alpha=0.8) overlaid.
    Bottom row shows cos and sin of angle instead of aspect ratio and azimuth."""
    legend_params = {
        'fontsize': 16, 'frameon': True, 'facecolor': 'white',
        'edgecolor': 'black', 'framealpha': 1,
    }
    plt.style.use('default')
    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('Predicted vs. Actual Values (Train & Test)', fontsize=24)

    # --- Height ---
    sns.scatterplot(x=actual_train['height'], y=predicted_train['height'], ax=ax[0, 0], alpha=0.3, label='Train Data')
    sns.scatterplot(x=actual_test['height'], y=predicted_test['height'], ax=ax[0, 0], alpha=0.8, label='Test Data')
    ax[0, 0].set_title('Model Performance for Lobe Height', fontsize=24)
    ax[0, 0].set_xlabel('Actual Height', fontsize=23)
    ax[0, 0].set_ylabel('Predicted Height', fontsize=23)
    lo = min(actual_test['height'].min(), predicted_test['height'].min())
    hi = max(actual_test['height'].max(), predicted_test['height'].max())
    ax[0, 0].plot([lo, hi], [lo, hi], 'r--', lw=2, label='Perfect Prediction')
    ax[0, 0].legend(**legend_params)
    ax[0, 0].grid(True)
    ax[0, 0].tick_params(axis='both', labelsize=18)

    # --- Radius ---
    sns.scatterplot(x=actual_train['radius'], y=predicted_train['radius'], ax=ax[0, 1], alpha=0.3, label='Train Data')
    sns.scatterplot(x=actual_test['radius'], y=predicted_test['radius'], ax=ax[0, 1], alpha=0.8, label='Test Data')
    ax[0, 1].set_title('Model Performance for Lobe Radius', fontsize=24)
    ax[0, 1].set_xlabel('Actual Radius', fontsize=23)
    ax[0, 1].set_ylabel('Predicted Radius', fontsize=23)
    lo = min(actual_test['radius'].min(), predicted_test['radius'].min())
    hi = max(actual_test['radius'].max(), predicted_test['radius'].max())
    ax[0, 1].plot([lo, hi], [lo, hi], 'r--', lw=2, label='Perfect Prediction')
    ax[0, 1].legend(**legend_params)
    ax[0, 1].grid(True)
    ax[0, 1].tick_params(axis='both', labelsize=18)

    # --- Cos of Angle ---
    sns.scatterplot(x=actual_train['anglcos'], y=predicted_train['anglcos'], ax=ax[1, 0], alpha=0.3, label='Train Data')
    sns.scatterplot(x=actual_test['anglcos'], y=predicted_test['anglcos'], ax=ax[1, 0], alpha=0.8, label='Test Data')
    ax[1, 0].set_title('Model Performance for Cos of Lobe Azimuth', fontsize=22)
    ax[1, 0].set_xlabel('Actual Cos of Lobe Azimuth', fontsize=21)
    ax[1, 0].set_ylabel('Predicted Cos of Lobe Azimuth', fontsize=21)
    lo = min(actual_test['anglcos'].min(), predicted_test['anglcos'].min())
    hi = max(actual_test['anglcos'].max(), predicted_test['anglcos'].max())
    ax[1, 0].plot([lo, hi], [lo, hi], 'r--', lw=2, label='Perfect Prediction')
    ax[1, 0].legend(**legend_params)
    ax[1, 0].grid(True)
    ax[1, 0].tick_params(axis='both', labelsize=18)

    # --- Sin of Angle ---
    sns.scatterplot(x=actual_train['anglsin'], y=predicted_train['anglsin'], ax=ax[1, 1], alpha=0.3, label='Train Data')
    sns.scatterplot(x=actual_test['anglsin'], y=predicted_test['anglsin'], ax=ax[1, 1], alpha=0.8, label='Test Data')
    ax[1, 1].set_title('Model Performance for Sin of Lobe Azimuth', fontsize=22)
    ax[1, 1].set_xlabel('Actual Sin of Azimuth', fontsize=21)
    ax[1, 1].set_ylabel('Predicted Sin of Azimuth', fontsize=21)
    lo = min(actual_test['anglsin'].min(), predicted_test['anglsin'].min())
    hi = max(actual_test['anglsin'].max(), predicted_test['anglsin'].max())
    ax[1, 1].plot([lo, hi], [lo, hi], 'r--', lw=2, label='Perfect Prediction')
    ax[1, 1].legend(**legend_params)
    ax[1, 1].grid(True)
    ax[1, 1].tick_params(axis='both', labelsize=18)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Train+Test overlay plot saved to {save_path}")
    plt.close()


def plot_error_histogram(errors, title, save_path):
    """Error distribution histogram with mean and std lines."""
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1
    plt.figure(figsize=(5, 4))
    plt.grid(False)
    plt.hist(errors, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.ylim(0, 2000)
    plt.title(title)

    mean_val = np.mean(errors)
    std_val = np.std(errors)
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5,
                label=f"Mean: {mean_val:.2f}")
    plt.axvline(mean_val + std_val, color='green', linestyle=':', linewidth=1.5,
                label=f'Std Dev: {std_val:.2f}')
    plt.axvline(mean_val - std_val, color='green', linestyle=':', linewidth=1.5)

    plt.legend(**{
        'fontsize': 10, 'frameon': True, 'facecolor': 'white',
        'edgecolor': 'black', 'framealpha': 1,
    }, loc='upper right')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Histogram saved to {save_path}")
    plt.close()


def plot_pca_residuals(y_true, y_pred, save_path):
    """PCA analysis on model residuals."""
    errors = y_pred - y_true
    scaler = StandardScaler()
    errors_scaled = scaler.fit_transform(errors)

    pca = PCA()
    principal_components = pca.fit_transform(errors_scaled)

    print("\n## PCA Analysis Results ##")
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance by component: {np.round(explained_variance, 3)}")
    print(f"Cumulative variance: {np.round(np.cumsum(explained_variance), 3)}")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Scree plot
    components = np.arange(1, len(explained_variance) + 1)
    ax1.bar(components, explained_variance * 100, color='royalblue', alpha=0.8)
    ax1.plot(components, np.cumsum(explained_variance * 100), 'ro-', markersize=8)
    ax1.set_title('Scree Plot of Explained Variance', fontsize=14)
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Percentage of Variance Explained (%)', fontsize=12)
    ax1.set_xticks(components)
    ax1.set_ylim(0, 105)

    # PC1 vs PC2 scatter
    ax2.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.5, c='black')
    ax2.set_title('PC2 vs. PC1 of Model Errors', fontsize=14)
    ax2.set_xlabel('Principal Component 1', fontsize=12)
    ax2.set_ylabel('Principal Component 2', fontsize=12)
    ax2.axhline(0, color='grey', linestyle='--')
    ax2.axvline(0, color='grey', linestyle='--')
    ax2.grid(True)
    ax2.set_aspect('equal', adjustable='box')

    plt.suptitle('PCA on Model Residuals', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"PCA plot saved to {save_path}")
    plt.close()


def unpack_predictions(values, angle_ntg_values):
    """Unpack inverse-transformed values and convert angles."""
    h = values[:, 0]
    r = values[:, 1]
    asp = values[:, 2]
    anglsin = values[:, 3]
    anglcos = values[:, 4]
    angl = np.arctan2(anglsin, anglcos) / np.pi * 180
    angl[angl < 0] += 360
    angl /= 2
    angl_real = angle_ntg_values[:, 0]
    ntg_real = angle_ntg_values[:, 1]
    return {
        'height': h, 'radius': r, 'aspect_ratio': asp,
        'anglsin': anglsin, 'anglcos': anglcos, 'angle': angl,
        'angle_real': angl_real, 'ntg_real': ntg_real,
    }


def main():
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs("results", exist_ok=True)

    # --- Load data (same as train.py) ---
    print("Loading data...")
    facies_data = np.load(CONFIG["data_path_facies"])[:, :, :, 10:-60]
    properties_data = pd.read_csv(CONFIG["data_path_properties"])[
        ["height", "radius", "aspect_ratio", "angle", "net_to_gross"]
    ]
    properties_data["sin"] = np.sin(2 * properties_data["angle"].values * np.pi / 180)
    properties_data["cos"] = np.cos(2 * properties_data["angle"].values * np.pi / 180)
    properties_data = properties_data[["height", "radius", "aspect_ratio", "sin", "cos", "angle", "net_to_gross"]]

    failed_cases = np.load("../data/failed_cases.npy")
    full_dataset = GeologyDataset(
        np.delete(facies_data, failed_cases, axis=0),
        np.delete(properties_data.values, failed_cases, axis=0),
    )

    test_size = int(CONFIG["test_split"] * len(full_dataset))
    val_size = int(CONFIG["validation_split"] * len(full_dataset))
    train_size = len(full_dataset) - test_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    print(f"Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

    # --- Load model ---
    model = CNN3D().to(device)
    model.load_state_dict(torch.load('cnn3d_property_predictor.pth', map_location=device, weights_only=True))
    print("Model loaded from cnn3d_property_predictor.pth")

    # --- Loss curves ---
    if os.path.exists('training_history.npz'):
        hist = np.load('training_history.npz')
        history = {
            'train_loss': hist['train_loss'].tolist(),
            'val_loss': hist['val_loss'].tolist(),
        }
        plot_loss_curves(history, save_path="results/loss_curves.png")
    else:
        print("WARNING: training_history.npz not found, skipping loss curve plot")

    # --- Get predictions ---
    print("\nGenerating predictions on train and test sets...")
    train_inputs, train_preds, train_targets, train_angl_ntgs = get_predictions(model, train_loader, device)
    test_inputs, test_preds, test_targets, test_angl_ntgs = get_predictions(model, test_loader, device)

    scaler = full_dataset.scaler

    # NTG-based filtering
    facies_sum_train = train_inputs.numpy().sum(axis=(1, 2, 3, 4))
    facies_sum_test = test_inputs.numpy().sum(axis=(1, 2, 3, 4))
    train_mask = (facies_sum_train > 12500) & (facies_sum_train < 112500)
    test_mask = (facies_sum_test > 12500) & (facies_sum_test < 112500)
    print(f"Train after filter: {train_mask.sum()}/{len(train_mask)}")
    print(f"Test after filter: {test_mask.sum()}/{len(test_mask)}")

    # Inverse transform and apply mask
    predicted_values_train = scaler.inverse_transform(train_preds.numpy())[train_mask]
    actual_values_train = scaler.inverse_transform(train_targets.numpy())[train_mask]
    actual_angl_ntgs_train = train_angl_ntgs.numpy()[train_mask]

    predicted_values_test = scaler.inverse_transform(test_preds.numpy())[test_mask]
    actual_values_test = scaler.inverse_transform(test_targets.numpy())[test_mask]
    actual_angl_ntgs_test = test_angl_ntgs.numpy()[test_mask]

    # Unpack into dicts
    actual_train = unpack_predictions(actual_values_train, actual_angl_ntgs_train)
    predicted_train = unpack_predictions(predicted_values_train, actual_angl_ntgs_train)
    actual_test = unpack_predictions(actual_values_test, actual_angl_ntgs_test)
    predicted_test = unpack_predictions(predicted_values_test, actual_angl_ntgs_test)

    # Resolve angle ambiguity
    residuals_train = actual_train['angle'] - predicted_train['angle']
    predicted_train['angle'][residuals_train < -90] -= 180
    predicted_train['angle'][residuals_train > 90] += 180

    residuals_test = actual_test['angle'] - predicted_test['angle']
    predicted_test['angle'][residuals_test < -90] -= 180
    predicted_test['angle'][residuals_test > 90] += 180

    # --- Parity plots colored by NTG (train) ---
    labels = [('height', 'Height'), ('radius', 'Radius'),
              ('aspect_ratio', 'Aspect Ratio'), ('angle', 'Azimuth')]
    # Use angle_real for x-axis on azimuth plots
    actual_train_plot = {**actual_train, 'angle': actual_train['angle_real']}
    actual_test_plot = {**actual_test, 'angle': actual_test['angle_real']}

    plot_parity_ntg_colored(
        actual_train_plot, predicted_train, actual_train['ntg_real'],
        'Predicted vs. Actual Values Colored by Net-to-Gross (Train)',
        labels, 'results/parity_train_ntg.png',
    )

    # --- Parity plots colored by NTG (test) ---
    plot_parity_ntg_colored(
        actual_test_plot, predicted_test, actual_test['ntg_real'],
        'Predicted vs. Actual Values Colored by Net-to-Gross (Test)',
        labels, 'results/parity_test_ntg.png',
    )

    # --- Train+Test overlay with sin/cos ---
    plot_parity_train_test_overlay(
        actual_train, predicted_train, actual_test, predicted_test,
        'results/parity_train_test_overlay.png',
    )

    # --- Error histograms ---
    plot_error_histogram(
        actual_test['height'] - predicted_test['height'],
        "Error distribution on Height", "results/hist_height.png",
    )
    plot_error_histogram(
        actual_test['radius'] - predicted_test['radius'],
        "Error distribution on Radius", "results/hist_radius.png",
    )
    plot_error_histogram(
        actual_test['aspect_ratio'] - predicted_test['aspect_ratio'],
        "Error distribution on Aspect ratio", "results/hist_aspect_ratio.png",
    )
    plot_error_histogram(
        actual_test['angle'] - predicted_test['angle'],
        "Error distribution on Azimuth", "results/hist_azimuth.png",
    )

    # --- RMSE ---
    print("\n--- Test RMSE ---")
    print(f"Height RMSE: {np.sqrt(np.mean(np.square(actual_test['height'] - predicted_test['height']))):.4f}")
    print(f"Radius RMSE: {np.sqrt(np.mean(np.square(actual_test['radius'] - predicted_test['radius']))):.4f}")
    print(f"Aspect Ratio RMSE: {np.sqrt(np.mean(np.square(actual_test['aspect_ratio'] - predicted_test['aspect_ratio']))):.4f}")
    print(f"Angle RMSE: {np.sqrt(np.mean(np.square(actual_test['angle'] - predicted_test['angle']))):.4f}")

    print("\n--- Train RMSE ---")
    print(f"Height RMSE: {np.sqrt(np.mean(np.square(actual_train['height'] - predicted_train['height']))):.4f}")
    print(f"Radius RMSE: {np.sqrt(np.mean(np.square(actual_train['radius'] - predicted_train['radius']))):.4f}")
    print(f"Aspect Ratio RMSE: {np.sqrt(np.mean(np.square(actual_train['aspect_ratio'] - predicted_train['aspect_ratio']))):.4f}")
    print(f"Angle RMSE: {np.sqrt(np.mean(np.square(actual_train['angle'] - predicted_train['angle']))):.4f}")

    # --- R-squared ---
    def r2(actual, predicted):
        return 1 - np.sqrt(np.mean(np.square(actual - predicted))) / np.sqrt(np.mean(np.square(actual.mean() - actual)))

    print("\n--- Test R2 ---")
    print(f"Height R2: {r2(actual_test['height'], predicted_test['height']):.4f}")
    print(f"Radius R2: {r2(actual_test['radius'], predicted_test['radius']):.4f}")
    print(f"Aspect Ratio R2: {r2(actual_test['aspect_ratio'], predicted_test['aspect_ratio']):.4f}")
    print(f"Angle R2: {r2(actual_test['angle'], predicted_test['angle']):.4f}")

    print("\n--- Train R2 ---")
    print(f"Height R2: {r2(actual_train['height'], predicted_train['height']):.4f}")
    print(f"Radius R2: {r2(actual_train['radius'], predicted_train['radius']):.4f}")
    print(f"Aspect Ratio R2: {r2(actual_train['aspect_ratio'], predicted_train['aspect_ratio']):.4f}")
    print(f"Angle R2: {r2(actual_train['angle'], predicted_train['angle']):.4f}")

    # --- PCA on residuals ---
    y_true = np.vstack([actual_test['height'], actual_test['radius'],
                        actual_test['aspect_ratio'], actual_test['angle']]).T
    y_pred = np.vstack([predicted_test['height'], predicted_test['radius'],
                        predicted_test['aspect_ratio'], predicted_test['angle']]).T
    plot_pca_residuals(y_true, y_pred, "results/pca_residuals.png")

    print("\nDone! All results saved to results/")


if __name__ == "__main__":
    main()
