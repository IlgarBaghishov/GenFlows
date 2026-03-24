import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class LobeDataset(Dataset):
    """Dataset for 3D geological lobe facies models with continuous conditioning.

    Loads facies volumes (binary int8, 50x50x50) and computes conditioning
    values: [height, radius, aspect_ratio, angle_deg, computed_ntg].

    Filtering:
        - Removes failed cases (from failed_cases.npy)
        - Removes samples with computed NTG < ntg_min or > ntg_max

    Normalization:
        - Facies: {0, 1} -> {-1, 1}
        - Conditioning: each variable normalized to [0, 1] using dataset min/max
    """

    def __init__(self, data_dir='data', ntg_min=0.05, ntg_max=0.95):
        facies_path = os.path.join(data_dir, 'facies.npy')
        params_path = os.path.join(data_dir, 'parameters.csv')
        failed_path = os.path.join(data_dir, 'failed_cases.npy')

        # Load metadata
        params = pd.read_csv(params_path)
        failed = np.load(failed_path)

        # Compute NTG from facies data
        facies_mmap = np.load(facies_path, mmap_mode='r')
        n_samples = facies_mmap.shape[0]
        total_voxels = facies_mmap.shape[1] * facies_mmap.shape[2] * facies_mmap.shape[3]

        computed_ntg = np.empty(n_samples, dtype=np.float32)
        batch_size = 10000
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            batch = np.array(facies_mmap[i:end])
            computed_ntg[i:end] = batch.sum(axis=(1, 2, 3)) / total_voxels

        # Build valid mask
        mask = np.ones(n_samples, dtype=bool)
        mask[failed] = False
        mask &= (computed_ntg >= ntg_min)
        mask &= (computed_ntg <= ntg_max)
        self.valid_indices = np.where(mask)[0]

        # Build conditioning: [height, radius, aspect_ratio, angle_deg, computed_ntg]
        cond_raw = np.column_stack([
            params['height'].values[self.valid_indices],
            params['radius'].values[self.valid_indices],
            params['aspect_ratio'].values[self.valid_indices],
            params['angle'].values[self.valid_indices],
            computed_ntg[self.valid_indices],
        ])

        # Normalize conditioning to [0, 1]
        self.cond_min = cond_raw.min(axis=0).astype(np.float32)
        self.cond_max = cond_raw.max(axis=0).astype(np.float32)
        self.cond = ((cond_raw - self.cond_min) / (self.cond_max - self.cond_min + 1e-8)).astype(np.float32)

        # Store facies path for lazy mmap access in workers
        self.facies_path = facies_path
        self._facies_mmap = None

    @property
    def facies_mmap(self):
        """Lazily open mmap — safe for DataLoader workers (each opens its own)."""
        if self._facies_mmap is None:
            self._facies_mmap = np.load(self.facies_path, mmap_mode='r')
        return self._facies_mmap

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        facies = self.facies_mmap[real_idx].astype(np.float32)
        facies = facies * 2 - 1  # {0, 1} -> {-1, 1}
        facies = torch.from_numpy(facies).unsqueeze(0)  # (1, 50, 50, 50)
        cond = torch.from_numpy(self.cond[idx])  # (5,)
        return facies, cond

    def denormalize_cond(self, cond_norm):
        """Convert normalized conditioning back to original units.

        Args:
            cond_norm: tensor of shape (..., 5) with values in [0, 1]

        Returns:
            tensor of shape (..., 5) in original units
        """
        cond_min = torch.from_numpy(self.cond_min).to(cond_norm.device)
        cond_max = torch.from_numpy(self.cond_max).to(cond_norm.device)
        return cond_norm * (cond_max - cond_min) + cond_min


class LobeInpaintDataset(Dataset):
    """Wraps a lobe dataset (or Subset) to add on-the-fly mask generation."""

    def __init__(self, base_dataset):
        from genflows.utils.masking_lobes import generate_training_mask
        self.base_dataset = base_dataset
        self._generate_mask = generate_training_mask

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        facies, cond = self.base_dataset[idx]
        mask = self._generate_mask((50, 50, 50))
        return facies, cond, mask


def get_lobe_loaders(data_dir='data', batch_size=32, ntg_min=0.05, ntg_max=0.95,
                     test_split=0.1, val_split=0.1, seed=42):
    """Create train/val/test DataLoaders for the lobe dataset."""
    dataset = LobeDataset(data_dir=data_dir, ntg_min=ntg_min, ntg_max=ntg_max)

    n = len(dataset)
    test_size = int(test_split * n)
    val_size = int(val_split * n)
    train_size = n - test_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, dataset


def get_lobe_inpaint_loaders(data_dir='data', batch_size=32, ntg_min=0.05, ntg_max=0.95,
                              test_split=0.1, val_split=0.1, seed=42):
    """Create train/val/test DataLoaders with on-the-fly mask generation for inpainting."""
    dataset = LobeDataset(data_dir=data_dir, ntg_min=ntg_min, ntg_max=ntg_max)

    n = len(dataset)
    test_size = int(test_split * n)
    val_size = int(val_split * n)
    train_size = n - test_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(LobeInpaintDataset(train_set), batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(LobeInpaintDataset(val_set), batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(LobeInpaintDataset(test_set), batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, dataset
