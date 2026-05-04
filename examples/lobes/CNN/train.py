"""Train a 3D CNN to predict lobe properties from binary facies volumes.

Predicts 5 targets: [height, radius, aspect_ratio, sin(2*angle*pi/180), cos(2*angle*pi/180)]
using StandardScaler normalization. Saves best model and training history.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

from resflow.models.cnn3d import CNN3D


# --- Configuration ---
CONFIG = {
    "data_path_facies": "../data/facies_uncropped.npy",
    "data_path_properties": "../data/parameters.csv",
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001,
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


def train_model(model, train_loader, val_loader, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(CONFIG["epochs"]):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_time = time.time() - start_time

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)

        print(f"Epoch {epoch+1}/{CONFIG['epochs']}.. "
              f"Train Loss: {epoch_train_loss:.4f}.. "
              f"Val Loss: {epoch_val_loss:.4f}.. "
              f"Time: {epoch_time:.2f}s")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'cnn3d_property_predictor.pth')
            print(f"   -> New best model saved with validation loss: {best_val_loss:.4f}")

    return history


def main():
    # Reproducibility
    torch.manual_seed(CONFIG["seed"])
    torch.Generator().manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    facies_data = np.load(CONFIG["data_path_facies"])[:, :, :, 10:-60]
    print("facies data shape is", facies_data.shape)
    properties_data = pd.read_csv(CONFIG["data_path_properties"])[
        ["height", "radius", "aspect_ratio", "angle", "net_to_gross"]
    ]
    properties_data["sin"] = np.sin(2 * properties_data["angle"].values * np.pi / 180)
    properties_data["cos"] = np.cos(2 * properties_data["angle"].values * np.pi / 180)
    properties_data = properties_data[["height", "radius", "aspect_ratio", "sin", "cos", "angle", "net_to_gross"]]
    print("Properties data shape is", properties_data.shape)
    print(properties_data)
    print("Data loaded successfully.")

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

    print(f"Dataset split into: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

    model = CNN3D().to(device)
    print("\nModel Architecture:")
    print(model)

    # Train
    print("\nStarting training...")
    training_history = train_model(model, train_loader, val_loader, device)
    print("Training finished.")

    # Save training history for evaluate.py
    np.savez('training_history.npz',
             train_loss=training_history['train_loss'],
             val_loss=training_history['val_loss'])
    print("Training history saved to training_history.npz")

    # Evaluate on test set
    print("\nEvaluating on the test set...")
    model.load_state_dict(torch.load('cnn3d_property_predictor.pth'))
    model.eval()
    test_loss = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for inputs, targets, _ in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)

    final_test_loss = test_loss / len(test_loader.dataset)
    print(f"Final Test MSE Loss: {final_test_loss:.4f}")


if __name__ == "__main__":
    main()
