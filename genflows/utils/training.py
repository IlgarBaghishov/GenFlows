import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_model(method, dataloader, epochs=5, lr=1e-3, device='cuda'):
    optimizer = torch.optim.AdamW(method.model.parameters(), lr=lr)
    method.model.to(device)
    method.model.train()

    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, labels in pbar:
            x = x.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = method.compute_loss(x, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_losses.append(total_loss / len(dataloader))

    return epoch_losses


def train_reflow(method, paired_dataset, epochs=5, lr=1e-3, batch_size=128, device='cuda'):
    """Train a model on pre-generated coupled (x0, x1, labels) pairs for reflow."""
    dataloader = DataLoader(paired_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(method.model.parameters(), lr=lr)
    method.model.to(device)
    method.model.train()

    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for x0, x1, labels in pbar:
            x0 = x0.to(device)
            x1 = x1.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = method.compute_loss(x1, labels, x0=x0)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_losses.append(total_loss / len(dataloader))

    return epoch_losses
