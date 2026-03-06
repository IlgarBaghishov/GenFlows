import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class EMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of weights updated each step:
        ema_param = decay * ema_param + (1 - decay) * model_param

    After training, call apply() to copy EMA weights into the model.
    """

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {name: p.clone().detach() for name, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model):
        """Copy EMA weights into the model (for sampling/checkpointing)."""
        for name, p in model.named_parameters():
            p.data.copy_(self.shadow[name])


def train_model(method, dataloader, epochs=5, lr=1e-3, device='cuda', ema_decay=0.9999):
    optimizer = torch.optim.AdamW(method.model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    method.model.to(device)
    method.model.train()
    ema = EMA(method.model, decay=ema_decay)

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
            torch.nn.utils.clip_grad_norm_(method.model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(method.model)

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()
        epoch_losses.append(total_loss / len(dataloader))

    ema.apply(method.model)
    return epoch_losses


def train_reflow(method, paired_dataset, epochs=5, lr=1e-3, batch_size=128, device='cuda', ema_decay=0.9999):
    """Train a model on pre-generated coupled (x0, x1, labels) pairs for reflow."""
    dataloader = DataLoader(paired_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(method.model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    method.model.to(device)
    method.model.train()
    ema = EMA(method.model, decay=ema_decay)

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
            torch.nn.utils.clip_grad_norm_(method.model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(method.model)

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()
        epoch_losses.append(total_loss / len(dataloader))

    ema.apply(method.model)
    return epoch_losses
