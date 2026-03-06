import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator


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


def train_model(method, dataloader, epochs=5, lr=1e-3, ema_decay=0.9999, accelerator=None):
    if accelerator is None:
        accelerator = Accelerator()

    optimizer = torch.optim.AdamW(method.model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    method.model, optimizer, dataloader, scheduler = accelerator.prepare(
        method.model, optimizer, dataloader, scheduler
    )

    method.model.train()
    ema = EMA(method.model, decay=ema_decay)
    use_ema_target = hasattr(method, 'compute_loss') and 'target_params' in method.compute_loss.__code__.co_varnames

    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", disable=not accelerator.is_main_process)
        for x, labels in pbar:
            optimizer.zero_grad()
            # For MeanFlow, pass EMA shadow weights as stable JVP target (target network)
            if use_ema_target:
                loss = method.compute_loss(x, labels, target_params=ema.shadow)
            else:
                loss = method.compute_loss(x, labels)
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(method.model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(method.model)

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()
        epoch_losses.append(total_loss / num_batches)

    ema.apply(method.model)
    # Unwrap DDP model so method.model is the raw module again
    method.model = accelerator.unwrap_model(method.model)
    return epoch_losses


def train_reflow(method, paired_dataset, epochs=5, lr=1e-3, batch_size=128, ema_decay=0.9999, accelerator=None):
    """Train a model on pre-generated coupled (x0, x1, labels) pairs for reflow."""
    if accelerator is None:
        accelerator = Accelerator()

    dataloader = DataLoader(paired_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(method.model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    method.model, optimizer, dataloader, scheduler = accelerator.prepare(
        method.model, optimizer, dataloader, scheduler
    )

    method.model.train()
    ema = EMA(method.model, decay=ema_decay)

    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", disable=not accelerator.is_main_process)
        for x0, x1, labels in pbar:
            optimizer.zero_grad()
            loss = method.compute_loss(x1, labels, x0=x0)
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(method.model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(method.model)

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()
        epoch_losses.append(total_loss / num_batches)

    ema.apply(method.model)
    method.model = accelerator.unwrap_model(method.model)
    return epoch_losses
