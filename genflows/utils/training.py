import math
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator


def _make_scheduler(optimizer, epochs, world_size):
    """Create LR scheduler with sqrt-scaled LR and linear warmup for multi-GPU."""
    # Scale LR by sqrt(world_size) for AdamW
    if world_size > 1:
        scale = math.sqrt(world_size)
        for pg in optimizer.param_groups:
            pg['lr'] *= scale

    warmup_epochs = max(1, epochs // 20)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0 / warmup_epochs, total_iters=warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_epochs]
    )


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
    scheduler = _make_scheduler(optimizer, epochs, accelerator.num_processes)

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


def _strip_module_prefix(state_dict):
    """Strip 'module.' prefix from DDP state dict keys."""
    return {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}


def _save_inference_checkpoint(method, ema, accelerator, path):
    """Save EMA-applied model weights for inference (backup/restore live weights)."""
    raw = accelerator.unwrap_model(method.model)
    backup = {k: v.clone() for k, v in raw.state_dict().items()}
    ema.apply(method.model)
    torch.save(raw.state_dict(), path)
    raw.load_state_dict(backup)


def train_model_inpaint(method, dataloader, epochs=5, lr=1e-3, ema_decay=0.9999,
                        accelerator=None, checkpoint_dir=None, save_every=None,
                        total_epochs=None):
    """Training loop for inpainting models. Dataloader yields (x, cond, mask).

    Args:
        epochs: Number of epochs to train in this invocation.
        checkpoint_dir: If set, enables checkpointing. Auto-resumes from
            training_state.pt if it exists.
        save_every: Save inference checkpoint every N epochs. Full training
            state is saved at the same interval for resume.
        total_epochs: Total training horizon for LR scheduler (e.g. 350).
            If None, defaults to epochs. When resuming, the scheduler state
            is restored so LR decays smoothly across segments.
    """
    if total_epochs is None:
        total_epochs = epochs
    if accelerator is None:
        accelerator = Accelerator()

    # Check for existing checkpoint to resume from
    resume_path = None
    if checkpoint_dir and os.path.exists(os.path.join(checkpoint_dir, 'training_state.pt')):
        resume_path = os.path.join(checkpoint_dir, 'training_state.pt')
        accelerator.print(f"Found checkpoint at {resume_path}, resuming...")

    optimizer = torch.optim.AdamW(method.model.parameters(), lr=lr)
    scheduler = _make_scheduler(optimizer, total_epochs, accelerator.num_processes)

    method.model, optimizer, dataloader, scheduler = accelerator.prepare(
        method.model, optimizer, dataloader, scheduler
    )

    # Resume: load model, optimizer, scheduler, EMA state
    start_epoch = 0
    prior_losses = []
    if resume_path:
        ckpt = torch.load(resume_path, map_location=accelerator.device, weights_only=False)
        raw = accelerator.unwrap_model(method.model)
        raw.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        else:
            # Fast-forward scheduler for checkpoints without scheduler state
            for _ in range(ckpt['epoch']):
                scheduler.step()
        start_epoch = ckpt['epoch']
        prior_losses = ckpt.get('epoch_losses', [])
        accelerator.print(f"Resumed from epoch {start_epoch}, prior loss: {prior_losses[-1]:.4f}")

    method.model.train()
    ema = EMA(method.model, decay=ema_decay)

    # Restore EMA shadow from checkpoint (handle DDP key mapping)
    if resume_path:
        saved_shadow = ckpt['ema_shadow']
        for name in ema.shadow:
            clean = name[7:] if name.startswith('module.') else name
            if clean in saved_shadow:
                ema.shadow[name] = saved_shadow[clean].to(ema.shadow[name].device)
        del ckpt  # free memory

    use_ema_target = (hasattr(method, 'compute_loss')
                      and 'target_params' in method.compute_loss.__code__.co_varnames)

    epoch_losses = list(prior_losses)

    for epoch in range(epochs):
        global_epoch = start_epoch + epoch + 1
        total_loss = 0
        num_batches = 0
        pbar = tqdm(dataloader, desc=f"Epoch {global_epoch} ({epoch+1}/{epochs})",
                    disable=not accelerator.is_main_process)
        for x, cond, mask in pbar:
            # Set inpaint context on the raw (unwrapped) model
            inpaint_data = x * mask
            raw = method.model.module if hasattr(method.model, 'module') else method.model
            raw.set_inpaint_context(mask, inpaint_data)

            optimizer.zero_grad()
            if use_ema_target:
                loss = method.compute_loss(x, cond, target_params=ema.shadow)
            else:
                loss = method.compute_loss(x, cond)
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(method.model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(method.model)

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()
        epoch_losses.append(total_loss / num_batches)

        # Save checkpoints at intervals
        if checkpoint_dir and save_every and global_epoch % save_every == 0:
            if accelerator.is_main_process:
                raw = accelerator.unwrap_model(method.model)
                # Full training state (for resume)
                torch.save({
                    'epoch': global_epoch,
                    'model_state_dict': raw.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'ema_shadow': _strip_module_prefix(ema.shadow),
                    'epoch_losses': epoch_losses,
                }, os.path.join(checkpoint_dir, 'training_state.pt'))
                # Inference checkpoint (EMA-applied weights)
                _save_inference_checkpoint(
                    method, ema, accelerator,
                    os.path.join(checkpoint_dir, f'inference_epoch{global_epoch:03d}.pt'))
                accelerator.print(f"Saved checkpoints at epoch {global_epoch}")
            accelerator.wait_for_everyone()

    # Final save if not already saved at an interval boundary
    final_epoch = start_epoch + epochs
    if checkpoint_dir and accelerator.is_main_process:
        already_saved = save_every and final_epoch % save_every == 0
        if not already_saved:
            raw = accelerator.unwrap_model(method.model)
            torch.save({
                'epoch': final_epoch,
                'model_state_dict': raw.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'ema_shadow': _strip_module_prefix(ema.shadow),
                'epoch_losses': epoch_losses,
            }, os.path.join(checkpoint_dir, 'training_state.pt'))
            _save_inference_checkpoint(
                method, ema, accelerator,
                os.path.join(checkpoint_dir, f'inference_epoch{final_epoch:03d}.pt'))
            accelerator.print(f"Saved checkpoints at epoch {final_epoch}")
        if save_every:
            accelerator.wait_for_everyone()

    # Clean up
    raw = method.model.module if hasattr(method.model, 'module') else method.model
    raw.clear_inpaint_context()
    ema.apply(method.model)
    method.model = accelerator.unwrap_model(method.model)
    return epoch_losses


def train_reflow(method, paired_dataset, epochs=5, lr=1e-3, batch_size=128, ema_decay=0.9999, accelerator=None):
    """Train a model on pre-generated coupled (x0, x1, labels) pairs for reflow."""
    if accelerator is None:
        accelerator = Accelerator()

    dataloader = DataLoader(paired_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(method.model.parameters(), lr=lr)
    scheduler = _make_scheduler(optimizer, epochs, accelerator.num_processes)

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
