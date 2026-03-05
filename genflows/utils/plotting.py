import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os

def plot_samples(samples, title, filename, nrow=8):
    # unnormalize from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)
    grid = vutils.make_grid(samples, nrow=nrow, padding=2, normalize=False)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_loss(losses, title, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label='Train Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=150)
    plt.close()
