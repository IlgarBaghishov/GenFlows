# Inpainting Interactive Training

Iterative training of the Flow Matching 3D inpainting model with checkpointing every 25 epochs. Designed for interactive SLURM sessions (4 nodes, 16 GPUs).

## Workflow

```bash
# Train 25 epochs (auto-resumes from last checkpoint)
srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=4 \
  --rdzv_backend=c10d --rdzv_endpoint=$(hostname):29500 train.py

# Sample from the latest checkpoint
python sample.py

# Or sample from a specific epoch
python sample.py --epoch 50

# Open notebook to inspect results
# sample_and_plot.ipynb
```

Repeat the train/sample cycle until satisfied or session ends.

## Autonomous Operation

When assisting with iterative training, act proactively:
1. After training completes, immediately run sampling and generate plots
2. Analyze results (loss trend, visual quality, NTG, well conditioning)
3. If quality is insufficient, start the next training segment without waiting
4. Keep the user informed of observations and any concerns
5. Do NOT wait for user prompts between train/sample/plot cycles

## What to Track During Training

### Loss Curve
- **Healthy**: loss decreases steadily, may plateau at lower values
- **Loss oscillation**: normal when LR is still moderate (early/mid training). Oscillation decreases as cosine LR decays toward zero near the end of the 350-epoch horizon
- **Red flags**: loss increasing over many epochs, sudden spikes that don't recover, loss stuck at a high value (not learning)
- **Typical range**: starts ~0.1-0.5, should decrease to ~0.01-0.05 range for well-trained models

### Visual Quality (in notebook)
- **Good**: elongated sand bodies (lobes) in a shale background, smooth boundaries, geologically plausible structures
- **Bad at early epochs (expected)**: noisy, random-looking, no clear lobe structure — this is normal and should improve
- **Red flags at later epochs**: still noisy after 100+ epochs, output is all-one-value (collapsed), visible grid artifacts

### Boundary Continuity
- At mask boundaries (red contour in plots), the generated region should smoothly transition from known voxels
- **Good**: lobes that cross the mask boundary look continuous, no abrupt changes
- **Bad**: visible seams, hard edges, or texture changes at the boundary

### Metrics
- **NTG (net-to-gross)**: fraction of sand (1s) in the volume. Should approach ground truth values (~0.5-0.6)
  - Too low (~0.3-0.4): model underfilling, generating too much shale
  - Too high (~0.8+): model overfilling
  - Getting closer to GT NTG across epochs = good
- **Known-region match rate**: should be 100% (hard replacement at final step guarantees this)

## File Organization

```
checkpoints/
  cond_stats.npz              # conditioning normalization (height, radius, etc.)
  training_state.pt           # full checkpoint for resume (model + optimizer + EMA + epoch + losses)
  loss_history.npy            # cumulative loss per epoch across all segments
  inference_epoch025.pt       # EMA-applied model weights at epoch 25
  inference_epoch050.pt       # ... at epoch 50
  flow_matching_epoch025.pt   # copy of inference checkpoint (for sample.py)
  flow_matching_epoch050.pt
  flow_matching.pt            # latest checkpoint (convenience symlink)
results/
  loss_curve.png              # full loss plot
  epoch_025/                  # sampling results at epoch 25
    ground_truth.npy
    ground_truth_cond.npy
    flow_matching_{scenario}_50steps.npy
    mask_{scenario}.npy
  epoch_050/                  # sampling results at epoch 50
    ...
```

## Architecture Notes

- Model: UNet3D with `in_channels=3` (noisy_x + known_data + mask), `out_channels=1`
- Method: Flow Matching with OT coupling, Euler integration, 50 sampling steps
- CFG: classifier-free guidance with scale=3.0, 10% drop rate during training
- EMA: decay=0.9999, applied to inference checkpoints
- Optimizer: AdamW, lr=1e-3, cosine LR over 350-epoch horizon (17 warmup + 333 cosine decay), continuous across segments via scheduler state restore
- Mask convention: 1 = known (keep), 0 = unknown (generate)
