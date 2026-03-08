# GenFlows

A unified implementation of modern generative modeling methods with a modular, model-agnostic design. Methods are pure math; models handle all conditioning details. Any method works with any model out of the box.

| Method | What it learns | Sampling | Key idea |
|---|---|---|---|
| **Diffusion** (DDPM/DDIM) | Noise prediction | Iterative denoising (stochastic or deterministic) | Reverse a gradual noising process |
| **Flow Matching** | Velocity field | Euler ODE integration | Learn the optimal transport map from noise to data |
| **Rectified Flow** | Velocity field (straighter) | Euler ODE integration | Reflow: straighten ODE paths for fewer-step generation |
| **MeanFlow** | Mean velocity | Single-step capable | JVP-based training enables one-step generation |

Currently supports two datasets:

| Dataset | Dimensionality | Conditioning | Model |
|---|---|---|---|
| **MNIST** | 2D (32x32 images) | Discrete class labels (10 digits) | UNet |
| **Lobes** | 3D (50x50x50 voxels) | Continuous parameters (height, radius, aspect ratio, angle, NTG) | UNet3D |

All methods share the same training loop, optimizer, and EMA setup -- making this a true apples-to-apples comparison. Every method supports conditional generation with classifier-free guidance (CFG). Training supports multi-GPU/multi-node via HuggingFace Accelerate.

## Why this repo?

Most generative modeling repos implement a single method in isolation. If you want to understand how diffusion, flow matching, rectified flow, and MeanFlow actually compare, you'd need to stitch together 4+ separate codebases with different architectures, different data preprocessing, and different training setups -- making any comparison meaningless.

This repo puts all methods on equal footing: same optimizer, same data pipeline, same training loop. The methods are model-agnostic -- swap UNet for UNet3D (or a future DiT) and everything works. The entire package is ~900 lines of PyTorch.

## Quick start

```bash
pip install -e .
```

### MNIST (2D)

```bash
python examples/mnist/train.py      # Train all 8 models
python examples/mnist/sample.py     # Generate samples from checkpoints
accelerate launch examples/mnist/train.py  # Multi-GPU
```

### Lobes (3D geological volumes)

```bash
python examples/lobes/train.py      # Train all 8 models
python examples/lobes/sample.py     # Generate 3D samples from checkpoints
accelerate launch examples/lobes/train.py  # Multi-GPU
```

Training saves 8 checkpoints to `checkpoints/` and loss curves to `results/`. Sampling loads the checkpoints and generates conditional samples at various step counts.

## Project structure

```
genflows/
├── models/
│   ├── unet.py             # 2D UNet: class-conditional (learned embedding + null token)
│   └── unet3d.py           # 3D UNet: continuous-conditional (MLP + learned null embedding)
├── methods/
│   ├── diffusion.py        # DDPM + DDIM sampling (noise prediction, linear beta schedule)
│   ├── flow_matching.py    # Flow Matching (velocity prediction, Euler integration)
│   ├── meanflow.py         # MeanFlow (mean velocity, JVP-based training, 1-step capable)
│   └── rectified_flow.py   # Rectified Flow (forward/backward/bidirectional reflow)
└── utils/
    ├── data.py             # MNIST loading (padded to 32x32, normalized to [-1,1])
    ├── data_lobes.py       # Lobe dataset: facies loading, NTG computation, filtering, normalization
    ├── training.py         # Training loops with EMA target network support
    └── plotting.py         # Sample grids and loss curves

examples/
├── mnist/
│   ├── train.py            # Train all methods on MNIST
│   └── sample.py           # Generate MNIST digit grids
└── lobes/
    ├── train.py            # Train all methods on geological lobes
    ├── sample.py           # Generate 3D lobe volumes
    └── data/               # facies.npy, parameters.csv, failed_cases.npy
```

## Architecture: Model ↔ Method interface

Methods and models communicate through a minimal, three-call interface:

```python
model(x, t, cond)                # conditional forward pass
model(x, t)                      # unconditional (model uses its own null representation)
model(x, t, cond, drop_mask=m)   # mixed batch for training-time CFG
```

**Methods** decide *when* to drop conditioning and *how* to combine conditional/unconditional predictions at sampling time. **Models** decide *what* "unconditional" means internally (null class token, learned null embedding, etc.). This separation means any method works with any model -- swap UNet for UNet3D (or a future DiT/ViT) without touching the method code.

## Methods at a glance

### Diffusion (DDPM/DDIM)

Learns to predict the noise added at each timestep. Supports two samplers: DDPM (stochastic, with proper posterior variance for arbitrary step counts) and DDIM (deterministic when eta=0, allows fewer steps). DDIM clips x0 predictions and recomputes eps for consistency, preventing drift under classifier-free guidance. Uses a linear beta schedule with 1000 steps.

### Flow Matching

Learns a velocity field that transports samples from a standard Gaussian to the data distribution along straight-line conditional paths. Sampling integrates the learned ODE with Euler steps.

### Rectified Flow (2-Rectified Flow)

Starts from a trained Flow Matching model. Generates coupled (noise, data) pairs by integrating the ODE, then trains a new model on these pairs. The "reflow" operation straightens the ODE trajectories, enabling high-quality generation with fewer steps. Four variants are trained:

- **Forward (random init)**: fresh model trained on forward ODE pairs
- **Forward (warm start)**: initialized from FM weights, trained on forward pairs
- **Backward (warm start)**: initialized from FM weights, trained on backward ODE pairs (exact data → approximate noise)
- **Bidirectional (warm start)**: trained on concatenated forward + backward pairs

### MeanFlow

Learns the *mean* velocity over a time interval rather than the instantaneous velocity. Training uses Jacobian-vector products (`torch.func.jvp`) to compute the MeanFlow identity target. The JVP target uses EMA shadow weights as a target network for stable training. Supports two CFG modes:

- **Standard CFG**: guidance applied at sampling time (2 network evaluations per step)
- **Embedded CFG**: guidance baked into the training target (Section 4.2 of the paper), enabling single-step generation with just 1 network evaluation

## Models

### UNet (2D, MNIST)

- Hidden dims [64, 128, 256], 2 down/up stages (32→16→8)
- Class conditioning: `nn.Embedding(num_classes + 1, time_dim)`, null token for unconditional
- Strided conv downsampling, ConvTranspose upsampling

### UNet3D (3D, Lobes)

- Hidden dims [64, 64, 128, 128], 3 down/up stages (50→25→12→6)
- MaxPool3d downsampling, trilinear interpolation upsampling (handles odd spatial dims)
- Continuous conditioning: 5 raw inputs → angle internally converted to sin/cos for 180° periodicity → MLP → embedding added to time embedding
- Learned null embedding vector for CFG unconditional
- Lobe NTG computed from actual voxel data (fraction of 1s), not from simulator input parameters

## Implementation details

- **Training**: AdamW optimizer, lr=1e-3 with cosine annealing, gradient clipping (max_norm=1.0), EMA (decay=0.9999). EMA shadow weights double as a target network for MeanFlow's JVP computation
- **CFG**: Methods create a `drop_mask` (10% probability) and pass it to the model via `drop_mask=` kwarg. Models apply the mask using their own null representation. At sampling time: `output = uncond + cfg_scale * (cond - uncond)`, default `cfg_scale=3.0`
- **MNIST**: 28x28 padded to 32x32, batch size 128
- **Lobes**: 50x50x50 binary voxels mapped to [-1,1], batch size 32. Samples with NTG < 0.05 or > 0.95 filtered out (~89k samples remain from 100k)

## References

- **DDPM**: Ho et al., [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (2020)
- **DDIM**: Song et al., [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) (2020)
- **Flow Matching**: Lipman et al., [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) (2022)
- **Rectified Flow**: Liu et al., [Flow Straight and Fast](https://arxiv.org/abs/2209.03003) (2022)
- **MeanFlow**: Geng et al., [Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447) (2025)

## License

MIT
