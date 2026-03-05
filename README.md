# GenFlows

A unified implementation of four modern generative modeling methods, trained and compared side-by-side on MNIST with a shared architecture.

| Method | What it learns | Sampling | Key idea |
|---|---|---|---|
| **Diffusion** (DDPM/DDIM) | Noise prediction | Iterative denoising (stochastic or deterministic) | Reverse a gradual noising process |
| **Flow Matching** | Velocity field | Euler ODE integration | Learn the optimal transport map from noise to data |
| **Rectified Flow** | Velocity field (straighter) | Euler ODE integration | Reflow: straighten ODE paths for fewer-step generation |
| **MeanFlow** | Mean velocity | Single-step capable | JVP-based training enables one-step generation |

All four methods share the same UNet backbone, the same data pipeline, and the same training loop -- making this a true apples-to-apples comparison. Every method supports class-conditional generation with classifier-free guidance (CFG).

## Why this repo?

Most generative modeling repos implement a single method in isolation. If you want to understand how diffusion, flow matching, rectified flow, and MeanFlow actually compare, you'd need to stitch together 4+ separate codebases with different architectures, different data preprocessing, and different training setups -- making any comparison meaningless.

This repo puts all four methods on equal footing: same UNet, same optimizer, same data. The entire package is ~565 lines of PyTorch. You can train everything, then re-sample as many times as you want without retraining.

## Quick start

```bash
pip install torch torchvision matplotlib tqdm
```

**Train all models and save checkpoints:**
```bash
python examples/train.py
```

**Generate samples from saved checkpoints (no retraining needed):**
```bash
python examples/sample.py
```

Training saves checkpoints to `checkpoints/` and loss curves to `results/`. Sampling loads the checkpoints, generates 10x10 class-conditional grids (rows = digits 0-9) at various step counts (1, 5, 10, 50, 100, 500, 1000), prints wall-clock timing for each, and saves everything to `results/`.

## Project structure

```
genflows/
├── models/unet.py          # Shared UNet with sinusoidal time embeddings and class conditioning
├── methods/
│   ├── diffusion.py        # DDPM + DDIM sampling (noise prediction, linear beta schedule)
│   ├── flow_matching.py    # Flow Matching (velocity prediction, Euler integration)
│   ├── meanflow.py         # MeanFlow (mean velocity, JVP-based training, 1-step capable)
│   └── rectified_flow.py   # Rectified Flow (reflow with coupled pairs for straighter paths)
└── utils/
    ├── data.py             # MNIST loading (padded to 32x32, normalized to [-1,1])
    ├── training.py         # Training loops (standard + reflow)
    └── plotting.py         # Sample grids and loss curves

examples/
├── train.py                # Train all methods, save checkpoints
├── sample.py               # Load checkpoints, generate samples with timing
└── compare_models.py       # All-in-one train + sample script
```

## Methods at a glance

### Diffusion (DDPM/DDIM)

Learns to predict the noise added at each timestep. Supports two samplers: DDPM (stochastic, original formulation) and DDIM (deterministic when eta=0, allows fewer steps). Uses a linear beta schedule with 1000 steps.

### Flow Matching

Learns a velocity field that transports samples from a standard Gaussian to the data distribution along straight-line conditional paths. Sampling integrates the learned ODE with Euler steps.

### Rectified Flow (2-Rectified Flow)

Starts from a trained Flow Matching model. Generates coupled (noise, data) pairs by integrating the ODE, then trains a new model on these pairs. The "reflow" operation straightens the ODE trajectories, enabling high-quality generation with fewer steps.

### MeanFlow

Learns the *mean* velocity over a time interval rather than the instantaneous velocity. Training uses Jacobian-vector products (`torch.func.jvp`) to compute the MeanFlow identity target. Supports two CFG modes:

- **Standard CFG**: guidance applied at sampling time (2 network evaluations per step)
- **Embedded CFG**: guidance baked into the training target (Section 4.2 of the paper), enabling single-step generation with just 1 network evaluation

## Implementation details

- **Architecture**: UNet with hidden dims [64, 128, 256], GroupNorm, SiLU activations, sinusoidal positional embeddings. MeanFlow uses `num_time_embs=2` (for t and t-r); all others use 1.
- **Class conditioning**: Learned embedding for 10 digit classes + 1 null token. Labels randomly dropped with 10% probability during training for CFG.
- **CFG sampling**: `output = uncond + cfg_scale * (cond - uncond)`, default `cfg_scale=3.0`
- **Training**: AdamW optimizer, lr=1e-3, batch size 128
- **Data**: MNIST padded from 28x28 to 32x32 (2px each side) for clean UNet downsampling/upsampling

## References

- **DDPM**: Ho et al., [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (2020)
- **DDIM**: Song et al., [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) (2020)
- **Flow Matching**: Lipman et al., [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) (2022)
- **Rectified Flow**: Liu et al., [Flow Straight and Fast](https://arxiv.org/abs/2209.03003) (2022)
- **MeanFlow**: Geng et al., [Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447) (2025)

## License

MIT
