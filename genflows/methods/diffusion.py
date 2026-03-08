import torch
import torch.nn.functional as F

class Diffusion:
    def __init__(self, model, n_steps=1000, beta_min=1e-4, beta_max=0.02, drop_prob=0.1):
        self.model = model
        self.n_steps = n_steps
        self.drop_prob = drop_prob
        self.betas = torch.linspace(beta_min, beta_max, n_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def compute_loss(self, x0, cond):
        t = torch.randint(0, self.n_steps, (x0.shape[0],), device=x0.device)
        eps = torch.randn_like(x0)

        alpha_bar_t = self.alpha_bars.to(x0.device)[t].view(-1, *([1] * (x0.ndim - 1)))
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps

        drop_mask = torch.rand(x0.shape[0], device=x0.device) < self.drop_prob
        t_norm = t.float() / self.n_steps
        eps_pred = self.model(xt, t_norm, cond, drop_mask=drop_mask)

        return F.mse_loss(eps_pred, eps)

    @torch.no_grad()
    def sample(self, shape, device, cond=None, cfg_scale=3.0, n_steps=None, sampler='ddpm', eta=0.0):
        """Sample images.

        Args:
            sampler: 'ddpm' (standard, stochastic) or 'ddim' (deterministic when eta=0).
            eta: DDIM noise scale. 0=deterministic, 1=same variance as DDPM. Only used when sampler='ddim'.
        """
        if n_steps is None: n_steps = self.n_steps

        x = torch.randn(shape, device=device)

        # Evenly spaced timesteps across the full schedule, starting from the noisiest
        stride = self.n_steps // n_steps
        timesteps = list(range(self.n_steps - 1, -1, -stride))[:n_steps]

        for idx, i in enumerate(timesteps):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            t_norm = t.float() / self.n_steps

            if cond is not None and cfg_scale > 0:
                eps_cond = self.model(x, t_norm, cond)
                eps_uncond = self.model(x, t_norm)
                eps_pred = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            else:
                eps_pred = self.model(x, t_norm)

            if sampler == 'ddpm':
                alpha_bar_t = self.alpha_bars.to(device)[i]
                if idx < len(timesteps) - 1:
                    alpha_bar_prev = self.alpha_bars.to(device)[timesteps[idx + 1]]
                else:
                    alpha_bar_prev = torch.tensor(1.0, device=device)

                # Predict x0
                x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
                x0_pred = x0_pred.clamp(-1, 1)

                # Posterior mean and variance for arbitrary stride
                posterior_mean = (
                    torch.sqrt(alpha_bar_prev) * (1 - alpha_bar_t / alpha_bar_prev) / (1 - alpha_bar_t) * x0_pred
                    + torch.sqrt(alpha_bar_t / alpha_bar_prev) * (1 - alpha_bar_prev) / (1 - alpha_bar_t) * x
                )
                posterior_var = (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)

                if idx < len(timesteps) - 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = posterior_mean + torch.sqrt(posterior_var) * noise

            elif sampler == 'ddim':
                alpha_bar_t = self.alpha_bars.to(device)[i]
                if idx < len(timesteps) - 1:
                    alpha_bar_prev = self.alpha_bars.to(device)[timesteps[idx + 1]]
                else:
                    alpha_bar_prev = torch.tensor(1.0, device=device)

                # Predict x0
                x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
                x0_pred = x0_pred.clamp(-1, 1)
                # Recompute eps consistent with clipped x0 (prevents CFG drift)
                eps_pred = (x - torch.sqrt(alpha_bar_t) * x0_pred) / torch.sqrt(1 - alpha_bar_t)

                # DDIM variance: eta=0 deterministic, eta=1 matches DDPM
                sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
                dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps_pred

                noise = torch.randn_like(x) if idx < len(timesteps) - 1 else torch.zeros_like(x)
                x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma * noise

        return x
