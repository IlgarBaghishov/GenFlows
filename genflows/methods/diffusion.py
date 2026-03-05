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

    def compute_loss(self, x0, labels):
        t = torch.randint(0, self.n_steps, (x0.shape[0],), device=x0.device)
        eps = torch.randn_like(x0)

        alpha_bar_t = self.alpha_bars.to(x0.device)[t][:, None, None, None]
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps

        # Randomly drop labels for classifier-free guidance training
        drop_mask = torch.rand(x0.shape[0], device=x0.device) < self.drop_prob
        labels = labels.clone()
        labels[drop_mask] = self.model.num_classes  # null class token

        t_norm = t.float() / self.n_steps
        eps_pred = self.model(xt, t_norm, labels)

        return F.mse_loss(eps_pred, eps)

    @torch.no_grad()
    def sample(self, shape, device, labels=None, cfg_scale=3.0, n_steps=None, sampler='ddpm', eta=0.0):
        """Sample images.

        Args:
            sampler: 'ddpm' (standard, stochastic) or 'ddim' (deterministic when eta=0).
            eta: DDIM noise scale. 0=deterministic, 1=same variance as DDPM. Only used when sampler='ddim'.
        """
        if n_steps is None: n_steps = self.n_steps
        null_labels = torch.full((shape[0],), self.model.num_classes, device=device, dtype=torch.long)

        x = torch.randn(shape, device=device)

        # Evenly spaced timesteps across the full schedule, starting from the noisiest
        stride = self.n_steps // n_steps
        timesteps = list(range(self.n_steps - 1, -1, -stride))[:n_steps]

        for idx, i in enumerate(timesteps):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            t_norm = t.float() / self.n_steps

            if labels is not None and cfg_scale > 0:
                eps_cond = self.model(x, t_norm, labels)
                eps_uncond = self.model(x, t_norm, null_labels)
                eps_pred = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            else:
                eps_pred = self.model(x, t_norm, null_labels)

            if sampler == 'ddpm':
                alpha_t = self.alphas.to(device)[i]
                alpha_bar_t = self.alpha_bars.to(device)[i]
                beta_t = self.betas.to(device)[i]

                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = 1 / torch.sqrt(alpha_t) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * eps_pred) + torch.sqrt(beta_t) * noise

            elif sampler == 'ddim':
                alpha_bar_t = self.alpha_bars.to(device)[i]
                if idx < len(timesteps) - 1:
                    alpha_bar_prev = self.alpha_bars.to(device)[timesteps[idx + 1]]
                else:
                    alpha_bar_prev = torch.tensor(1.0, device=device)

                # Predict x0
                x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
                x0_pred = x0_pred.clamp(-1, 1)

                # DDIM variance: eta=0 deterministic, eta=1 matches DDPM
                sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
                dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps_pred

                noise = torch.randn_like(x) if idx < len(timesteps) - 1 else torch.zeros_like(x)
                x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma * noise

        return x
