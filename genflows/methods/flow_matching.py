import torch
import torch.nn.functional as F

class FlowMatching:
    def __init__(self, model, drop_prob=0.1):
        self.model = model
        self.drop_prob = drop_prob

    def compute_loss(self, x1, cond):
        x0 = torch.randn_like(x1)
        t = torch.rand((x1.shape[0],), device=x1.device)

        t_expand = t.view(-1, *([1] * (x1.ndim - 1)))
        xt = (1 - t_expand) * x0 + t_expand * x1

        v_target = x1 - x0

        drop_mask = torch.rand(x1.shape[0], device=x1.device) < self.drop_prob
        v_pred = self.model(xt, t, cond, drop_mask=drop_mask)
        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def sample(self, shape, device, cond=None, cfg_scale=3.0, n_steps=50):
        x = torch.randn(shape, device=device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((shape[0],), i * dt, device=device)

            if cond is not None and cfg_scale > 0:
                v_cond = self.model(x, t, cond)
                v_uncond = self.model(x, t)
                v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v_pred = self.model(x, t)

            x = x + v_pred * dt
        return x
