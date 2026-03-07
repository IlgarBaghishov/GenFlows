import torch
import torch.nn.functional as F

class FlowMatching:
    def __init__(self, model, drop_prob=0.1):
        self.model = model
        self.num_classes = model.num_classes
        self.drop_prob = drop_prob

    def compute_loss(self, x1, labels):
        x0 = torch.randn_like(x1)
        t = torch.rand((x1.shape[0],), device=x1.device)

        t_expand = t[:, None, None, None]
        xt = (1 - t_expand) * x0 + t_expand * x1

        # Conditional instantaneous velocity (ground truth)
        v_target = x1 - x0

        # Randomly drop labels for classifier-free guidance training
        drop_mask = torch.rand(x1.shape[0], device=x1.device) < self.drop_prob
        labels = labels.clone()
        labels[drop_mask] = self.num_classes

        v_pred = self.model(xt, t, labels)
        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def sample(self, shape, device, labels=None, cfg_scale=3.0, n_steps=50):
        null_labels = torch.full((shape[0],), self.num_classes, device=device, dtype=torch.long)

        # We start from noise (t=0) and integrate to t=1
        x = torch.randn(shape, device=device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((shape[0],), i * dt, device=device)

            if labels is not None and cfg_scale > 0:
                v_cond = self.model(x, t, labels)
                v_uncond = self.model(x, t, null_labels)
                v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v_pred = self.model(x, t, null_labels)

            x = x + v_pred * dt  # Euler step integration
        return x
