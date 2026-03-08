import torch
import torch.nn.functional as F

class MeanFlow:
    def __init__(self, model, drop_prob=0.1, cfg_mode='standard', omega=3.0, kappa=0.0):
        self.model = model
        self.drop_prob = drop_prob
        self.cfg_mode = cfg_mode  # 'standard' or 'embedded'
        self.omega = omega        # guidance scale for embedded CFG (Eq. 21)
        self.kappa = kappa        # mixing scale for improved embedded CFG (Eq. 20)

    def compute_loss(self, x_data, cond, target_params=None):
        device = x_data.device
        b = x_data.shape[0]

        x_noise = torch.randn_like(x_data)

        # Sample time variables r and t
        t_rand = torch.rand(b, device=device)
        r_rand = torch.rand(b, device=device)

        # Enforce t > r
        t = torch.max(t_rand, r_rand)
        r = torch.min(t_rand, r_rand)

        # Set r = t with 10% probability
        mask = torch.rand(b, device=device) < 0.1
        r = torch.where(mask, t, r)

        # Conditional path: t=1 is noise, t=0 is data
        t_expand = t.view(-1, *([1] * (x_data.ndim - 1)))
        xt = t_expand * x_noise + (1 - t_expand) * x_data
        v_t = x_noise - x_data  # Instantaneous velocity dz_t/dt

        # Compute effective velocity: ṽ_t (Eq. 21)
        # For standard mode: v_eff = v_t (no CFG in target)
        # For embedded mode: v_eff = ω·v_t + κ·u_cond(zt,t,t) + (1-ω-κ)·u_uncond(zt,t,t)
        if self.cfg_mode == 'embedded':
            with torch.no_grad():
                zero_dt = torch.zeros_like(t)
                u_cond = self.model(xt, t, zero_dt, cond)
                u_uncond = self.model(xt, t, zero_dt)
            v_eff = self.omega * v_t + self.kappa * u_cond + (1 - self.omega - self.kappa) * u_uncond
        else:
            v_eff = v_t

        # CFG dropping: method decides which samples to drop, model applies it
        drop_mask = torch.rand(b, device=device) < self.drop_prob

        # Predict average velocity: u_theta(xt, t, t - r)
        u_pred = self.model(xt, t, t - r, cond, drop_mask=drop_mask)

        # MeanFlow Identity Target: v_eff - (t - r) * (v_eff·∂_z u + ∂_t u)
        with torch.no_grad():
            # Unwrap DDP model for torch.func compatibility (JVP is no_grad target-only,
            # so no distributed sync is lost — the trainable forward pass above still uses DDP)
            raw_model = self.model.module if hasattr(self.model, 'module') else self.model
            # Use EMA target params for stable JVP target (like a target network),
            # fall back to live weights if no target params provided
            if target_params is not None:
                # Strip 'module.' prefix from DDP-wrapped EMA keys to match raw_model
                prefix = 'module.'
                params = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in target_params.items()}
            else:
                params = dict(raw_model.named_parameters())
            buffers = dict(raw_model.named_buffers())

            def stateless_u_fn(z_in, t_in, r_in):
                return torch.func.functional_call(
                    raw_model, (params, buffers),
                    (z_in, t_in, t_in - r_in, cond),
                    {'drop_mask': drop_mask}
                )

            _, jvp_out = torch.func.jvp(
                stateless_u_fn,
                (xt, t, r),
                (v_eff, torch.ones_like(t), torch.zeros_like(r))
            )

            dt = (t - r).view(-1, *([1] * (x_data.ndim - 1)))
            u_tgt = v_eff - dt * jvp_out

        return F.mse_loss(u_pred, u_tgt.detach())

    @torch.no_grad()
    def sample(self, shape, device, cond=None, cfg_scale=3.0, n_steps=1):
        # We start from noise (t=1) and go to data (t=0).
        x = torch.randn(shape, device=device)

        # Time steps from 1 to 0
        t_vals = torch.linspace(1.0, 0.0, n_steps + 1, device=device)

        for i in range(n_steps):
            t = torch.full((shape[0],), t_vals[i].item(), device=device)
            r = torch.full((shape[0],), t_vals[i+1].item(), device=device)

            if self.cfg_mode == 'embedded':
                # CFG is baked into the model — 1 NFE per step
                if cond is not None:
                    u_pred = self.model(x, t, t - r, cond)
                else:
                    u_pred = self.model(x, t, t - r)
            elif cond is not None and cfg_scale > 0:
                # Standard CFG at sampling time — 2 NFE per step
                u_cond = self.model(x, t, t - r, cond)
                u_uncond = self.model(x, t, t - r)
                u_pred = u_uncond + cfg_scale * (u_cond - u_uncond)
            else:
                u_pred = self.model(x, t, t - r)

            dt = (t - r).view(-1, *([1] * (x.ndim - 1)))
            x = x - dt * u_pred

        return x
