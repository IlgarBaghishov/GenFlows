import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import SinusoidalPosEmb


class DownBlock3D(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_c)
        self.conv1 = nn.Conv3d(in_c, out_c, 3, padding=1)
        self.conv2 = nn.Conv3d(out_c, out_c, 3, padding=1)
        self.downsample = nn.MaxPool3d(2)

        self.gn1 = nn.GroupNorm(8, out_c)
        self.gn2 = nn.GroupNorm(8, out_c)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.gn1(self.act(self.conv1(x)))
        time_emb = self.act(self.time_mlp(t_emb))
        h = h + time_emb[..., None, None, None]
        h = self.gn2(self.act(self.conv2(h)))
        return self.downsample(h), h


class UpBlock3D(nn.Module):
    def __init__(self, in_c, skip_c, out_c, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_c)
        self.conv1 = nn.Conv3d(in_c + skip_c, out_c, 3, padding=1)
        self.conv2 = nn.Conv3d(out_c, out_c, 3, padding=1)

        self.gn1 = nn.GroupNorm(8, out_c)
        self.gn2 = nn.GroupNorm(8, out_c)
        self.act = nn.SiLU()

    def forward(self, x, res, t_emb):
        x = F.interpolate(x, size=res.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat((x, res), dim=1)
        h = self.gn1(self.act(self.conv1(x)))
        time_emb = self.act(self.time_mlp(t_emb))
        h = h + time_emb[..., None, None, None]
        h = self.gn2(self.act(self.conv2(h)))
        return h


class UNet3D(nn.Module):
    """3D UNet for volumetric generative modeling with continuous conditioning.

    Conditioning inputs (height, radius, aspect_ratio, angle_deg, ntg) are
    expected to be normalized to [0, 1] by the dataset. Angle is internally
    converted to sin(2*pi*angle_norm) and cos(2*pi*angle_norm) to handle
    the 180-degree periodicity of lobe orientation.
    """

    def __init__(self, in_channels=1, hidden_dims=None, time_dim=256,
                 num_cond=5, num_time_embs=1, out_channels=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64, 128, 128]
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.time_dim = time_dim
        self.num_time_embs = num_time_embs
        self.num_cond = num_cond

        # Inpaint context (set via set_inpaint_context for channel concat)
        self._inpaint_mask = None
        self._inpaint_data = None

        # Time embedding
        sub_dim = time_dim // num_time_embs
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(sub_dim),
            nn.Linear(sub_dim, sub_dim),
            nn.SiLU()
        )
        self.joint_time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )

        # Conditioning embedding
        # 5 inputs -> angle becomes sin+cos -> 6 values -> MLP -> time_dim
        cond_input_dim = num_cond + 1  # angle replaced by sin, cos
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_input_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )

        # Learned null embedding for classifier-free guidance
        self.null_cond_emb = nn.Parameter(torch.randn(time_dim))

        # Encoder
        self.init_conv = nn.Conv3d(in_channels, hidden_dims[0], 3, padding=1)

        self.downs = nn.ModuleList()
        in_c = hidden_dims[0]
        channels = [hidden_dims[0]]
        for out_c in hidden_dims[1:]:
            self.downs.append(DownBlock3D(in_c, out_c, time_dim))
            channels.append(out_c)
            in_c = out_c

        # Mid block
        self.mid_block1 = nn.Conv3d(hidden_dims[-1], hidden_dims[-1], 3, padding=1)
        self.mid_gn1 = nn.GroupNorm(8, hidden_dims[-1])
        self.mid_time_mlp = nn.Linear(time_dim, hidden_dims[-1])
        self.mid_block2 = nn.Conv3d(hidden_dims[-1], hidden_dims[-1], 3, padding=1)
        self.mid_gn2 = nn.GroupNorm(8, hidden_dims[-1])
        self.mid_act = nn.SiLU()

        # Decoder
        self.ups = nn.ModuleList()
        for skip_c, out_c in zip(reversed(channels[1:]), reversed(hidden_dims[:-1])):
            self.ups.append(UpBlock3D(in_c, skip_c, out_c, time_dim))
            in_c = out_c

        # Final conv
        self.final_conv = nn.Sequential(
            nn.Conv3d(hidden_dims[0], hidden_dims[0], 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden_dims[0], out_channels, 1)
        )

    def set_inpaint_context(self, mask, data):
        """Store inpaint mask and known data for channel concatenation in forward.

        Args:
            mask: (B, 1, D, H, W) float tensor, 1=known 0=unknown
            data: (B, 1, D, H, W) float tensor, clean values where mask=1
        """
        self._inpaint_mask = mask.detach()
        self._inpaint_data = data.detach()

    def clear_inpaint_context(self):
        """Remove stored inpaint context."""
        self._inpaint_mask = None
        self._inpaint_data = None

    def _process_conditioning(self, cond):
        """Convert raw conditioning to model input.

        Args:
            cond: (B, 5) tensor [height, radius, aspect_ratio, angle, ntg]
                  all normalized to [0, 1] by the dataset.

        Returns:
            (B, 6) tensor [height, radius, aspect_ratio, sin, cos, ntg]
        """
        angle_norm = cond[:, 3:4]
        sin_angle = torch.sin(2 * math.pi * angle_norm)
        cos_angle = torch.cos(2 * math.pi * angle_norm)
        return torch.cat([cond[:, :3], sin_angle, cos_angle, cond[:, 4:5]], dim=1)

    def forward(self, x, *args, drop_mask=None):
        # Parse args: last 2D tensor is conditioning, rest are time(s)
        # drop_mask: optional BoolTensor (B,), True = replace with null embedding for CFG
        if len(args) > 1 and args[-1].dim() == 2:
            times = args[:-1]
            cond = args[-1]
        else:
            times = args
            cond = None

        # Time embedding
        t_embs = [self.time_mlp(t) for t in times]
        if len(t_embs) < self.num_time_embs:
            t_embs.extend([torch.zeros_like(t_embs[0])] * (self.num_time_embs - len(t_embs)))
        t_emb = torch.cat(t_embs, dim=-1)
        t_emb = self.joint_time_mlp(t_emb)

        # Conditioning embedding
        if cond is not None:
            cond_processed = self._process_conditioning(cond)
            c_emb = self.cond_mlp(cond_processed)

            if drop_mask is not None:
                c_emb = c_emb.clone()
                c_emb[drop_mask] = self.null_cond_emb
        else:
            c_emb = self.null_cond_emb.unsqueeze(0).expand(x.shape[0], -1)

        t_emb = t_emb + c_emb

        # Inpaint channel concatenation (only for in_channels > 1)
        if self.in_channels > 1:
            if self._inpaint_mask is not None:
                x = torch.cat([x, self._inpaint_data, self._inpaint_mask], dim=1)
            else:
                zeros = torch.zeros(
                    x.shape[0], self.in_channels - 1, *x.shape[2:],
                    device=x.device, dtype=x.dtype
                )
                x = torch.cat([x, zeros], dim=1)

        # Encoder
        x = self.init_conv(x)
        res_stack = [x]

        for down in self.downs:
            x, res = down(x, t_emb)
            res_stack.append(res)

        # Mid
        x = self.mid_gn1(self.mid_act(self.mid_block1(x)))
        x = x + self.mid_act(self.mid_time_mlp(t_emb))[..., None, None, None]
        x = self.mid_gn2(self.mid_act(self.mid_block2(x)))

        # Decoder
        for up in self.ups:
            res = res_stack.pop()
            x = up(x, res, t_emb)

        return self.final_conv(x)
