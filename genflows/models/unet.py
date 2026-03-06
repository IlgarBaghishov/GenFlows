import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DownBlock(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.downsample = nn.Conv2d(out_c, out_c, 4, 2, 1)
        
        self.gn1 = nn.GroupNorm(8, out_c)
        self.gn2 = nn.GroupNorm(8, out_c)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.gn1(self.act(self.conv1(x)))
        time_emb = self.act(self.time_mlp(t_emb))
        h = h + time_emb[..., None, None]
        h = self.gn2(self.act(self.conv2(h)))
        return self.downsample(h), h

class UpBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_c)
        self.upsample = nn.ConvTranspose2d(in_c, in_c, 4, 2, 1)
        self.conv1 = nn.Conv2d(in_c + skip_c, out_c, 3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        
        self.gn1 = nn.GroupNorm(8, out_c)
        self.gn2 = nn.GroupNorm(8, out_c)
        self.act = nn.SiLU()

    def forward(self, x, res, t_emb):
        x = self.upsample(x)
        x = torch.cat((x, res), dim=1)
        h = self.gn1(self.act(self.conv1(x)))
        time_emb = self.act(self.time_mlp(t_emb))
        h = h + time_emb[..., None, None]
        h = self.gn2(self.act(self.conv2(h)))
        return h

class UNet(nn.Module):
    def __init__(self, in_channels=1, hidden_dims=[64, 128, 256], time_dim=256, num_time_embs=1, num_classes=10):
        super().__init__()
        self.time_dim = time_dim
        self.num_time_embs = num_time_embs
        self.num_classes = num_classes

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

        # Class conditioning: num_classes + 1 for the null/unconditional token
        self.class_emb = nn.Embedding(num_classes + 1, time_dim)
        self.class_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )
        
        self.init_conv = nn.Conv2d(in_channels, hidden_dims[0], 3, padding=1)
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        in_c = hidden_dims[0]
        channels = [hidden_dims[0]]
        for out_c in hidden_dims[1:]:
            self.downs.append(DownBlock(in_c, out_c, time_dim))
            channels.append(out_c)
            in_c = out_c
            
        self.mid_block1 = nn.Conv2d(hidden_dims[-1], hidden_dims[-1], 3, padding=1)
        self.mid_gn1 = nn.GroupNorm(8, hidden_dims[-1])
        self.mid_time_mlp = nn.Linear(time_dim, hidden_dims[-1])
        self.mid_block2 = nn.Conv2d(hidden_dims[-1], hidden_dims[-1], 3, padding=1)
        self.mid_gn2 = nn.GroupNorm(8, hidden_dims[-1])
        self.mid_act = nn.SiLU()
        
        for skip_c, out_c in zip(reversed(channels[1:]), reversed(hidden_dims[:-1])):
            self.ups.append(UpBlock(in_c, skip_c, out_c, time_dim))
            in_c = out_c
            
        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_dims[0], hidden_dims[0], 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dims[0], in_channels, 1)
        )

    def forward(self, x, *args):
        # Last argument is class_label if it's a long/int tensor, otherwise all are times
        # Convention: forward(x, t1, ..., class_label)
        # class_label: LongTensor of shape (B,), values 0-9 for digits, num_classes for unconditional
        if len(args) > 0 and args[-1].dtype == torch.long:
            times = args[:-1]
            class_label = args[-1]
        else:
            times = args
            class_label = torch.full((x.shape[0],), self.num_classes, device=x.device, dtype=torch.long)

        t_embs = [self.time_mlp(t) for t in times]
        if len(t_embs) < self.num_time_embs:
            t_embs.extend([torch.zeros_like(t_embs[0])] * (self.num_time_embs - len(t_embs)))

        t_emb = torch.cat(t_embs, dim=-1)
        t_emb = self.joint_time_mlp(t_emb)

        # Add class conditioning
        c_emb = self.class_mlp(self.class_emb(class_label))
        t_emb = t_emb + c_emb

        x = self.init_conv(x)
        res_stack = [x]
        
        for down in self.downs:
            x, res = down(x, t_emb)
            res_stack.append(res)
            
        x = self.mid_gn1(self.mid_act(self.mid_block1(x)))
        x = x + self.mid_act(self.mid_time_mlp(t_emb))[..., None, None]
        x = self.mid_gn2(self.mid_act(self.mid_block2(x)))
        
        for up in self.ups:
            res = res_stack.pop()
            x = up(x, res, t_emb)
            
        return self.final_conv(x)
