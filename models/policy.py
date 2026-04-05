import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders import SinusoidalTimeEmb, ResBlock


class VectorFieldUNetCFG(nn.Module):
    def __init__(self, img_ch: int = 1, base_ch: int = 64, t_dim: int = 128, NUM_CLASSES: int = 10, DROP_PROB: float = 0.1):
        super().__init__()
        self.DROP_PROB = DROP_PROB
        self.NULL_CLASS = NUM_CLASSES  # index for null token in class embedding
        tc = t_dim * 2   # projected time dim used throughout

        # ── Class embedding (NUM_CLASSES + 1 for null token) ──────────
        self.class_emb = nn.Embedding(NUM_CLASSES + 1, t_dim)

        # ── Time embedding MLP ────────────────────────────────────────
        self.t_emb = nn.Sequential(
            SinusoidalTimeEmb(t_dim),
            nn.Linear(t_dim, tc),
            nn.SiLU(),
            nn.Linear(tc, tc),
        )

        # ── Class projection: maps class emb to same dim as t_emb ─────
        self.c_proj = nn.Sequential(
            nn.Linear(t_dim, tc),
            nn.SiLU(),
        )

        # ── Encoder ───────────────────────────────────────────────────
        self.in_conv = nn.Conv2d(img_ch, base_ch, 3, padding=1)          # 28x28
        self.enc1    = ResBlock(base_ch,      base_ch * 2, tc)            # 28x28
        self.down1   = nn.Conv2d(base_ch * 2, base_ch * 2, 4, 2, 1)      # →14x14
        self.enc2    = ResBlock(base_ch * 2,  base_ch * 4, tc)            # 14x14
        self.down2   = nn.Conv2d(base_ch * 4, base_ch * 4, 4, 2, 1)      # →7x7

        # ── Bottleneck ────────────────────────────────────────────────
        self.mid1 = ResBlock(base_ch * 4, base_ch * 4, tc)
        self.mid2 = ResBlock(base_ch * 4, base_ch * 4, tc)

        # ── Decoder ───────────────────────────────────────────────────
        # After upsampling, we concat encoder skip → 2x channels
        self.up1     = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, 4, 2, 1)  # 7→14
        self.dec1    = ResBlock(base_ch * 8, base_ch * 2, tc)                  # concat
        self.up2     = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 4, 2, 1)  # 14→28
        self.dec2    = ResBlock(base_ch * 4, base_ch,     tc)                  # concat

        # ── Output head ───────────────────────────────────────────────
        self.out = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, img_ch, 3, padding=1),
        )

    def forward(self, x, t, c) -> torch.Tensor:
        """
        x : (B, C, H, W)  — noisy image at time t
        t : (B,)           — time in [0, 1]
        c : (B,)           — class indices 0–9, or NULL_CLASS (10) for uncond
        returns: (B, C, H, W) — predicted velocity field
        """
        te = self.t_emb(t)                   # (B, tc)
        ce  = self.c_proj(self.class_emb(c))
        # element-wise addition
        cond = te + ce   # additive fusion  ← key line
        x0  = self.in_conv(x)
        e1  = self.enc1(x0, cond)
        e2  = self.enc2(self.down1(e1), cond)
        m   = self.mid2(self.mid1(self.down2(e2), cond), cond)
        d1  = self.dec1(torch.cat([self.up1(m), e2], dim=1), cond)
        d2  = self.dec2(torch.cat([self.up2(d1), e1], dim=1), cond)
        return self.out(d2) 
    
    def flow_matching_loss_cfg(self, x1, labels) -> torch.Tensor:
        B = x1.size(0)

        # Step 1: Sample noise and time
        x0 = torch.randn_like(x1)                              # source: N(0, I)
        t  = torch.rand(B, device=x1.device)                   # t ~ U[0, 1]

        # Label dropout: replace with null token randomly
        drop_mask = torch.rand(B, device=x1.device) < self.DROP_PROB
        c = labels.clone()
        c[drop_mask] = self.NULL_CLASS              # ← the only new line vs base loss

        # Step 2: Interpolate along the straight-line path
        t_b = t.view(B, 1, 1, 1)
        xt  = t_b * x1 + (1.0 - t_b) * x0                     # conditional path

        # Step 3: Target conditional vector field (constant, direction to data) (that vekocity component, data Pdata - noise P0 time)
        target = x1 - x0

        # Step 4 & 5: Predict and compute loss
        pred = self(xt, t, c)
        return F.mse_loss(pred, target)# (