import torch
import torch.nn as nn
from .encoders import SinusoidalTimeEmb

class VLAFlowMatching(nn.Module):
    def __init__(self, img_encoder, txt_encoder, state_encoder, d_model=128, action_dim=4):
        super().__init__()
        self.img_enc   = img_encoder
        self.txt_enc   = txt_encoder
        self.state_enc = state_encoder
        self.t_emb     = nn.Sequential(SinusoidalTimeEmb(d_model), nn.Linear(d_model, d_model))

        # denoiser MLP
        self.denoiser = nn.Sequential(
            nn.Linear(action_dim + d_model, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, xt, t, img, tokens, state):
        cond = (self.img_enc(img) + 
                self.txt_enc(tokens) + 
                self.state_enc(state) + 
                self.t_emb(t))             
        return self.denoiser(torch.cat([xt, cond], dim=-1))