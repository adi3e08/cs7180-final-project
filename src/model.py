import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from src.utils import construct_observation_tensor

class CNN1(nn.Module):
    def __init__(self, d_emb):
        super().__init__()
        self.conv1 = nn.Conv2d(4,  32,  kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64,  kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.pool  = nn.AdaptiveAvgPool2d((1, 1))
        self.proj  = nn.Linear(128, d_emb)

    def forward(self, x):
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        x = F.silu(self.conv3(x))
        x = self.pool(x).flatten(1)  # (B, 128)
        x = self.proj(x)
        return x  # (B, d_emb)

class ResNetBackbone(nn.Module):
    """Pretrained ResNet-18 backbone adapted for 4-channel (RGB + depth) input.

    The first conv layer is re-initialized to accept 4 channels: pretrained
    RGB weights are copied into channels 0-2, and channel 3 (depth) is
    initialized as the mean of the RGB weights so it starts in a reasonable range.
    All layers except the first conv and the final projection are frozen.
    """
    def __init__(self, d_emb):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Adapt first conv: 3ch → 4ch
        old_conv = backbone.conv1  # (64, 3, 7, 7)
        new_conv = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight          # copy RGB weights
            new_conv.weight[:, 3:] = old_conv.weight.mean(1, keepdim=True)  # depth init
        backbone.conv1 = new_conv

        # Remove the original classifier head
        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
            backbone.avgpool,
        )

        # Freeze everything except the new first conv layer
        for name, param in self.features.named_parameters():
            if not name.startswith("0."):  # "0." is conv1 in the Sequential
                param.requires_grad = False

        # Projection head (trainable)
        self.proj = nn.Linear(512, d_emb)

    def forward(self, x):
        x = self.features(x)   # (B, 512, 1, 1)
        x = x.flatten(1)       # (B, 512)
        x = self.proj(x)       # (B, d_emb)
        return x

class MLPVectorField2(nn.Module):
    def __init__(self, arglist):
        super().__init__()
        self.arglist = arglist 
        
        # Observation encoding
        self.proprio_encoder = nn.Linear(arglist.d_proprio, arglist.d_emb)
        if arglist.image:
            backbone = getattr(arglist, 'backbone', 'cnn')
            if backbone == 'resnet18':
                self.image_encoder = ResNetBackbone(arglist.d_emb)
            else:
                self.image_encoder = CNN1(arglist.d_emb)
        if arglist.text:
            self.text_encoder = nn.Linear(512, arglist.d_emb)

        # Time encoding
        self.time_encoder = nn.Linear(1, arglist.d_emb)

        # Action encoding
        self.action_encoder = nn.Linear(arglist.d_act, arglist.d_emb)
        
        input_dim = arglist.d_emb * (3+int(self.arglist.image)+int(self.arglist.text))

        # Core vector field
        self.mlp = nn.Sequential(nn.Linear(input_dim, arglist.d_model), nn.SiLU(),
                                 nn.Linear(arglist.d_model, arglist.d_model), nn.SiLU(),
                                 nn.Linear(arglist.d_model, arglist.d_act))

    def forward(self, O, A, tau):
        """
        O['proprio']: B, d_proprio
        O['image']: B, 3, 64, 64
        O['text']: B, 512 # later
        A: B, d_act
        tau: B, 1
        """
        
        # Observation encoding
        obs_emb = []
        
        proprio_emb = self.proprio_encoder(O['proprio'])
        obs_emb.append(proprio_emb)

        if self.arglist.image:
            rgbd = torch.cat((O['rgb'],O['depth']),1)
            image_emb = self.image_encoder(rgbd)
            obs_emb.append(image_emb)
        
        if self.arglist.text:
            text_emb = self.text_encoder(O['text'])
            obs_emb.append(text_emb)
        
        # Time encoding
        time_emb = self.time_encoder(tau)

        # Action encoding
        A_emb = self.action_encoder(A)

        v = self.mlp(torch.cat(obs_emb + [A_emb, time_emb], dim=-1))
        return v

class MLPVectorField1(nn.Module):
    def __init__(self, arglist):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(arglist.d_proprio + arglist.d_act + 1, arglist.d_model), nn.SiLU(),
                                 nn.Linear(arglist.d_model, arglist.d_model), nn.SiLU(),
                                 nn.Linear(arglist.d_model, arglist.d_act))

    def forward(self, O, A, tau):
        return self.mlp(torch.cat([O['proprio'], A, tau], dim=-1))

class FlowMatchingModel(nn.Module):
    def __init__(self, arglist):
        super().__init__()
        self.arglist = arglist
        if arglist.expt == "expt_1":
            self.vector_field = MLPVectorField1(arglist)
        elif arglist.expt in ("expt_2", "expt_2_resnet"):
            self.vector_field = MLPVectorField2(arglist)
        # expt_2_resnet shares data with expt_2
        data_expt = "expt_2" if arglist.expt == "expt_2_resnet" else arglist.expt
        data_dir = os.path.join("./data/raw", data_expt)
        self.stats = np.load(os.path.join(data_dir, "stats.npz"), allow_pickle=True)
    
    def loss(self, O, A):
        eps = torch.randn_like(A)
        tau = torch.rand_like(A[:,:1])
        A_noisy = tau * A + (1-tau) * eps
        return nn.functional.mse_loss(self.vector_field(O, A_noisy, tau), A - eps)

    def rk1(self, O, A, tau, h):
        k1 = self.vector_field(O, A, tau)
        return A + h * k1

    def rk2(self, O, A, tau, h): # Ralston's method
        k1 = self.vector_field(O, A, tau)
        k2 = self.vector_field(O, A + h * k1, tau + h)
        alpha = 2.0/3.0 
        return A + h * ((1.0 - 1.0/(2.0*alpha))*k1 + (1.0/(2.0*alpha))*k2)

    @torch.no_grad()
    def sample(self, o, env, device):
        O = construct_observation_tensor(o, env, self.arglist, self.stats, device)
        n_samples = 1
        h = 1 / self.arglist.T_flow
        tau = torch.zeros(n_samples, 1, device=device)
        A = torch.randn(n_samples, self.arglist.d_act, device=device)
        with torch.no_grad():
            for i in range(self.arglist.T_flow):
                A = self.rk2(O, A, tau, h)
                tau = tau + h
        a = A.cpu().numpy()[0]
        if self.arglist.normalize:
            a = a * self.stats['action_std'] + self.stats['action_mean']
        return a