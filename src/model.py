import os
from xml.parsers.expat import model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import construct_observation_tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
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
    
    
class FasterRCNNBackbone(nn.Module):
    def __init__(self, n_classes=5, d_emb=128):
        super().__init__()
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        fasterrcnn = fasterrcnn_resnet50_fpn(weights=weights)
        
        # Replace detection head for your classes
        in_features = fasterrcnn.roi_heads.box_predictor.cls_score.in_features
        fasterrcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)
        
        # Freeze backbone, unfreeze detection head
        for param in fasterrcnn.backbone.parameters():
            param.requires_grad = False
        for param in fasterrcnn.backbone.body.layer4.parameters():
            param.requires_grad = True
        for param in fasterrcnn.backbone.fpn.parameters():
            param.requires_grad = True
        for param in fasterrcnn.roi_heads.box_predictor.parameters():
            param.requires_grad = True
        
        # Store components separately
        self.transform = fasterrcnn.transform
        self.backbone = fasterrcnn.backbone  # ResNet50 + FPN
        self.rpn = fasterrcnn.rpn
        self.roi_heads = fasterrcnn.roi_heads
        
        # Branch 2: VLA feature projection
        self.vla_projection = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.1),
            nn.Linear(128, d_emb)
        )
    
    def forward(self, images, targets):
        # Transform images
        images, targets = self.transform(images, targets)
        
        # Shared FPN features
        features = self.backbone(images.tensors)
        
        # Branch 2: VLA features from FPN level
        vla_features = self.vla_projection(features['0'])  # pick an FPN level
        if targets is not None:
            proposals, rpn_losses = self.rpn(images, features, targets)
            detections, det_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
            return vla_features, detections, rpn_losses, det_losses
        else:
            proposals, _ = self.rpn(images, features)
            detections, _ = self.roi_heads(features, proposals, images.image_sizes)
            return vla_features, detections, None, None

class MLPVectorField2(nn.Module):
    def __init__(self, arglist):
        super().__init__()
        self.arglist = arglist 
        
        # Observation encoding
        self.proprio_encoder = nn.Linear(arglist.d_proprio, arglist.d_emb)
        input_dim = arglist.d_emb * (3+int(self.arglist.image)+int(self.arglist.text))
        if arglist.image:
            self.image_encoder = CNN1(arglist.d_emb)
            if arglist.use_backbone:
                self.detection_backbone = FasterRCNNBackbone(n_classes=6, d_emb=arglist.d_emb)   
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.detection_backbone = self.detection_backbone.to(device)
                input_dim = arglist.d_emb * (3 + int(self.arglist.image) + int(self.arglist.use_backbone) + int(self.arglist.text))
            
        if arglist.text:
            self.text_encoder = nn.Linear(512, arglist.d_emb)

        # Time encoding
        self.time_encoder = nn.Linear(1, arglist.d_emb)

        # Action encoding
        self.action_encoder = nn.Linear(arglist.d_act, arglist.d_emb)

        # Core vector field

        # Core vector field
        layers = [nn.Linear(input_dim, arglist.d_model), nn.SiLU()]
        for _ in range(arglist.num_layers - 2):
            layers += [nn.Linear(arglist.d_model, arglist.d_model), nn.SiLU()]
        layers.append(nn.Linear(arglist.d_model, arglist.d_act))
        self.mlp = nn.Sequential(*layers)

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
        detections, rpn_losses, det_losses = None, None, None
        proprio_emb = self.proprio_encoder(O['proprio'])
        obs_emb.append(proprio_emb)

        if self.arglist.image:
            rgbd = torch.cat((O['rgb'],O['depth']),1)
            image_emb = self.image_encoder(rgbd)
            obs_emb.append(image_emb)
            if self.arglist.use_backbone and 'target' in O and 'topdown' in O:
                vla_features, detections, rpn_losses, det_losses = self.detection_backbone(O['topdown'], O['target'])
                obs_emb.append(vla_features)
        
        if self.arglist.text:
            text_emb = self.text_encoder(O['text'])
            obs_emb.append(text_emb)
        
        # Time encoding
        time_emb = self.time_encoder(tau)

        # Action encoding
        A_emb = self.action_encoder(A)

        v = self.mlp(torch.cat(obs_emb + [A_emb, time_emb], dim=-1))
        return v, (detections, rpn_losses, det_losses)

class MLPVectorField1(nn.Module):
    def __init__(self, arglist):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(arglist.d_proprio + arglist.d_act + 1, arglist.d_model), nn.SiLU(),
                                 nn.Linear(arglist.d_model, arglist.d_model), nn.SiLU(),
                                 nn.Linear(arglist.d_model, arglist.d_act))

    def forward(self, O, A, tau):
        return self.mlp(torch.cat([O['proprio'], A, tau], dim=-1)), None

class FlowMatchingModel(nn.Module):
    def __init__(self, arglist):
        super().__init__()
        self.arglist = arglist
        if arglist.expt == "expt_1":
            self.vector_field = MLPVectorField1(arglist)
        elif arglist.expt == "expt_2":
            self.vector_field = MLPVectorField2(arglist)
        data_dir = os.path.join("/content/drive/MyDrive/APLDL/data/raw/", arglist.expt)
        self.stats = np.load(os.path.join(data_dir, "stats.npz"), allow_pickle=True)
        self.detections = None
    
    def loss(self, O, A):
        eps = torch.randn_like(A)
        tau = torch.rand_like(A[:,:1])
        A_noisy = tau * A + (1-tau) * eps
        action_preds, detection_params = self.vector_field(O, A_noisy, tau)
        action_loss = nn.functional.mse_loss(action_preds, A - eps)
        detection_loss = torch.tensor(0.0, device=action_preds.device)
        rpn_loss, det_loss = None, None
        if self.arglist.use_backbone and detection_params is not None:
          self.detections, rpn_losses, det_losses = detection_params
          rpn_loss = 0.0
          det_loss = 0.0
          if isinstance(rpn_losses, dict) and len(rpn_losses) > 0:
              rpn_loss = sum(v for v in rpn_losses.values() if v is not None)

          if isinstance(det_losses, dict) and len(det_losses) > 0:
              det_loss = sum(v for v in det_losses.values() if v is not None)
        
          detection_loss = detection_loss + (
              rpn_loss if torch.is_tensor(rpn_loss) else torch.tensor(rpn_loss, device=action_preds.device)
          ) + (
              det_loss if torch.is_tensor(det_loss) else torch.tensor(det_loss, device=action_preds.device)
          )
        return action_loss, detection_loss

    def rk1(self, O, A, tau, h):
        k1, _ = self.vector_field(O, A, tau)
        return A + h * k1

    def rk2(self, O, A, tau, h): # Ralston's method
        k1, _ = self.vector_field(O, A, tau)
        k2, _ = self.vector_field(O, A + h * k1, tau + h)
        alpha = 2.0/3.0 
        return A + h * ((1.0 - 1.0/(2.0*alpha))*k1 + (1.0/(2.0*alpha))*k2)

    @torch.no_grad()
    def sample(self, o, env, env_top, device, target=None):
        O = construct_observation_tensor(o, env, env_top, self.arglist, self.stats, device, target)
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