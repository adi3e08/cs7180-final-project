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
    
class VLAProjection(nn.Module):
    def __init__(self, d_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), # Gives you a 4x4 grid
            nn.Flatten(start_dim=2),      # [B, 128, 16]
        )
        self.proj = nn.Linear(128, d_emb) # Maps 128 features to VLA size
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.net(x)          # [B, 128, 16]
        x = x.transpose(1, 2)    # [B, 16, 128] 
        x = self.proj(x)         # [B, 16, d_emb]
        x = self.dropout(x)
        return  x.reshape(x.shape[0], -1)
    
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
        # for param in fasterrcnn.backbone.body.layer4.parameters():
        #     param.requires_grad = True
        # for param in fasterrcnn.backbone.fpn.parameters():
        #     param.requires_grad = True
        for param in fasterrcnn.roi_heads.box_predictor.parameters():
            param.requires_grad = True
        
        # Store components separately
        self.transform = fasterrcnn.transform
        self.backbone = fasterrcnn.backbone  # ResNet50 + FPN
        self.rpn = fasterrcnn.rpn
        self.roi_heads = fasterrcnn.roi_heads
        
        # Branch 2: VLA feature projection
        self.vla_proj_0 = VLAProjection(d_emb=d_emb)
         # Outputs [B, d_emb] directly
        self.vla_proj_4_global = nn.Sequential(
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
        vla_features1 = self.vla_proj_0(features['0'])
        vla_features2 = self.vla_proj_4_global(features['3'])
        
        if targets is not None:
            proposals, rpn_losses = self.rpn(images, features, targets)
            detections, det_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
            return vla_features1, vla_features2, detections, rpn_losses, det_losses
        else:
            proposals, _ = self.rpn(images, features)
            detections, _ = self.roi_heads(features, proposals, images.image_sizes)
            return vla_features1, vla_features2, detections, None, None

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
                input_dim = arglist.d_emb * (3 + int(self.arglist.image) + (17 * int(self.arglist.use_backbone)) + int(self.arglist.text))
            
        if arglist.text:
            self.text_encoder = nn.Embedding(arglist.num_objects, arglist.d_emb)

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
                vla_features1, vla_features2, detections, rpn_losses, det_losses = self.detection_backbone(O['topdown'], O['target'])
                obs_emb.append(vla_features1)
                obs_emb.append(vla_features2)
        
        if self.arglist.text:
            text_emb = self.text_encoder(O['text']).squeeze(1)
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
        elif arglist.expt == "expt_4":
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

class CroCoAutoencoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=256, num_heads=4, enc_depth=6, dec_depth=6):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # ==========================================
        # 1. PATCH EMBEDDING
        # ==========================================
        # Shared projection to convert 16x16 image patches into 1D tokens
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embeddings to retain spatial coordinates
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # The learnable MASK token used to replace hidden patches in the Top-Down view
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # ==========================================
        # 2. SHARED ENCODER (ViT)
        # ==========================================
        # Extracts visual features from whatever patches it is given
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_depth)
        
        # ==========================================
        # 3. CROCO DECODER (Self-Attn + Cross-Attn)
        # ==========================================
        # In PyTorch, a TransformerDecoderLayer does exactly what CroCo specifies:
        # Top-Down Self-Attention -> Top-Down-to-Gripper Cross-Attention -> MLP
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_depth)
        
        # ==========================================
        # 4. RECONSTRUCTION HEAD
        # ==========================================
        # Projects the tokens back into raw RGB pixel values for the patch
        self.reconstruction_head = nn.Linear(embed_dim, 3 * patch_size * patch_size)

    def forward(self, topdown_img, gripper_img, mask_ratio=0.75, return_embedding_only=False):
        B = topdown_img.size(0)
        
        # --- PATCHIFICATION ---
        # [B, 3, 256, 256] -> [B, embed_dim, 16, 16] -> [B, 256, embed_dim]
        top_tokens = self.patch_embed(topdown_img).flatten(2).transpose(1, 2)
        grip_tokens = self.patch_embed(gripper_img).flatten(2).transpose(1, 2)
        
        # Add spatial positional embeddings
        top_tokens = top_tokens + self.pos_embed
        grip_tokens = grip_tokens + self.pos_embed
        
        # --- CROCO MASKING STRATEGY (Training Phase) ---
        # Randomly mask out e.g. 75% of the Top-Down tokens
        num_masked = int(mask_ratio * self.num_patches)
        noise = torch.rand(B, self.num_patches, device=top_tokens.device)
        mask_indices = torch.argsort(noise, dim=1)[:, :num_masked]
        visible_indices = torch.argsort(noise, dim=1)[:, num_masked:]
        
        # Gather only the visible Top-Down tokens for the encoder
        batch_indices = torch.arange(B).unsqueeze(1).expand(-1, self.num_patches - num_masked)
        visible_top_tokens = top_tokens[batch_indices, visible_indices]
        
        # --- ENCODING ---
        # The encoder only processes visible Top-Down patches and the FULL Gripper view
        enc_top = self.encoder(visible_top_tokens)
        enc_grip = self.encoder(grip_tokens)
        
        # --- DECODER PREP ---
        # Reconstruct the full Top-Down token sequence using the MASK tokens
        full_top_tokens = torch.zeros(B, self.num_patches, self.mask_token.size(-1), device=top_tokens.device)
        full_top_tokens[batch_indices, visible_indices] = enc_top
        
        mask_batch_indices = torch.arange(B).unsqueeze(1).expand(-1, num_masked)
        full_top_tokens[mask_batch_indices, mask_indices] = self.mask_token
        
        # Add positional embeddings again for the decoder so the mask tokens know where they are
        full_top_tokens = full_top_tokens + self.pos_embed
        
        # --- CROSS-VIEW COMPLETION ---
        # target = full_top_tokens (Queries)
        # memory = enc_grip (Keys/Values)
        fused_embeddings = self.decoder(tgt=full_top_tokens, memory=enc_grip)
        
        # If we are in the downstream phase, output the bottleneck embedding here!
        if return_embedding_only:
            return fused_embeddings.mean(dim=1) # Pooling into a 1D vector for Flow Matching
            
        # --- RECONSTRUCTION ---
        reconstructed_patches = self.reconstruction_head(fused_embeddings)
        
        return reconstructed_patches, mask_indices
    
def patchify(imgs, patch_size=16):
    """
    Converts a batch of images [B, 3, H, W] into a sequence of flattened patches.
    Output shape: [B, num_patches, 3 * patch_size * patch_size]
    """
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
    
    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x) # Rearrange dimensions
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3)) # Flatten the patch
    return x

def compute_croco_loss(model, topdown_img, gripper_img, mask_ratio=0.75):
    """
    Executes the forward pass and computes the MSE loss only on the masked patches.
    """
    B = topdown_img.size(0)
    
    # 1. Forward Pass
    # reconstructed_patches shape: [B, num_patches, 3 * 16 * 16]
    # mask_indices shape: [B, num_masked]
    reconstructed_patches, mask_indices = model(topdown_img, gripper_img, mask_ratio=mask_ratio)
    
    # 2. Prepare the Ground Truth
    # Convert the original top-down image into patches to match the output format
    target_patches = patchify(topdown_img, patch_size=model.patch_size)
    
    # 3. Extract ONLY the masked patches
    # We create a batch index tensor to advanced-index into the sequences
    batch_indices = torch.arange(B, device=topdown_img.device).unsqueeze(1).expand(-1, mask_indices.size(1))
    
    # Gather the predictions and targets for just the hidden patches
    pred_masked = reconstructed_patches[batch_indices, mask_indices]
    target_masked = target_patches[batch_indices, mask_indices]
    
    # 4. Compute MSE Loss
    loss = F.mse_loss(pred_masked, target_masked)
    
    return loss