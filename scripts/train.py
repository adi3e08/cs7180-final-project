import argparse
import torch
import sys
from metaworld.policies.sawyer_reach_v3_policy import SawyerReachV3Policy
sys.path.append('/content/cs7180-final-project')
sys.path.append('/content/cs7180-final-project/utils')
sys.path.append('/content/cs7180-final-project/models/')
sys.path.append('/content/cs7180-final-project/results/checkpoints/')
from utils.data_process import load_data, collect_expert_demos
from models.policy import VLAFlowMatching
from models.encoders import ImageEncoderTinyCNN, TextEncoderTinyGRU, StateEncoderMLP
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str,
                        default="data/dataset.npz")
    parser.add_argument("--resize-to", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--diffusion-T", type=int, default=16)
    parser.add_argument("--save-path", type=str,
                      default="checkpoints/")
    parser.add_argument("--device", type=str, default="cuda",
                        help="'cuda' or 'cpu'")
    return parser.parse_args()

def main():
    args = parse_args()
    model = VLAFlowMatching(
        img_encoder   = ImageEncoderTinyCNN(d_model=128),
        txt_encoder   = TextEncoderTinyGRU(vocab_size=128, d_model=128),
        state_encoder = StateEncoderMLP(state_dim=4, d_model=128),
    ).to(args.device)

    train_loader, val_loader = load_data(args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_losses, val_losses = [], []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, leave=False, desc=f"Epoch {epoch}"):
            x1    = batch["action"].to(args.device)
            img   = batch["top_view"].to(args.device)
            text  = batch["text"].to(args.device)
            state = batch["proprioception"].to(args.device)

            loss = model.flow_matching_loss_vla(x1, img, text, state)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x1    = batch["action"].to(args.device)
                img   = batch["top_view"].to(args.device)
                text  = batch["text"].to(args.device)
                state = batch["proprioception"].to(args.device)
                val_loss += model.flow_matching_loss_vla(x1, img, text, state).item()

        val_losses.append(val_loss / len(val_loader))
        print(f"Epoch {epoch:3d} | Train: {train_losses[-1]:.4f} | Val: {val_losses[-1]:.4f}")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_path, f"ckpt_ep{epoch}.pt"))

    torch.save(model.state_dict(), os.path.join(args.save_path, "model_final.pt"))
    
def emain():
    args = parse_args()
    collect_expert_demos("reach-v3", SawyerReachV3Policy, num_episodes=10)

if __name__ == "__main__":
    emain()