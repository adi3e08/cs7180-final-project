import argparse
import torch
import sys
sys.path.append('/content/cs7180-final-project')
sys.path.append('/content/cs7180-final-project/utils')
sys.path.append('/content/cs7180-final-project/models')
sys.path.append('/content/cs7180-final-project/results/checkpoints/')
from utils.data_process import load_data
from models.policy import VectorFieldUNetCFG
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
    model = VectorFieldUNetCFG(img_ch=1, base_ch=64, t_dim=128, NUM_CLASSES=10, DROP_PROB=0.1).to(args.device)
    train_loader, val_loader = load_data(args.batch_size, args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_losses, val_losses = [], []

    for epoch in range(1, args.epochs + 1):

        # ── Training ──────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for x1, labels in tqdm(train_loader, leave=False, desc=f"Epoch {epoch}"):
            x1, labels = x1.to(args.device), labels.to(args.device)
            loss = model.flow_matching_loss_cfg(x1, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stability
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))
        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for x1, labels in val_loader:
                x1, labels = x1.to(args.device), labels.to(args.device)
                v_loss += model.flow_matching_loss_cfg(x1, labels).item()
        val_losses.append(v_loss / len(val_loader))

        print(f"Epoch {epoch:3d} | Train: {train_losses[-1]:.4f} | Val: {val_losses[-1]:.4f}")

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"{args.save_path}ckpt_ep{epoch}.pt")

    # Save final model
    torch.save(model.state_dict(), f"{args.save_path}model_final.pt")


if __name__ == "__main__":
    main()