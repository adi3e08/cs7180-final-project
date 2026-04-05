import torch
import matplotlib.pyplot as plt
from models.policy import VectorFieldUNetCFG
import argparse

@torch.no_grad()
def sample_cfg(model, digit: int, n: int = 16, steps: int = N_STEPS, w: float = GUIDANCE_W, DEVICE: torch.device = torch.device("cuda"), CHANNELS: int = 1, IMG_SIZE: int = 64, NULL_CLASS: int = 10) -> torch.Tensor:
    model.eval()
    x = torch.randn(n, CHANNELS, IMG_SIZE, IMG_SIZE, device=DEVICE)   # x_0 ~ N(0,I)
    dt = 1.0 / steps
    c_cond  = torch.full((n,), digit,      dtype=torch.long, device=DEVICE)
    c_uncond = torch.full((n,), NULL_CLASS, dtype=torch.long, device=DEVICE)
    for i in range(steps):
        t_val = i * dt
        t     = torch.full((n,), t_val, device=DEVICE)

        v_cond   = model(x, t, c_cond)     # conditional velocity
        v_uncond = model(x, t, c_uncond)   # unconditional velocity

        # CFG interpolation
        v = v_uncond + w * (v_cond - v_uncond)

        x     = x + dt * v            # Euler step

    return x.clamp(-1, 1)             # clip to valid image range


def visualize_results(model, SAVE_PATH, GUIDANCE_W, DEVICE, NUM_CLASSES, NULL_CLASS, N_STEPS, CHANNELS, IMG_SIZE):
    fig, axes = plt.subplots(10, 8, figsize=(10, 12))

    for digit in range(10):
        imgs = sample_cfg(model, digit=digit, n=8, w=GUIDANCE_W, DEVICE=DEVICE, NUM_CLASSES=NUM_CLASSES, NULL_CLASS=NULL_CLASS, N_STEPS=N_STEPS, CHANNELS=CHANNELS, IMG_SIZE=IMG_SIZE)
        imgs = (imgs + 1) / 2  # [-1,1] → [0,1]

        for j, ax in enumerate(axes[digit]):
            ax.imshow(imgs[j].squeeze().cpu(), cmap="gray", vmin=0, vmax=1)

            # Set title on each subplot
            ax.set_title(f"{digit}", fontsize=8)

            if j == 0:
                ax.set_ylabel(str(digit), fontsize=12, rotation=0, labelpad=15)

            ax.axis("off")

    plt.suptitle(f"CFG Flow Matching  |  guidance w={GUIDANCE_W}", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{SAVE_PATH}digits_0_9.png", dpi=150)
    plt.show()
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str,
                        default="checkpoints/", help="Path to load the trained model and save results")
    parser.add_argument("--guidance-w", type=float, help="Classifier-free guidance weight")
    parser.add_argument("--device", type=str, default="cuda", help="'cuda' or 'cpu'")
    return parser.parse_args()
    
def main():
    args = parse_args()
    
    model = VectorFieldUNetCFG(img_ch=1, base_ch=64, t_dim=128, NUM_CLASSES=10, DROP_PROB=0.1).to(args.device)
    model.load_state_dict(torch.load(f"{args.save_path}model_final.pt", map_location=args.device))
    visualize_results(model, args.save_path, args.guidance_w, args.device, 10, 10, 100, 1, 64)