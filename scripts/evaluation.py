import torch
import matplotlib.pyplot as plt
from models.policy import VLAFlowMatching
from models.encoders import ImageEncoderTinyCNN, TextEncoderTinyGRU, StateEncoderMLP
import argparse
from utils.data_process import tokenize, resize_frame,VOCAB_SIZE
import gymnasium as gym
import imageio

@torch.no_grad()
def sample_action(model, img, tokens, state, action_dim=4, steps=100, device="cuda"):
    model.eval()
    x  = torch.randn(1, action_dim, device=device)   # x0 ~ N(0, I) in action space
    dt = 1.0 / steps

    for i in range(steps):
        t = torch.full((1,), i * dt, device=device)
        v = model(x, t, img, tokens, state)           # predicted velocity in action space
        x = x + dt * v                                # Euler step

    return x.squeeze(0)   # (4,) — the generated action


@torch.no_grad()
def visualize_results(model, env_name, camera="topview",
                      filename="rollout.gif", max_steps=200, seed=42, device="cuda"):
    env = gym.make("Meta-World/MT1", env_name=env_name, seed=seed,
                   render_mode="rgb_array", camera_name=camera)
    obs, info = env.reset()
    model.eval()
    frames  = []
    success = False
    tokens  = tokenize("pick the red ball").unsqueeze(0).to(device)  # fixed, never changes

    for step in range(max_steps):
        frame = env.render()
        frames.append(frame)

        img    = resize_frame(frame)   
        img = torch.FloatTensor(img).permute(2, 0, 1) / 255.0
        img = img.unsqueeze(0).to(device) 
        state  = torch.FloatTensor(obs[:4]).unsqueeze(0).to(device)      # (1, 4)
        action = sample_action(model, img, tokens, state,
                               action_dim=4, steps=100, device=device)   # (4,)

        obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())

        if int(info["success"]) == 1:
            success = True
            for _ in range(20):                    # a few extra frames after success
                frames.append(env.render())
                obs, _, _, _, _ = env.step(action.cpu().numpy())
            break

        if terminated or truncated:
            break

    env.close()
    imageio.mimsave(filename, frames, fps=30, loop=0)
    print(f"{'Success' if success else 'Failed'} | {len(frames)} frames saved to {filename}")
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str,
                        default="checkpoints/", help="Path to load the trained model and save results")
    parser.add_argument("--model_path", type=str, help="Path to the trained model")
    parser.add_argument("--device", type=str, default="cuda", help="'cuda' or 'cpu'")
    return parser.parse_args()
    
def main():
    args = parse_args()
    model = VLAFlowMatching(
        img_encoder   = ImageEncoderTinyCNN(d_model=128),
        txt_encoder   = TextEncoderTinyGRU(vocab_size=VOCAB_SIZE, d_model=128),
        state_encoder = StateEncoderMLP(state_dim=4, d_model=128),
    )
    model.load_state_dict(torch.load(f"{args.model_path}", map_location=args.device))
    visualize_results(model.to(args.device), "reach-v3", camera="topview", device=args.device)
    
    
if __name__ == "__main__":
    main()