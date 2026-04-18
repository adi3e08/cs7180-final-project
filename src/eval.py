import os
import time
import argparse
import numpy as np
import torch
import gymnasium as gym
import metaworld
from src.model import FlowMatchingModel
from src.utils import get_expert_policy

def parse_args():
    parser = argparse.ArgumentParser("Flow matching")
    parser.add_argument("--env", type=str, default="bin-picking-v3")
    parser.add_argument("--expt", type=str, default="expt_2", help="expt name")
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--ckpt", type=str, default="best.ckpt")
    parser.add_argument("--episodes", type=int, default=1)
    # Simulation parameters
    parser.add_argument("--d-proprio", type=int, default=4, help="proprio dimension, expt_1: 11, expt2: 4")
    parser.add_argument("--d-act", type=int, default=4, help="action dimension is 4 across meta-world tasks")
    parser.add_argument("--display", action="store_true", default=True)
    parser.add_argument("--image", action="store_true", default=True, help="expt_1: False, expt_2: True")
    parser.add_argument("--camera-id", type=int, default=6, help="6: gripper pov")
    parser.add_argument("--image-height", type=int, default=240, help="image height")
    parser.add_argument("--image-width", type=int, default=240, help="image width")
    parser.add_argument("--text", action="store_true", default=False)
    # Model parameters
    parser.add_argument("--T-flow", type=int, default=20, help="flow time steps for sampling")
    parser.add_argument("--d-model", type=int, default=128, help="hidden size dim, expt_1: 64, expt_2: 128")
    parser.add_argument("--d-emb", type=int, default=32, help="embedding dim (only for expt_2 currently)")
    parser.add_argument("--normalize", action="store_true", default=True)
    return parser.parse_args()

def eval_expert_policy(arglist):
    if arglist.display:
        render_mode = "human"
    else:
        render_mode = "none"

    env = gym.make('Meta-World/MT1', env_name=arglist.env, seed=arglist.seed, render_mode=render_mode) 
    policy = get_expert_policy(arglist)

    for episode in range(arglist.episodes):
        o, info = env.reset()
        while True:
            a = policy.get_action(o)
            o_1, r, terminated, truncated, info = env.step(a)
            if arglist.display:
                env.render()
                time.sleep(0.05)
            o = o_1
            success = int(info['success'])
            done = terminated or truncated or success
            if done:
                break
        print(f"episode: {episode}, success: {bool(success)}")
        
    env.close()

def eval_model(arglist):
    np.random.seed(arglist.seed)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(arglist.seed)
    device = torch.device("cpu")

    if arglist.display:
        render_mode = "human"
    elif arglist.image:
        render_mode = "rgb_array"
    else:
        render_mode = "none"
    
    if arglist.image:
        env = gym.make('Meta-World/MT1', env_name=arglist.env, seed=arglist.seed, render_mode=render_mode,\
                        camera_id=arglist.camera_id,height=arglist.image_height,width=arglist.image_width)
        env_top = gym.make('Meta-World/MT1', env_name=arglist.env, seed=arglist.seed,
                        render_mode="rgb_array", camera_name="topview",height=arglist.image_height, 
                        width=arglist.image_width)
    else:
        env = gym.make('Meta-World/MT1', env_name=arglist.env, seed=arglist.seed, render_mode=render_mode)  

    model_dir = os.path.join("./models", arglist.expt)
    checkpoint_path = os.path.join(model_dir, arglist.ckpt)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = FlowMatchingModel(arglist).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    metric = []
    for episode in range(arglist.episodes):
        o, info = env.reset()
        env_top.reset()
        while True:
            a = model.sample(o, env, env_top, device)
            o_1, r, terminated, truncated, info = env.step(a)
            if arglist.display:
                env.render()
                time.sleep(0.05)
            success = int(info['success'])
            done = terminated or truncated or success
            o = o_1
            if done:
                print(f"episode: {episode}, success: {bool(success)}")
                metric.append(success)
                break
    print("Task success rate, ", np.mean(metric))
    env.close()

if __name__ == '__main__':
    arglist = parse_args()
    # eval_expert_policy(arglist)
    eval_model(arglist)