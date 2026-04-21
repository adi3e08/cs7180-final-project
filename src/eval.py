import os
import time
import argparse
import numpy as np
import torch
import gymnasium as gym
import metaworld
from src.model import FlowMatchingModel
from src.utils import get_expert_policy, swap_obs, check_success
import imageio

def parse_args():
    parser = argparse.ArgumentParser("Flow matching")
    parser.add_argument("--env", type=str, default="bin-picking-three-objects-v3")
    parser.add_argument("--expt", type=str, default="expt_2", help="expt name")
    parser.add_argument("--seed", type=int, default=153)
    parser.add_argument("--ckpt", type=str, default="best.ckpt")
    parser.add_argument("--episodes", type=int, default=3)
    # Simulation parameters
    parser.add_argument("--d-proprio", type=int, default=4, help="proprio dimension, expt_1: 11, expt2: 4")
    parser.add_argument("--d-act", type=int, default=4, help="action dimension is 4 across meta-world tasks")
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--image", action="store_true", default=True, help="expt_1: False, expt_2: True")
    parser.add_argument("--camera-id", type=int, default=6, help="6: gripper pov")
    parser.add_argument("--image-height", type=int, default=240, help="image height")
    parser.add_argument("--image-width", type=int, default=240, help="image width")
    parser.add_argument("--text", action="store_true", default=True)
    parser.add_argument("--num_objects", type=int, default=3, help="number of objects in the environment")

    # Model parameters
    parser.add_argument("--T-flow", type=int, default=20, help="flow time steps for sampling")
    parser.add_argument("--d-model", type=int, default=128, help="hidden size dim, expt_1: 64, expt_2: 128")
    parser.add_argument("--d-emb", type=int, default=32, help="embedding dim (only for expt_2 currently)")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--use_backbone", action="store_true", default=True, help="use backbone for image encoding")
    parser.add_argument("--num_layers", type=int, default=4, help="number of layers in the model")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size for training")
    return parser.parse_args()

def eval_expert_policy(arglist):
    if arglist.display:
        render_mode = "human"
    elif arglist.image:
        render_mode = "rgb_array"
    else:
        render_mode = "none"

    env = gym.make('Meta-World/MT1', env_name=arglist.env, seed=arglist.seed, render_mode=render_mode) 
    policy = get_expert_policy(arglist)
    colors = ["green", "yellow", "purple"]
    metric = []
    for episode in range(arglist.episodes):
        o, info = env.reset()
        frames = []
        step=0
        if arglist.text:
            target = np.random.choice(arglist.num_objects)
        else:
            target = 0
        while True:
            if arglist.text:
                o = swap_obs(o, target, arglist)
            a = policy.get_action(o)
            o_1, r, terminated, truncated, info = env.step(a)
            if arglist.display:
                env.render()
                time.sleep(0.05)
            else:
              frame = env.render()  
              if frame is not None:
                  frames.append(frame)
            step+=1
            print(f"step {step}")
            o = o_1
            success = check_success(o_1, target, arglist)
            done = terminated or truncated or success
            if done:
                metric.append(success)
                break
        print(f"episode: {episode}, target:{colors[target]}, success: {bool(success)}")
        gif_path = f"/content/episode_{episode}.gif"
        imageio.mimsave(gif_path, frames, fps=20)
        print(f"Saved GIF: {gif_path}")
    print(f"Task success rate on {arglist.episodes} episodes: {np.mean(metric)}")
        
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
    print(f"Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    model = FlowMatchingModel(arglist).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    metric = []
    all_gifs = []
    for episode in range(arglist.episodes):
        o, info = env.reset()
        env_top.reset()
        frames = []
        step=0
        while True:
            a = model.sample(o, env, env_top, device)
            o_1, r, terminated, truncated, info = env.step(a)
            env_top.step(a)
            if arglist.display:
                env.render()
                time.sleep(0.05)
            else:
              frame = env.render()  
              if frame is not None:
                  frames.append(frame)
            step+=1
            print(f"step {step}")
            success = int(info['success'])
            done = terminated or truncated or success
            o = o_1
            if done:
                print(f"episode: {episode}, success: {bool(success)}")
                metric.append(success)
                break
        gif_path = f"/content/episode_{episode}.gif"
        imageio.mimsave(gif_path, frames, fps=20)
        print(f"Saved GIF: {gif_path}")
    print("Task success rate, ", np.mean(metric))
    env.close()

if __name__ == '__main__':
    arglist = parse_args()
    # eval_expert_policy(arglist)
    eval_model(arglist)