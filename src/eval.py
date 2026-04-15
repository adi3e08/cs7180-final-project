import warnings
warnings.filterwarnings("ignore")
import os
import time
import argparse
import numpy as np
import torch
import gymnasium as gym
import metaworld
from src.model import FlowMatchingModel
from src.utils import get_expert_policy, swap_obs, check_success, add_expt_config

def parse_args():
    parser = argparse.ArgumentParser("Flow matching")
    parser.add_argument("--expt", type=str, default="expt_4", help="expt name")
    parser.add_argument("--seed", type=int, default=153)
    parser.add_argument("--ckpt", type=str, default="best.ckpt")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--display", action=argparse.BooleanOptionalAction)
    arglist = parser.parse_args()
    arglist = add_expt_config(arglist)
    return arglist

def eval_expert_policy():
    arglist = parse_args()
    
    if arglist.display:
        render_mode = "human"
    else:
        render_mode = "none"

    env = gym.make('Meta-World/MT1', env_name=arglist.env, seed=arglist.seed, render_mode=render_mode) 
    policy = get_expert_policy(arglist)
    colors = ["green", "yellow", "purple"]

    metric = []
    for episode in range(arglist.episodes):
        o, info = env.reset()
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
            o = o_1
            success = check_success(o_1, target, arglist)
            done = terminated or truncated or success
            if done:
                metric.append(success)
                break
        print(f"episode: {episode}, target:{colors[target]}, success: {bool(success)}")
    
    print(f"Task success rate on {arglist.episodes} episodes: {np.mean(metric)}")
    env.close()

def eval_model():
    arglist = parse_args()

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
    else:
        env = gym.make('Meta-World/MT1', env_name=arglist.env, seed=arglist.seed, render_mode=render_mode)  
    colors = ["green", "yellow", "purple"]

    model_dir = os.path.join("./models", arglist.expt)
    checkpoint_path = os.path.join(model_dir, arglist.ckpt)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = FlowMatchingModel(arglist).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    metric = []
    for episode in range(arglist.episodes):
        o, info = env.reset()
        if arglist.text:
            target = np.random.choice(arglist.num_objects)
        else:
            target = 0
        while True:
            a = model.sample(o, env, device, target)
            o_1, r, terminated, truncated, info = env.step(a)
            if arglist.display:
                env.render()
                time.sleep(0.05)
            success = check_success(o_1, target, arglist)
            done = terminated or truncated or success
            o = o_1
            if done:
                print(f"episode: {episode}, target:{target}, success: {bool(success)}")
                metric.append(success)
                break
    print(f"Task success rate on {arglist.episodes} episodes: {np.mean(metric)}")
    env.close()

if __name__ == '__main__':
    # eval_expert_policy()
    eval_model()