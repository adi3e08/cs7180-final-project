import os
import argparse
import numpy as np
import gymnasium as gym
import metaworld
import time
import torch
from utils import get_expert_policy, get_images, log_image

def parse_args():
    parser = argparse.ArgumentParser("Flow matching")
    # Settings
    parser.add_argument("--env", type=str, default="bin-picking-v3", help="reach-v3, pick-place-v3, shelf-place-v3, \
                                                                          pick-place-wall-v3, bin-picking-v3, \
                                                                          bin-picking-v3-two-objects")
    parser.add_argument("--display", action="store_true", default=True)
    parser.add_argument("--image", action="store_true", default=True)
    parser.add_argument("--camera-id", type=int, default=6, help="6: gripper pov")
    parser.add_argument("--image-height", type=int, default=240, help="image height")
    parser.add_argument("--image-width", type=int, default=240, help="image width")
    parser.add_argument("--text", action="store_true", default=False)
    
    parser.add_argument("--expt", type=str, default="expt_proprio_image_128_3_emb_32", help="expt name")
    parser.add_argument("--train-seed",   type=int,            default=0)
    parser.add_argument("--eval-seed",    type=int,            default=60)
    parser.add_argument("--ckpt",         type=int,            default=20)
    parser.add_argument("--episodes",     type=int,            default=1)

    parser.add_argument("--T-flow", type=int, default=20, help="flow time steps for sampling")
    parser.add_argument("--d-model", type=int, default=128, help="hidden size dim")
    parser.add_argument("--d-emb", type=int, default=32, help="hidden size dim")
    parser.add_argument("--normalize", action="store_true", default=True)
    return parser.parse_args()

def eval_expert_policy(arglist):
    if arglist.display:
        render_mode = "human"
    elif arglist.image:
        render_mode = "rgb_array"
    else:
        render_mode = "none"

    if arglist.image:
        env = gym.make('Meta-World/MT1', env_name=arglist.env, seed=arglist.eval_seed, render_mode=render_mode,\
                       camera_id=arglist.camera_id ,height=arglist.image_height,width=arglist.image_width)
    else:
        env = gym.make('Meta-World/MT1', env_name=arglist.env, seed=arglist.eval_seed, render_mode=render_mode) 
    
    policy = get_expert_policy(arglist)

    for episode in range(arglist.episodes):
        t = 0
        o, info = env.reset()
        while True:
            a = policy.get_action(o)
            o_1, r, terminated, truncated, info = env.step(a)
            if arglist.image:
                rgb_array, depth_array = get_images(env)
                if t % 10 == 0:
                    log_image(t, rgb_array, depth_array, os.path.join("./images",str(arglist.camera_id)))
            if arglist.display:
                env.render()
                time.sleep(0.05)
            t += 1
            o = o_1
            success = int(info['success'])
            done = terminated or truncated or success
            if done:
                break

        print(f"episode: {episode}, success: {bool(success)}")
        
    env.close()

def eval_agent(arglist):
    np.random.seed(arglist.eval_seed)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(arglist.eval_seed)
    device = torch.device("cpu")

    if arglist.display:
        render_mode = "human"
    elif arglist.image:
        render_mode = "rgb_array"
    else:
        render_mode = "none"
    
    if arglist.image:
        env = gym.make('Meta-World/MT1', env_name=arglist.env, seed=arglist.eval_seed, render_mode=render_mode,\
                        camera_id=arglist.camera_id,height=arglist.image_height,width=arglist.image_width)
    else:
        env = gym.make('Meta-World/MT1', env_name=arglist.env, seed=arglist.eval_seed, render_mode=render_mode)  

    checkpoint_path = os.path.join("./log", arglist.env, "flow_matching", arglist.expt,
                                   "seed_" + str(arglist.train_seed),
                                   "models", "best.ckpt") # str(arglist.ckpt)+".ckpt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    from flow_matching import FlowModel, normalize, get_tensor
    if arglist.image:
        arglist.d_proprio = 4
    else:
        arglist.d_proprio = 14
    arglist.d_act = 4
    model = FlowModel(arglist).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    if arglist.normalize:
        dataset = np.load(os.path.join("./log", arglist.env, "flow_matching", "train.npz"), allow_pickle=True)
        proprio_mean = dataset['proprio_mean']
        proprio_std = dataset['proprio_std']
        A_mean = dataset['action_mean']
        A_std = dataset['action_std']
        if arglist.image:
            rgb_mean = dataset['rgb_mean']
            rgb_std = dataset['rgb_std']
            depth_mean = dataset['depth_mean']
            depth_std = dataset['depth_std']

    metric = []
    for episode in range(arglist.episodes):
        o, info = env.reset()
        obj_start = o[4:7]
        goal = o[-3:]
        while True:
            if arglist.image:
                proprio = o[:4]
            else:
                proprio = np.concatenate((o[:11],o[-3:]))
            if arglist.normalize:
                O = {'proprio': get_tensor(normalize(proprio, proprio_mean, proprio_std)).unsqueeze(0).to(device)}
            else:
                O = {'proprio': get_tensor(proprio).unsqueeze(0).to(device)}

            if arglist.image:
                rgb_array, depth_array = get_images(env)
                if arglist.normalize:
                    O['rgb'] = get_tensor(normalize(rgb_array.astype(np.float32), rgb_mean, rgb_std)).unsqueeze(0).to(device)
                    O['depth'] = get_tensor(normalize(depth_array, depth_mean, depth_std)).unsqueeze(0).to(device)
                else:
                    O['rgb'] = get_tensor(rgb_array).unsqueeze(0).to(device)
                    O['depth'] = get_tensor(depth_array).unsqueeze(0).to(device)

            with torch.no_grad():
                A = model.sample(O)
            a = A.cpu().numpy()[0]
            if arglist.normalize:
                a = a * A_std + A_mean
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
    eval_agent(arglist)