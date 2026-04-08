import gymnasium as gym
import metaworld
# from metaworld.policies.sawyer_reach_v3_policy import SawyerReachV3Policy as ExpertPolicy
from metaworld.policies.sawyer_pick_place_v3_policy import SawyerPickPlaceV3Policy as ExpertPolicy
# from metaworld.policies.sawyer_pick_place_wall_v3_policy import SawyerPickPlaceWallV3Policy as ExpertPolicy
# from metaworld.policies.sawyer_shelf_place_v3_policy import SawyerShelfPlaceV3Policy as ExpertPolicy
# from metaworld.policies.sawyer_bin_picking_v3_policy import SawyerBinPickingV3Policy as ExpertPolicy

import time
import os
import argparse
import numpy as np
import torch
from flow_matching import normalize

def parse_args():
    parser = argparse.ArgumentParser("Flow matching")
    # Settings
    parser.add_argument("--env", type=str, default="pick-place-v3", help="")
    parser.add_argument("--image", action="store_true", default=False)
    parser.add_argument("--text", action="store_true", default=False)
    parser.add_argument("--expt", type=str, default="expt_proprio_128_3", help="expt name")
    parser.add_argument("--train-seed",   type=int,            default=0)
    parser.add_argument("--eval-seed",    type=int,            default=49)
    parser.add_argument("--ckpt",         type=int,            default=249)
    parser.add_argument("--episodes",     type=int,            default=2)
    parser.add_argument("--display",      action="store_true", default=True)
    parser.add_argument("--T-flow", type=int, default=20, help="flow time steps for sampling")
    parser.add_argument("--d-model", type=int, default=128, help="hidden size dim")
    parser.add_argument("--d-emb", type=int, default=32, help="hidden size dim")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=250, help="number of epochs to train")
    return parser.parse_args()

def eval_expert_policy(arglist):
    env_name = "pick-place-v3" # reach-v3, pick-place-v3, shelf-place-v3, pick-place-wall-v3, bin-picking-v3, bin-picking-v3-two-objects
    render_mode = "human" # human, rgb_array, depth_array, or rgbd_tuple
    
    seed = 0
    env = gym.make('Meta-World/MT1', env_name=env_name, seed=seed, render_mode=render_mode) #
    policy = ExpertPolicy()

    for episode in range(1):
        t = 0
        ep_r = 0
        o, info = env.reset()
        while True:
            a = policy.get_action(o)
            # a = env.action_space.sample()
            o_1, r, terminated, truncated, info = env.step(a)
            env.render()
            time.sleep(0.15)
            t += 1
            # print(f"\nt: {t}"
            #       f"\nobservation: {o.shape}, action: {a.shape}, reward: {r}, "
            #       f"new observation: {o_1.shape}, terminated: {terminated}, truncated: {truncated}"
            #       f"\ninfo: {info}")
            ep_r += r
            o = o_1
            success = int(info['success'])
            # done = success
            done = terminated or truncated or success
            if done:
                break

        print(episode, t, terminated, truncated, success)
        
    env.close()

def eval_agent(arglist):
    np.random.seed(arglist.eval_seed)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(arglist.eval_seed)
    device = torch.device("cpu")

    env_name = "pick-place-v3" # reach-v3, pick-place-v3, shelf-place-v3, pick-place-wall-v3, bin-picking-v3, bin-picking-v3-two-objects
    render_mode = "human" # human, rgb_array, depth_array, or rgbd_tuple
    
    env = gym.make('Meta-World/MT1', env_name=env_name, seed=arglist.eval_seed, render_mode=render_mode, camera_id=1) #

    checkpoint_path = os.path.join("./log", arglist.env, "flow_matching", arglist.expt,
                                   "seed_" + str(arglist.train_seed),
                                   "models", str(arglist.ckpt)+".ckpt") # "best.ckpt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    from flow_matching import FlowModel
    arglist.d_proprio = 39
    arglist.d_act = 4
    model = FlowModel(arglist).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    dataset = np.load(os.path.join("./log", arglist.env,
                                   "flow_matching", "train.npz"), allow_pickle=True)
    if arglist.normalize:
        Q_mu = dataset['proprio_mean']
        Q_sigma = dataset['proprio_std']
        A_mu = dataset['action_mean']
        A_sigma = dataset['action_std']

    metric = []
    for episode in range(3):
        o, info = env.reset()
        while True:
            if arglist.normalize:
                O = {'proprio': torch.tensor(normalize(o, Q_mu, Q_sigma),
                     dtype=torch.float32, device=device).unsqueeze(0)}
            else:
                O = {'proprio': torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)}

            if arglist.image:
                O['image'] = torch.tensor(o['image']/np.float32(255.0), dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                A = model.sample(O)
            a = A.cpu().numpy()[0]
            if arglist.normalize:
                a = a * A_sigma + A_mu
            o_1, r, terminated, truncated, info = env.step(a)
            env.render()
            time.sleep(0.15)
            success = int(info['success'])
            done = terminated or truncated or success
            o = o_1
            if done:
                print(success)
                metric.append(success)
                break
    print("Task success rate, ", np.mean(metric))
    env.close()

if __name__ == '__main__':
    arglist = parse_args()
    eval_expert_policy(arglist)
    # eval_agent(arglist)