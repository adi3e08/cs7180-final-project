import os
import numpy as np
import torch
from PIL import Image
import json

def add_expt_config(arglist):
    # Load config file and merge into arglist
    config_path = os.path.join("./config", f"{arglist.expt}.json")
    with open(config_path) as f:
        cfg = json.load(f)
    for k, v in cfg.items():
        setattr(arglist, k, v)
    return arglist

def check_success(o_1, target, arglist):
    # Meta-world's info['success'] only checks if the default object (green) reached the goal
    # This does not work when we have multiple objects
    # Hence we define our function, which checks if the actual target object reached the goal
    
    if arglist.num_objects == 1:
        obj_pos = o_1[4:7]   # obj1 position

    elif arglist.num_objects == 2:
        if target == 0:
            obj_pos = o_1[4:7]   # obj1 position
        else:
            obj_pos = o_1[11:14] # obj2 position
    elif arglist.num_objects == 3:
        if target == 0:
            obj_pos = o_1[4:7]   # obj1 position
        elif target == 1:
            obj_pos = o_1[11:14] # obj2 position                    
        elif target == 2:
            obj_pos = o_1[18:21] # obj3 position
    
    goal_pos = o_1[-3:]
    correct_obj_success = float(np.linalg.norm(obj_pos - goal_pos) < 0.05)
    success = int(correct_obj_success)
    return success

def swap_obs(o, target, arglist):
    # Meta-world's expert policy goes to the first object in the observation by default
    # Hence, when the target is not the first object, we swap the target with the first object
    # in the observation vector
    if target == 0:
        return o
    elif arglist.num_objects == 2 and target == 1:
        new_o = o.copy()
        new_o[4:11] = o[11:18]
        new_o[11:18] = o[4:11]
        new_o[22:29] = o[29:36]
        new_o[29:36] = o[22:29]
        return new_o
    elif arglist.num_objects == 3 and target == 1:
        new_o = o.copy()
        new_o[4:11] = o[11:18]
        new_o[11:18] = o[4:11]
        new_o[29:36] = o[36:43]
        new_o[36:43] = o[29:36]
        return new_o
    elif arglist.num_objects == 3 and target == 2:
        new_o = o.copy()
        new_o[4:11] = o[18:25]
        new_o[18:25] = o[4:11]
        new_o[29:36] = o[43:50]
        new_o[43:50] = o[29:36]
        return new_o

def construct_observation_tensor(o, env, arglist, stats, device, target=None):
    # Proprioception
    if arglist.image:
        # In proprio we store only end-effector position and gripper state
        proprio = o[:4]
    else:
        # In proprio we store end-effector position, gripper state, object position, object orientation
        proprio = o[:11]
    if arglist.normalize:
        O = {'proprio': get_tensor(normalize(proprio, stats['proprio_mean'], stats['proprio_std'])).unsqueeze(0).to(device)}
    else:
        O = {'proprio': get_tensor(proprio).unsqueeze(0).to(device)}

    # Image
    if arglist.image:
        # Object position, object orientation must be inferred from rgb and depth images 
        rgb_array, depth_array = get_images(env)
        if arglist.normalize:
            O['rgb'] = get_tensor(normalize(rgb_array.astype(np.float32), stats['rgb_mean'], stats['rgb_std'])).unsqueeze(0).to(device)
            O['depth'] = get_tensor(normalize(depth_array, stats['depth_mean'], stats['depth_std'])).unsqueeze(0).to(device)
        else:
            O['rgb'] = get_tensor(rgb_array).unsqueeze(0).to(device)
            O['depth'] = get_tensor(depth_array).unsqueeze(0).to(device)

    # Text
    if arglist.text:
        O['text'] = get_tensor(np.array([target]),dtype=torch.long).unsqueeze(0).to(device)
    
    return O

def normalize(x, mean, std):
    return (x-mean)/std

def get_tensor(x, dtype=torch.float32):
    return torch.from_numpy(x).to(dtype=dtype)

def get_expert_policy(arglist):
    if arglist.env == "bin-picking-v3" or arglist.env == "bin-picking-two-objects-v3" or arglist.env == "bin-picking-three-objects-v3":
        from metaworld.policies.sawyer_bin_picking_v3_policy import SawyerBinPickingV3Policy as ExpertPolicy
    return ExpertPolicy()

def print_info(t, o, a, r, o_1, terminated, truncated, info):
    print(f"\nt: {t}"
          f"\nobservation: {o.shape}, action: {a.shape}, reward: {r}, "
          f"new observation: {o_1.shape}, terminated: {terminated}, truncated: {truncated}"
          f"\ninfo: {info}")

def get_images(env):
    rgb_copy = env.unwrapped.mujoco_renderer.render("rgb_array").copy()
    depth_copy = env.unwrapped.mujoco_renderer.render("depth_array").copy()
    rgb_array   = np.transpose(rgb_copy, (2,0,1))
    depth_array = depth_copy[None,:,:]
    return rgb_array, depth_array

def log_image(t, rgb_array, depth_array, path):
    rgb_img = Image.fromarray(np.transpose(rgb_array, (1,2,0)))
    rgb_img.save(os.path.join(path, "rgb_"+str(t)+".png"))

    depth_array_processed = ((depth_array[0] - depth_array.min()) / (depth_array.max() - depth_array.min()) * 255).astype(np.uint8)
    depth_img = Image.fromarray(depth_array_processed)
    depth_img.save(os.path.join(path, "depth_"+str(t)+".png"))