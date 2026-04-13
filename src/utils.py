import os
import numpy as np
import torch
from PIL import Image

def construct_observation_tensor(o, env, arglist, stats, device):
    if arglist.image:
        # In proprio we store only end-effector position and gripper state
        proprio = o[:4]
    else:
        # In proprio we store end-effector position, gripper state,  
        # object position, object orientation
        proprio = o[:11]
    if arglist.normalize:
        O = {'proprio': get_tensor(normalize(proprio, stats['proprio_mean'], stats['proprio_std'])).unsqueeze(0).to(device)}
    else:
        O = {'proprio': get_tensor(proprio).unsqueeze(0).to(device)}

    if arglist.image:
        # Object position, object orientation must be inferred from rgb and depth images 
        rgb_array, depth_array = get_images(env)
        if arglist.normalize:
            O['rgb'] = get_tensor(normalize(rgb_array.astype(np.float32), stats['rgb_mean'], stats['rgb_std'])).unsqueeze(0).to(device)
            O['depth'] = get_tensor(normalize(depth_array, stats['depth_mean'], stats['depth_std'])).unsqueeze(0).to(device)
        else:
            O['rgb'] = get_tensor(rgb_array).unsqueeze(0).to(device)
            O['depth'] = get_tensor(depth_array).unsqueeze(0).to(device)
    return O

def normalize(x, mean, std):
    return (x-mean)/std

def get_tensor(x, dtype=torch.float32):
    return torch.from_numpy(x).to(dtype=dtype)

def get_expert_policy(arglist):
    if arglist.env == "reach-v3":
        from metaworld.policies.sawyer_reach_v3_policy import SawyerReachV3Policy as ExpertPolicy
    elif arglist.env == "pick-place-v3":
        from metaworld.policies.sawyer_pick_place_v3_policy import SawyerPickPlaceV3Policy as ExpertPolicy
    elif arglist.env == "pick-place-wall-v3":
        from metaworld.policies.sawyer_pick_place_wall_v3_policy import SawyerPickPlaceWallV3Policy as ExpertPolicy
    elif arglist.env == "shelf-place-v3":
        from metaworld.policies.sawyer_shelf_place_v3_policy import SawyerShelfPlaceV3Policy as ExpertPolicy
    elif arglist.env == "bin-picking-v3":
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