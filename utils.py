import os
import numpy as np
from PIL import Image

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