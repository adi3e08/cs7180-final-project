import os
import numpy as np
import torch
from PIL import Image
import mujoco

def construct_observation_tensor(o, env, env_top, arglist, stats, device):
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
        rgb_array, depth_array, top_rgb = get_images(env, env_top)
        if arglist.normalize:
            O['rgb'] = get_tensor(normalize(rgb_array.astype(np.float32), stats['rgb_mean'], stats['rgb_std'])).unsqueeze(0).to(device)
            O['depth'] = get_tensor(normalize(depth_array, stats['depth_mean'], stats['depth_std'])).unsqueeze(0).to(device)
            O['topdown'] = [get_tensor(normalize(top_rgb, stats['topdown_mean'], stats['topdown_std'])).to(device)]
            O["target"] = None
        else:
            O['rgb'] = get_tensor(rgb_array).unsqueeze(0).to(device)
            O['depth'] = get_tensor(depth_array).unsqueeze(0).to(device)
            O['topdown'] = [get_tensor(top_rgb).unsqueeze(0).to(device)]
            O["target"] = None
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

def get_images(env, env_top):
    rgb_copy = env.unwrapped.mujoco_renderer.render("rgb_array").copy()
    depth_copy = env.unwrapped.mujoco_renderer.render("depth_array").copy()
    rgb_array   = np.transpose(rgb_copy, (2,0,1))
    depth_array = depth_copy[None,:,:]
    top_rgb = env_top.unwrapped.mujoco_renderer.render("rgb_array").copy()
    top_rgb = np.transpose(top_rgb, (2, 0, 1))
    return rgb_array, depth_array, top_rgb

def log_image(t, rgb_array, depth_array, path):
    rgb_img = Image.fromarray(np.transpose(rgb_array, (1,2,0)))
    rgb_img.save(os.path.join(path, "rgb_"+str(t)+".png"))

    depth_array_processed = ((depth_array[0] - depth_array.min()) / (depth_array.max() - depth_array.min()) * 255).astype(np.uint8)
    depth_img = Image.fromarray(depth_array_processed)
    depth_img.save(os.path.join(path, "depth_"+str(t)+".png"))
    
    
def world_to_pixel(model, data, cam_name, obj_world_pos, img_h, img_w):
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    cam_pos = data.cam_xpos[cam_id]
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)

    point_cam = cam_mat @ (obj_world_pos - cam_pos)

    fovy = model.cam_fovy[cam_id]
    f = (img_h / 2) / np.tan(np.deg2rad(fovy / 2))
    px = int(f * -point_cam[0] / point_cam[2] + img_w / 2)
    py = int(f * point_cam[1] / point_cam[2] + img_h / 2)
    return px, py

def make_bbox_from_3d(model, data, cam_name, obj_world_pos, obj_size, img_h, img_w):
    cx, cy, cz = obj_world_pos
    w, h, d = obj_size  # half-extents or full size depending on your convention
    
    # 8 corners of the 3D bounding box
    corners = np.array([
        [cx - w/2, cy - h/2, cz - d/2],
        [cx + w/2, cy - h/2, cz - d/2],
        [cx - w/2, cy + h/2, cz - d/2],
        [cx + w/2, cy + h/2, cz - d/2],
        [cx - w/2, cy - h/2, cz + d/2],
        [cx + w/2, cy - h/2, cz + d/2],
        [cx - w/2, cy + h/2, cz + d/2],
        [cx + w/2, cy + h/2, cz + d/2],
    ])
    
    # Project all corners to pixel space
    pixels = []
    for corner in corners:
        px, py = world_to_pixel(model, data, cam_name, corner, img_h, img_w)
        pixels.append((px, py))
    
    pixels = np.array(pixels)
    x1 = max(pixels[:, 0].min(), 0)
    y1 = max(pixels[:, 1].min(), 0)
    x2 = min(pixels[:, 0].max(), img_w)
    y2 = min(pixels[:, 1].max(), img_h)
    
    return np.array([x1, y1, x2, y2], dtype=np.int32)

def make_pixel_bbox(px, py, half_w, half_h, img_w, img_h):
    x1 = max(px - half_w, 0)
    y1 = max(py - half_h, 0)
    x2 = min(px + half_w, img_w)
    y2 = min(py + half_h, img_h)
    return np.array([x1, y1, x2, y2], dtype=np.int32)