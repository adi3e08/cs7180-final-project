import warnings
warnings.filterwarnings("ignore")
import os
import glob
import argparse
import numpy as np
import gymnasium as gym
import metaworld
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from src.utils import get_expert_policy, get_images, swap_obs, check_success, add_expt_config

def parse_args():
    parser = argparse.ArgumentParser("Create Dataset")
    parser.add_argument("--expt", type=str, default="expt_4", help="expt name")
    parser.add_argument("--seed", type=int,  default=0)
    arglist = parser.parse_args()
    arglist = add_expt_config(arglist)
    return arglist

def main():
    arglist = parse_args()
    np.random.seed(arglist.seed)

    if arglist.image:
        env = gym.make('Meta-World/MT1', env_name=arglist.env, seed=arglist.seed, render_mode="rgb_array",\
                        camera_id=arglist.camera_id ,height=arglist.image_height,width=arglist.image_width)
    else:
        env = gym.make('Meta-World/MT1', env_name=arglist.env, seed=arglist.seed, render_mode="none")        

    policy = get_expert_policy(arglist)
    
    data_dir = os.path.join("./data/raw", arglist.expt)
    os.mkdir(data_dir)  
    
    metric = []
    for episode in tqdm(range(arglist.dataset_episodes)):
        proprio, action = [], []
        if arglist.image:
            rgb = []
            depth = []
        if arglist.text:
            text = []
        o, info = env.reset()
        if arglist.text:
            target = np.random.choice(arglist.num_objects)
        else:
            target = 0
        while True:
            if arglist.text: # expt_3 and expt_4
                o = swap_obs(o, target, arglist)
                text.append(np.uint8(target))
            
            if arglist.image: # expt_2 and # expt_3
                # In proprio we store only end-effector position and gripper state
                proprio.append(o[:4].astype(np.float32))
                # Object position, object orientation 
                # must be inferred from rgb and depth images 
                rgb_array, depth_array = get_images(env)
                rgb.append(rgb_array.astype(np.uint8))
                depth.append(depth_array.astype(np.float32))
            
            else: # expt_1
                # In proprio we store end-effector position, gripper state, object position, 
                # object orientation
                proprio.append(o[:11].astype(np.float32))
            
            a = policy.get_action(o)
            o_1, r, terminated, truncated, info = env.step(a)
            success = check_success(o_1, target, arglist)
            done = terminated or truncated or success
            action.append(a.astype(np.float32))
            o = o_1
            if done:
                if success:
                    # Save episode immediately, don't accumulate in RAM
                    data = {'proprio': np.array(proprio), 'action': np.array(action)}
                    if arglist.image:
                        data['rgb'] = np.array(rgb).astype(np.uint8)
                        data['depth'] = np.array(depth).astype(np.float32)
                    if arglist.text:
                        data['text'] = np.array(text)
                    np.savez_compressed(os.path.join(data_dir, f"ep_{episode}.npz"), **data)
                metric.append(success)
                break

    print(f"Task success rate on {arglist.dataset_episodes} episodes: {np.mean(metric)}")

    # Merge all episode files
    all_files = sorted(glob.glob(os.path.join(data_dir, "ep_*.npz")))

    proprio = np.concatenate([np.load(f)['proprio'] for f in all_files])
    action  = np.concatenate([np.load(f)['action']  for f in all_files])
    if arglist.image:
        rgb     = np.concatenate([np.load(f)['rgb']     for f in all_files])
        depth   = np.concatenate([np.load(f)['depth']   for f in all_files])
    if arglist.text:
        text    = np.concatenate([np.load(f)['text']    for f in all_files])
    
    train_indices, test_indices = train_test_split(np.arange(proprio.shape[0]), test_size=0.2, random_state=arglist.seed)
    train_data = {'proprio': proprio[train_indices], 'action': action[train_indices]}
    test_data = {'proprio': proprio[test_indices], 'action': action[test_indices]}
    if arglist.image:
        train_data['rgb'] = rgb[train_indices]
        test_data['rgb'] = rgb[test_indices]

        train_data['depth'] = depth[train_indices]
        test_data['depth'] = depth[test_indices]

    if arglist.text:
        train_data['text'] = text[train_indices]
        test_data['text'] = text[test_indices]

    # Calculate Mean and Std
    stats = {}
    stats['proprio_mean'] = np.mean(train_data['proprio'], axis=0)
    stats['proprio_std']  = np.std(train_data['proprio'], axis=0) + 1e-6 # Avoid division by zero
    
    stats['action_mean'] = np.mean(train_data['action'], axis=0)
    stats['action_std']  = np.std(train_data['action'], axis=0) + 1e-6

    if arglist.image:
        stats['rgb_mean'] = train_data['rgb'].astype(np.float32).mean(axis=(0,2,3)).reshape(3,1,1)
        stats['rgb_std']  = train_data['rgb'].astype(np.float32).std(axis=(0,2,3)).reshape(3,1,1)  + 1e-6
        stats['depth_mean'] = train_data['depth'].mean(axis=(0,2,3)).reshape(1,1,1)
        stats['depth_std']  = train_data['depth'].std(axis=(0,2,3)).reshape(1,1,1) + 1e-6

    # Save everything
    np.savez_compressed(os.path.join(data_dir, "train.npz"), **train_data)
    np.savez_compressed(os.path.join(data_dir, "test.npz"), **test_data)
    np.savez_compressed(os.path.join(data_dir, "stats.npz"), **stats)

    print("Saved dataset")

    # Delete the episode files
    for f in all_files:
        os.remove(f)

    print("Delted episode files")

    print("Done")

def simple_check():
    arglist = parse_args()
    data_dir = os.path.join("./data/raw", arglist.expt)
    for mode in ["train", "test", "stats"]:
        dataset = np.load(os.path.join(data_dir, mode+".npz"), allow_pickle=True)
        print(f"\n{mode}")
        for key in dataset:
            value = dataset[key]
            print(f"  {key}: dtype: {value.dtype} shape: {value.shape}")
            # if key == 'text':
            #     print(dataset[key])


if __name__ == '__main__':
    main()
    simple_check()