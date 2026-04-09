import os
import argparse
import numpy as np
import gymnasium as gym
import metaworld
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils import get_expert_policy, get_images

def parse_args():
    parser = argparse.ArgumentParser("Create Dataset")
    parser.add_argument("--env",      type=str,  default="bin-picking-v3")
    parser.add_argument("--image",    action="store_true", default=True)
    parser.add_argument("--camera-id", type=int, default=6, help="6: gripper pov")
    parser.add_argument("--image-height", type=int, default=240, help="image height")
    parser.add_argument("--image-width", type=int, default=240, help="image width")
    parser.add_argument("--text",     action="store_true", default=False)
    parser.add_argument("--seed",     type=int,  default=0)
    parser.add_argument("--episodes", type=int,  default=100)
    return parser.parse_args()

def main(arglist):
    np.random.seed(arglist.seed)

    if arglist.image:
        env = gym.make('Meta-World/MT1', env_name=arglist.env, seed=arglist.seed, render_mode="rgb_array",\
                        camera_id=arglist.camera_id ,height=arglist.image_height,width=arglist.image_width)
    else:
        env = gym.make('Meta-World/MT1', env_name=arglist.env, seed=arglist.seed, render_mode='none')        

    policy = get_expert_policy(arglist)

    proprio, action = [], []
    if arglist.image:
        rgb = []
        depth = []
    # if arglist.text:
    #     T = []
    for episode in tqdm(range(arglist.episodes)):
        o, info = env.reset()
        while True:
            if arglist.image:
                proprio.append(o[:4].astype(np.float32))
            else:
                proprio.append(np.concatenate((o[:11],o[-3:])).astype(np.float32))
            if arglist.image:
                rgb_array, depth_array = get_images(env)
                rgb.append(rgb_array.astype(np.uint8))
                depth.append(depth_array.astype(np.float32))
            a = policy.get_action(o)
            o_1, r, terminated, truncated, info = env.step(a)
            done = int(info['success']) == 1
            action.append(a.astype(np.float32))
            o = o_1
            if done:
                break

        # if arglist.text:
        #     T.append(env.text_instruction)

    proprio, action = np.array(proprio), np.array(action)
    if arglist.image:
        rgb = np.array(rgb)
        depth = np.array(depth)
    # if arglist.text:
    #     T = np.array(T, dtype=np.dtypes.StringDType())

    train_indices, test_indices = train_test_split(np.arange(proprio.shape[0]), test_size=0.2, random_state=arglist.seed)
    train_data = {'proprio': proprio[train_indices], 'action': action[train_indices]}
    test_data = {'proprio': proprio[test_indices], 'action': action[test_indices]}
    if arglist.image:
        train_data['rgb'] = rgb[train_indices]
        test_data['rgb'] = rgb[test_indices]

        train_data['depth'] = depth[train_indices]
        test_data['depth'] = depth[test_indices]

    # if arglist.text:
    #     train_data['text'] = T[train_indices]
    #     test_data['text'] = T[test_indices]

    # 2. Calculate Mean and Std
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

    # 3. Save everything
    expt_dir = os.path.join("./log", arglist.env, "flow_matching")
    try:
        os.makedirs(expt_dir)
    except:
        pass
    
    # Save train/test as usual, but include the stats in train.npz or a separate meta.npz
    np.savez_compressed(os.path.join(expt_dir, "train.npz"), **train_data, **stats)
    np.savez_compressed(os.path.join(expt_dir, "test.npz"), **test_data, **stats)

    print("Done.")

def simple_check(arglist):
    for mode in ["train", "test"]:
        dataset = np.load(os.path.join("./log", arglist.env,
                                       "flow_matching", mode+".npz"), allow_pickle=True)
        print(f"\n{mode}")
        for key in dataset:
            value = dataset[key]
            print(f"  {key}: dtype: {value.dtype} shape: {value.shape}, min: {np.min(value)}, max: {np.max(value)}")
            # if key == 'text':
            #     print(dataset[key])

if __name__ == '__main__':
    arglist = parse_args()
    main(arglist)
    simple_check(arglist)