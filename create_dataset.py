import os
import argparse
import numpy as np
import gymnasium as gym
import metaworld
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser("Create Dataset")
    parser.add_argument("--env",      type=str,  default="pick-place-v3")
    parser.add_argument("--image",    action="store_true", default=False)
    parser.add_argument("--text",     action="store_true", default=False)
    parser.add_argument("--seed",     type=int,  default=0)
    parser.add_argument("--episodes", type=int,  default=100)
    return parser.parse_args()

def main(arglist):
    if arglist.env == "pick-place-v3":
        from metaworld.policies.sawyer_pick_place_v3_policy import SawyerPickPlaceV3Policy as ExpertPolicy
    if arglist.image:
        render_mode = 'rgb_array'
    else:
        render_mode = 'none'
    np.random.seed(arglist.seed)

    env = gym.make('Meta-World/MT1', env_name=arglist.env, seed=arglist.seed, render_mode=render_mode)

    policy = ExpertPolicy()

    Q, A = [], []
    # if arglist.image:
    #     I = []
    # if arglist.text:
    #     T = []
    for episode in tqdm(range(arglist.episodes)):
        o, info = env.reset()
        while True:
            Q.append(o.astype(np.float32))
            # if arglist.image:
            #     I.append(o['image'].astype(np.uint8))
            a = policy.get_action(o)
            o_1, r, terminated, truncated, info = env.step(a)
            done = int(info['success']) == 1
            A.append(a.astype(np.float32))
            o = o_1
            if done:
                break

        # if arglist.text:
        #     T.append(env.text_instruction)

    Q, A = np.array(Q), np.array(A)
    # if arglist.image:
    #     I = np.array(I)
    # if arglist.text:
    #     T = np.array(T, dtype=np.dtypes.StringDType())

    train_indices, test_indices = train_test_split(np.arange(Q.shape[0]), test_size=0.2, random_state=arglist.seed)
    train_data = {'proprio': Q[train_indices], 'action': A[train_indices]}
    test_data = {'proprio': Q[test_indices], 'action': A[test_indices]}
    # if arglist.image:
    #     train_data['image'] = I[train_indices]
    #     test_data['image'] = I[test_indices]
    # if arglist.text:
    #     train_data['text'] = T[train_indices]
    #     test_data['text'] = T[test_indices]

    # 2. Calculate Mean and Std
    proprio_mean = np.mean(train_data['proprio'], axis=0)
    proprio_std  = np.std(train_data['proprio'], axis=0) + 1e-6 # Avoid division by zero
    
    action_mean = np.mean(train_data['action'], axis=0)
    action_std  = np.std(train_data['action'], axis=0) + 1e-6

    # 3. Add stats to the saved dictionary so the model can load them later
    stats = {
        'proprio_mean': proprio_mean,
        'proprio_std':  proprio_std,
        'action_mean':  action_mean,
        'action_std':   action_std
    }

    # 4. Save everything
    expt_dir = os.path.join("./log", arglist.env, "flow_matching")
    
    # Save train/test as usual, but include the stats in train.npz or a separate meta.npz
    np.savez_compressed(os.path.join(expt_dir, "train.npz"), **train_data, **stats)
    np.savez_compressed(os.path.join(expt_dir, "test.npz"), **test_data, **stats)
    print(f"Stats saved: Proprio Std range [{proprio_std.min():.4f}, {proprio_std.max():.4f}]")

    print("Done.")

def simple_check(arglist):
    for mode in ["train", "test"]:
        dataset = np.load(os.path.join("./log", arglist.env,
                                       "flow_matching", mode+".npz"), allow_pickle=True)
        print(f"\n{mode}")
        for key in dataset:
            shape = dataset[key].shape
            print(f"  {key}: shape: {shape}")
            # if key == 'text':
            #     print(dataset[key])

if __name__ == '__main__':
    arglist = parse_args()
    # main(arglist)
    simple_check(arglist)