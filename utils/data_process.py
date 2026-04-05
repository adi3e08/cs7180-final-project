from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
import numpy as np
from PIL import Image
import os
import random
import shutil
import torch

class MetaWorldDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        demo_files = sorted([f for f in os.listdir(data_dir) 
                            if f.endswith(".npz")])
        for file in demo_files:
            data = np.load(os.path.join(data_dir, file), allow_pickle=True)
            T = len(data["actions"])
            text = str(data["text"])
            for t in range(T):
                self.samples.append({
                    "top_view": data["top_view"][t],           # (224, 224, 3)
                    "gripper_pov": data["gripper_pov"][t],     
                    "behind_gripper": data["behind_gripper"][t],
                    "proprioception": data["proprioception"][t],  # (4,)
                    "action": data["actions"][t],                 # (4,)
                    "text": text,
                })
        print(f"Loaded {len(self.samples)} timesteps from {len(demo_files)} episodes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "top_view": torch.FloatTensor(s["top_view"]).permute(2, 0, 1) / 255.0,
            "gripper_pov": torch.FloatTensor(s["gripper_pov"]).permute(2, 0, 1) / 255.0,
            "behind_gripper": torch.FloatTensor(s["behind_gripper"]).permute(2, 0, 1) / 255.0,
            "proprioception": torch.FloatTensor(s["proprioception"]),
            "action": torch.FloatTensor(s["action"]),
            "text": s["text"],
        }
    
def resize_frame(frame, size=(224, 224)):
    return np.array(Image.fromarray(frame).resize(size))

def collect_expert_demos(env_name, policy_class, num_episodes=10, max_steps=500, seed=42):
    """
    Collect expert demonstrations for imitation learning.
    Returns list of trajectories, each containing (obs, actions, rewards, success).
    """
    data_dir = "/content/cs7180-final-project/data"
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    envs = {}
    for cam in ["topview", "gripperPOV", "behindGripper"]:
        envs[cam] = gym.make("Meta-World/MT1", env_name=env_name,
                            seed=seed, render_mode="rgb_array", camera_name=cam)
    policy = policy_class()

    for ep in range(num_episodes):
        obs, info = envs["topview"].reset(seed=seed + ep)
        envs["gripperPOV"].reset(seed=seed + ep)
        envs["behindGripper"].reset(seed=seed + ep)
        trajectory = {
            "proprioception": [],  
            "actions":        [],  
            "text":           "pick the red ball", 
            "rewards":        [],
            "top_view_images": [],  
            "gripper_pov_images": [], 
            "behind_gripper_images": [],  
            "success":        False
        }
        success = False

        for step in range(max_steps):
            trajectory["top_view_images"].append(resize_frame(envs["topview"].render()))
            trajectory["gripper_pov_images"].append(resize_frame(envs["gripperPOV"].render()))
            trajectory["behind_gripper_images"].append(resize_frame(envs["behindGripper"].render()))
            action = policy.get_action(obs)
            trajectory["proprioception"].append(obs.copy()[:4])
            trajectory["actions"].append(action.copy())

            # Step ALL three environments with the same action
            obs, reward, terminated, truncated, info = envs["topview"].step(action)
            envs["gripperPOV"].step(action)
            envs["behindGripper"].step(action)
            trajectory["rewards"].append(reward)

            if int(info["success"]) == 1:
                success = True
                break
            if terminated or truncated:
                break

        trajectory["success"] = success
        trajectory["length"] = len(trajectory["rewards"])
        trajectory["proprioception"] = np.array(trajectory["proprioception"])
        trajectory["actions"] = np.array(trajectory["actions"])
        trajectory["rewards"] = np.array(trajectory["rewards"])
        trajectory["top_view_images"] = np.array(trajectory["top_view_images"])      
        trajectory["gripper_pov_images"] = np.array(trajectory["gripper_pov_images"]) 
        trajectory["behind_gripper_images"] = np.array(trajectory["behind_gripper_images"]) 
        np.savez_compressed(
            f"/content/cs7180-final-project/data/demo_{ep:03d}.npz",
            proprioception=trajectory["proprioception"],
            actions=trajectory["actions"],
            rewards=trajectory["rewards"],
            top_view=trajectory["top_view_images"],
            gripper_pov=trajectory["gripper_pov_images"],
            behind_gripper=trajectory["behind_gripper_images"],
            text=trajectory["text"],
            success=trajectory["success"],
        )
        # After saving each episode
        print(f"Episode {ep+1}/{num_episodes} | "
            f"steps={trajectory['length']} | "
            f"success={success} | "
            f"saved demo_{ep:03d}.npz")

    for cam in ["topview", "gripperPOV", "behindGripper"]:
        envs[cam].close()
    
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npz")])

    random.seed(101)
    random.shuffle(all_files)

    split = int(0.8 * len(all_files))
    train_files = all_files[:split]
    test_files = all_files[split:]

    for folder in ["train", "test"]:
        os.makedirs(f"{data_dir}/{folder}", exist_ok=True)

    for f in train_files:
        shutil.move(f"{data_dir}/{f}", f"{data_dir}/train/{f}")
    for f in test_files:
        shutil.move(f"{data_dir}/{f}", f"{data_dir}/test/{f}")

    print(f"Train: {len(train_files)} episodes, Test: {len(test_files)} episodes")

def load_data(batch_size):
    train_dataset = MetaWorldDataset("/content/cs7180-final-project/data/train")
    test_dataset = MetaWorldDataset("/content/cs7180-final-project/data/test")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

