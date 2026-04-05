from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import gymnasium as gym
import numpy as np
from pil import Image
import os

def resize_frame(frame, size=(224, 224)):
    return np.array(Image.fromarray(frame).resize(size))

def collect_expert_demos(env_name, policy_class, num_episodes=10, max_steps=500, seed=42):
    """
    Collect expert demonstrations for imitation learning.
    Returns list of trajectories, each containing (obs, actions, rewards, success).
    """
    os.makedirs("/content/cs7180-final-project/data", exist_ok=True)
    envs = {}
    for cam in ["topview", "gripperPOV", "behindGripper"]:
        envs[cam] = gym.make("Meta-World/MT1", env_name=env_name,
                            seed=seed, render_mode="rgb_array", camera_name=cam)
    # env = gym.make("Meta-World/MT1",env_name=env_name,seed=seed,render_mode="rgb_array",camera_name="topview",)
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

def load_data(BATCH, DEVICE):
    
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  
    ])

    train_ds = datasets.MNIST(root="data", train=True,  download=True, transform=tfm)
    val_ds   = datasets.MNIST(root="data", train=False, download=True, transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Device: {DEVICE}")
    return train_loader, val_loader

