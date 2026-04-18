import os
import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import metaworld
import sys
from src.model import FlowMatchingModel
from src.utils import normalize, get_tensor
import tqdm

class Dataset(torch.utils.data.Dataset):
    """
    dataset['proprio']: N, d_obs
    dataset['rgb']: N, 3, 240, 240
    dataset['depth']: N, 1, 240, 240
    dataset['text']: To be decided
    dataset['action']: N, d_act
    """
    def __init__(self, arglist, mode):
        super().__init__()
        data_dir = os.path.join("/content/drive/MyDrive/APLDL/data/raw/", arglist.expt)
        dataset = np.load(os.path.join(data_dir, mode+".npz"), allow_pickle=True)
        stats = np.load(os.path.join(data_dir, "stats.npz"), allow_pickle=True)
        self.arglist = arglist
        self.proprio = dataset['proprio']
        if self.arglist.image:
            self.rgb = dataset['rgb']
            self.depth = dataset['depth']
            self.topdown = dataset['topdown']
            self.bboxes = dataset['bboxes']
            self.labels = dataset['labels']
        if self.arglist.text:
            self.T = dataset['text']
        self.A = dataset['action']
        if self.arglist.normalize:
            self.proprio_mean = stats['proprio_mean']
            self.proprio_std = stats['proprio_std']        
            self.A_mean = stats['action_mean']
            self.A_std = stats['action_std']
            if self.arglist.image:
                self.rgb_mean = stats['rgb_mean']
                self.rgb_std = stats['rgb_std']
                self.depth_mean = stats['depth_mean']
                self.depth_std = stats['depth_std']
                self.topdown_mean = stats['topdown_mean']
                self.topdown_std = stats['topdown_std']
        self.dims = self.A.shape
    
    def __len__(self):
        return self.dims[0]

    def __getitem__(self, idx):
        n = idx

        if self.arglist.normalize:
            o =  {'proprio': get_tensor(normalize(self.proprio[n], self.proprio_mean, self.proprio_std))}
            a = get_tensor(normalize(self.A[n], self.A_mean, self.A_std))
            if self.arglist.image:
                o['rgb'] = get_tensor(normalize(self.rgb[n].astype(np.float32), self.rgb_mean, self.rgb_std))
                o['depth'] = get_tensor(normalize(self.depth[n], self.depth_mean, self.depth_std))
                o['topdown'] = get_tensor(normalize(self.topdown[n].astype(np.float32), self.topdown_mean, self.topdown_std))
                o['target'] = { 'boxes': get_tensor(self.bboxes[n]), 'labels': get_tensor(self.labels[n], dtype=torch.long) }
        else:
            o =  {'proprio': get_tensor(self.proprio[n])}
            a = get_tensor(self.A[n])
            if self.arglist.image:
                o['rgb'] = get_tensor(self.rgb[n])
                o['depth'] = get_tensor(self.depth[n])
                o['topdown'] = get_tensor(self.topdown[n])
                o['target'] = { 'boxes': get_tensor(self.bboxes[n]), 'labels': get_tensor(self.labels[n], dtype=torch.long) }
        if self.arglist.text:
            o['text'] = get_tensor(self.T[n])
        
        return o, a

def check(data_loader, model):
    O, A = next(iter(data_loader))
    for key in O:
        print(key, O[key].size())
    print("action", A.size())
    loss = model.loss(O, A)
    print("loss", loss)
    sys.exit(0)

def parse_args():
    parser = argparse.ArgumentParser("Flow matching")
    parser.add_argument("--env", type=str, default="bin-picking-v3", help="")
    parser.add_argument("--expt", type=str, default="expt_2", help="expt name")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    # Simulation parameters
    parser.add_argument("--d-proprio", type=int, default=4, help="proprio dimension, expt_1: 11, expt2: 4")
    parser.add_argument("--d-act", type=int, default=4, help="action dimension is 4 across meta-world tasks")
    parser.add_argument("--image", action=argparse.BooleanOptionalAction, default=False, help="expt_1: False, expt_2: True")
    parser.add_argument("--camera-id", type=int, default=6, help="6: gripper pov")
    parser.add_argument("--image-height", type=int, default=240, help="image height")
    parser.add_argument("--image-width", type=int, default=240, help="image width")
    parser.add_argument("--text", action="store_true", default=False)
    # Training parameters
    parser.add_argument("--T-flow", type=int, default=20, help="flow time steps for sampling")
    parser.add_argument("--d-model", type=int, default=128, help="hidden size dim, expt_1: 64, expt_2: 128")
    parser.add_argument("--d-emb", type=int, default=32, help="embedding dim (only for expt_2 currently)")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs to train, expt_1: 100, expt_2: 500")
    parser.add_argument("--evaluate-agent", action="store_true", default=False, help="evaluate agent performance periodically")
    parser.add_argument("--use_backbone", action="store_true", default=False, help="use backbone for image encoding")
    return parser.parse_args()


def collate_fn(batch):
    Os, As = zip(*batch)
    O_out = {}
    for k in Os[0].keys():
        if k == "target":
            O_out[k] = [o[k] for o in Os]   
        elif k == "topdown":
            O_out[k] = [o[k] for o in Os]  
        else:
            O_out[k] = torch.stack([o[k] for o in Os])  
    A = torch.stack(As)

    return O_out, A

def main():
    arglist = parse_args()

    np.random.seed(arglist.seed)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(arglist.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = os.path.join("./models", arglist.expt)
    results_dir = os.path.join("./results", arglist.expt)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=results_dir)

    if arglist.evaluate_agent:
        if arglist.image:
            env = gym.make('Meta-World/MT1', env_name=arglist.env, seed=arglist.seed, render_mode="rgb_array",\
                            camera_id=arglist.camera_id ,height=arglist.image_height,width=arglist.image_width)
        else:
            env = gym.make('Meta-World/MT1', env_name=arglist.env, seed=arglist.seed, render_mode='none')
    
    model = FlowMatchingModel(arglist).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    train_data = Dataset(arglist, "train")
    test_data = Dataset(arglist, "test")

    if torch.cuda.is_available():
        num_workers=2
        pin_memory=True
    else:
        num_workers=0
        pin_memory=False
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=arglist.batch_size, shuffle=True, 
                                               num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=arglist.batch_size, shuffle=True, 
                                              num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)
    print("Data loaded")
    # check(train_loader, model)

    # Loop over epochs
    best_test_loss = np.inf
    for epoch in range(arglist.epochs):
        print("Epoch ", epoch + 1, "/", arglist.epochs)
        # Training
        model.train()
        train_loss = []
        action_losses = []
        detection_losses = []
        for O, A in tqdm(train_loader, total=len(train_loader), desc="Training"):
            for k in O:
                if k not in ["target", "topdown"]:
                    O[k] = O[k].to(device)
            A = A.to(device)
            if "topdown" in O and "topdown" in O:
                O["topdown"] = [img.to(device) for img in O["topdown"]]
                O["target"] = [
                    {
                        "boxes": t["boxes"].to(device),
                        "labels": t["labels"].to(device),
                    }
                    for t in O["target"]
                ]
            action_loss, detection_loss = model.loss(O, A)
            loss = action_loss+ detection_loss
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss.append(loss.item())
            action_losses.append(action_loss.item())
            detection_losses.append(detection_loss.item())
        train_loss = np.array(train_loss).mean()
        action_loss = np.array(action_losses).mean()
        detection_loss = np.array(detection_losses).mean()
        print("train loss: ", train_loss, "action_loss: ", action_loss, "detection_loss: ", detection_loss)
        writer.add_scalar('train_loss', train_loss, epoch)

        # Testing
        model.eval()
        test_loss = []
        action_losses = []
        detection_losses = []
        for O, A in tqdm(test_loader, total=len(test_loader), desc="Testing"):
            for k in O:
                if k not in ["target", "topdown"]:
                    O[k] = O[k].to(device)
            if "topdown" in O and "topdown" in O:
                
                O["topdown"] = [img.to(device) for img in O["topdown"]]
                O["target"] = [
                    {
                        "boxes": t["boxes"].to(device),
                        "labels": t["labels"].to(device),
                    }
                    for t in O["target"]
                ]
            A = A.to(device)
            action_loss, detection_loss = model.loss(O, A)
            loss = action_loss+ detection_loss
            test_loss.append(loss.item())
            action_losses.append(action_loss.item())
            detection_losses.append(detection_loss.item())
        test_loss = np.array(test_loss).mean()
        action_loss = np.array(action_losses).mean()
        detection_loss = np.array(detection_losses).mean()
        print("test loss: ", test_loss, "action_loss: ", action_loss, "detection_loss: ", detection_loss)
        writer.add_scalar('test_loss', test_loss, epoch)
        if test_loss < best_test_loss:
            torch.save({'model' : model.state_dict(),
                        'optimizer' : optimizer.state_dict(), 
                        'epoch' : epoch}, os.path.join(model_dir, "best.ckpt"))
            best_test_loss = test_loss
        
        if epoch % 20 == 0 or epoch == arglist.epochs-1:
            torch.save({'model' : model.state_dict(),
                        'optimizer' : optimizer.state_dict(), 
                        'epoch' : epoch}, os.path.join(model_dir, str(epoch)+".ckpt"))

            if arglist.evaluate_agent:            
                # Evaluate agent performance over several episodes
                # This block runs fine on local. 
                # Throws errors in Colab which is a headless environment (no display). Need to fix.
                metric = []
                for episode in range(5):
                    o, info = env.reset()
                    while True:
                        a = model.sample(o, env, device)
                        o_1, r, terminated, truncated, info = env.step(a)
                        success = int(info['success'])
                        done = terminated or truncated or success
                        o = o_1
                        if done:
                            metric.append(success)
                            break
                writer.add_scalar('task_success_rate', np.mean(metric), epoch)

    writer.close()

if __name__ == '__main__':
    main()
