import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import metaworld
import math
import sys
from utils import get_images

class CNN1(nn.Module):
    def __init__(self, d_emb):
        super().__init__()
        self.conv1 = nn.Conv2d(4,  32,  kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64,  kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.pool  = nn.AdaptiveAvgPool2d((1, 1))
        self.proj  = nn.Linear(128, d_emb)

    def forward(self, x):
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        x = F.silu(self.conv3(x))
        x = self.pool(x).flatten(1)  # (B, 128)
        x = self.proj(x)
        return x  # (B, d_emb)

class MLPVectorField2(nn.Module):
    def __init__(self, arglist):
        super().__init__()
        self.arglist = arglist 
        
        # Observation encoding
        self.proprio_encoder = nn.Linear(arglist.d_proprio, arglist.d_emb)
        if arglist.image:
            self.image_encoder = CNN1(arglist.d_emb)
        if arglist.text:
            self.text_encoder = nn.Linear(512, arglist.d_emb)

        # Time encoding
        self.time_encoder = nn.Linear(1, arglist.d_emb)

        # Action encoding
        self.action_encoder = nn.Linear(arglist.d_act, arglist.d_emb)
        
        input_dim = arglist.d_emb * (3+int(self.arglist.image)+int(self.arglist.text))

        # Core vector field
        self.mlp = nn.Sequential(nn.Linear(input_dim, arglist.d_model), nn.SiLU(),
                                 nn.Linear(arglist.d_model, arglist.d_model), nn.SiLU(),
                                 # nn.Linear(arglist.d_model, arglist.d_model), nn.SiLU(),
                                 nn.Linear(arglist.d_model, arglist.d_act))

    def forward(self, O, A, tau):
        """
        O['proprio']: B, d_proprio
        O['image']: B, 3, 64, 64
        O['text']: B, 512 # later
        A: B, d_act
        tau: B, 1
        """
        
        # Observation encoding
        obs_emb = []
        
        proprio_emb = self.proprio_encoder(O['proprio'])
        obs_emb.append(proprio_emb)

        if self.arglist.image:
            rgbd = torch.cat((O['rgb'],O['depth']),1)
            image_emb = self.image_encoder(rgbd)
            obs_emb.append(image_emb)
        
        if self.arglist.text:
            text_emb = self.text_encoder(O['text'])
            obs_emb.append(text_emb)
        
        # Time encoding
        time_emb = self.time_encoder(tau)

        # Action encoding
        A_emb = self.action_encoder(A)

        v = self.mlp(torch.cat(obs_emb + [A_emb, time_emb], dim=-1))
        return v

class MLPVectorField1(nn.Module):
    def __init__(self, arglist):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(arglist.d_proprio + arglist.d_act + 1, arglist.d_model), nn.SiLU(),
                                 nn.Linear(arglist.d_model, arglist.d_model), nn.SiLU(),
                                 nn.Linear(arglist.d_model, arglist.d_act))

    def forward(self, O, A, tau):
        return self.mlp(torch.cat([O['proprio'], A, tau], dim=-1))

class FlowModel(nn.Module):
    def __init__(self, arglist):
        super().__init__()
        self.arglist = arglist
        self.vector_field = MLPVectorField2(arglist)
    
    def loss(self, O, A):
        eps = torch.randn_like(A)
        tau = torch.rand_like(A[:,:1])
        A_noisy = tau * A + (1-tau) * eps
        return nn.functional.mse_loss(self.vector_field(O, A_noisy, tau), A - eps)

    def rk1(self, O, A, tau, h):
        k1 = self.vector_field(O, A, tau)
        return A + h * k1

    def rk2(self, O, A, tau, h):
        k1 = self.vector_field(O, A, tau)
        k2 = self.vector_field(O, A + h * k1, tau + h)
        # return A + h * (k1 + k2) / 2.0
        alpha = 2.0/3.0 # Ralston's method
        return A + h * ((1.0 - 1.0/(2.0*alpha))*k1 + (1.0/(2.0*alpha))*k2)   

    def rk4(self, O, A, tau, h):
        k1 = self.vector_field(O, A, tau)
        k2 = self.vector_field(O, A + h * k1 / 2, tau + h / 2)
        k3 = self.vector_field(O, A + h * k2 / 2, tau + h / 2)
        k4 = self.vector_field(O, A + h * k3, tau + h)
        return A + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    def sample(self, O):
        device = O['proprio'].device
        n_samples = O['proprio'].size(0)
        h = 1 / self.arglist.T_flow
        tau = torch.zeros(n_samples, 1, device=device)
        A = torch.randn(n_samples, self.arglist.d_act, device=device)
        with torch.no_grad():
            for i in range(self.arglist.T_flow):
                A = self.rk2(O, A, tau, h)
                tau = tau + h
        return A

def normalize(x, mean, std):
    return (x-mean)/std

def get_tensor(x, dtype=torch.float32):
    return torch.from_numpy(x).to(dtype=dtype)

class Dataset(torch.utils.data.Dataset):
    """
    dataset['proprio']: N, d_obs
    dataset['action']: N, d_act
    """
    def __init__(self, path, arglist):
        super().__init__()
        dataset = np.load(path)
        self.arglist = arglist
        self.proprio = dataset['proprio']
        if self.arglist.image:
            self.rgb = dataset['rgb']
            self.depth = dataset['depth']
        if self.arglist.text:
            self.T = dataset['text']
        self.A = dataset['action']
        if self.arglist.normalize:
            self.proprio_mean = dataset['proprio_mean']
            self.proprio_std = dataset['proprio_std']        
            self.A_mean = dataset['action_mean']
            self.A_std = dataset['action_std']
            if self.arglist.image:
                self.rgb_mean = dataset['rgb_mean']
                self.rgb_std = dataset['rgb_std']
                self.depth_mean = dataset['depth_mean']
                self.depth_std = dataset['depth_std']
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
        else:
            o =  {'proprio': get_tensor(self.proprio[n])}
            a = get_tensor(self.A[n])
            if self.arglist.image:
                o['rgb'] = get_tensor(self.rgb[n])
                o['depth'] = get_tensor(self.depth[n])
        
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
    # a = model.sample(O)
    # print("sampled action", a.size())
    sys.exit(0)

def parse_args():
    parser = argparse.ArgumentParser("Flow matching")
    # Settings
    parser.add_argument("--env", type=str, default="bin-picking-v3", help="")
    parser.add_argument("--image", action="store_true", default=True)
    parser.add_argument("--camera-id", type=int, default=6, help="6: gripper pov")
    parser.add_argument("--image-height", type=int, default=240, help="image height")
    parser.add_argument("--image-width", type=int, default=240, help="image width")
    parser.add_argument("--text", action="store_true", default=False)
    parser.add_argument("--expt", type=str, default="expt_proprio_image_128_3_emb_32", help="expt name")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--T-flow", type=int, default=20, help="flow time steps for sampling")
    parser.add_argument("--d-model", type=int, default=128, help="hidden size dim")
    parser.add_argument("--d-emb", type=int, default=32, help="hidden size dim")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs to train")
    parser.add_argument("--evaluate-agent", action="store_true", default=False, help="evaluate agent performance periodically")
    return parser.parse_args()

def main():
    arglist = parse_args()

    np.random.seed(arglist.seed)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(arglist.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_dir = os.path.join("./log", arglist.env)
    flow_matching_dir = os.path.join(env_dir, "flow_matching")
    expt_dir = os.path.join(flow_matching_dir, arglist.expt)
    seed_dir = os.path.join(expt_dir, "seed_"+str(arglist.seed))
    model_dir = os.path.join(seed_dir, "models")
    tensorboard_dir = os.path.join(seed_dir, "tensorboard")
    if os.path.exists(expt_dir):
        pass            
    else:
        os.makedirs(expt_dir)
    os.mkdir(seed_dir)
    os.mkdir(model_dir)
    os.mkdir(tensorboard_dir)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    if arglist.evaluate_agent:
        if arglist.image:
            env = gym.make('Meta-World/MT1', env_name=arglist.env, seed=arglist.seed, render_mode="rgb_array",\
                            camera_id=arglist.camera_id ,height=arglist.image_height,width=arglist.image_width)
        else:
            env = gym.make('Meta-World/MT1', env_name=arglist.env, seed=arglist.seed, render_mode='none')
    
    if arglist.image:
        arglist.d_proprio = 4
    else:
        arglist.d_proprio = 14
    arglist.d_act = 4 # action size
    model = FlowModel(arglist).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    train_data = Dataset(os.path.join(flow_matching_dir, "train.npz"), arglist)
    test_data = Dataset(os.path.join(flow_matching_dir, "test.npz"), arglist)
    if arglist.normalize:
        proprio_mean = train_data.proprio_mean
        proprio_std = train_data.proprio_std
        A_mean = train_data.A_mean
        A_std = train_data.A_std
        if arglist.image:
            rgb_mean = train_data.rgb_mean
            rgb_std = train_data.rgb_std
            depth_mean = train_data.depth_mean
            depth_std = train_data.depth_std
    if torch.cuda.is_available():
        num_workers=2
        pin_memory=True
    else:
        num_workers=0
        pin_memory=False
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=arglist.batch_size, shuffle=True, 
                                               num_workers=num_workers, pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=arglist.batch_size, shuffle=True, 
                                              num_workers=num_workers, pin_memory=pin_memory)

    # check(train_loader, model)

    # Loop over epochs
    best_test_loss = np.inf
    for epoch in range(arglist.epochs):
        
        # Training
        model.train()
        train_loss = []
        for O, A in train_loader:
            O = {k: v.to(device) for k, v in O.items()}
            A = A.to(device)
            loss = model.loss(O, A)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss.append(loss.item())
        train_loss = np.array(train_loss).mean()
        writer.add_scalar('train_loss', train_loss, epoch)

        # Testing
        model.eval()
        test_loss = []
        for O, A in test_loader:
            O = {k: v.to(device) for k, v in O.items()}
            A = A.to(device)
            loss = model.loss(O, A)
            test_loss.append(loss.item())
        test_loss = np.array(test_loss).mean()
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
                metric = []
                for episode in range(5):
                    o, info = env.reset()
                    while True:
                        if arglist.image:
                            proprio = o[:4]
                        else:
                            proprio = np.concatenate((o[:11],o[-3:]))
                        if arglist.normalize:
                            O = {'proprio': get_tensor(normalize(proprio, proprio_mean, proprio_std)).unsqueeze(0).to(device)}
                        else:
                            O = {'proprio': get_tensor(proprio).unsqueeze(0).to(device)}

                        if arglist.image:
                            rgb_array, depth_array = get_images(env)
                            if arglist.normalize:
                                O['rgb'] = get_tensor(normalize(rgb_array.astype(np.float32), rgb_mean, rgb_std)).unsqueeze(0).to(device)
                                O['depth'] = get_tensor(normalize(depth_array, depth_mean, depth_std)).unsqueeze(0).to(device)
                            else:
                                O['rgb'] = get_tensor(rgb_array).unsqueeze(0).to(device)
                                O['depth'] = get_tensor(depth_array).unsqueeze(0).to(device)

                        with torch.no_grad():
                            A = model.sample(O)
                        a = A.cpu().numpy()[0]
                        if arglist.normalize:
                            a = a * A_std + A_mean
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
