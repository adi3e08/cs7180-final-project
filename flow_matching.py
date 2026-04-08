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

class MLPVectorField1(nn.Module):
    def __init__(self, arglist):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(arglist.d_proprio + arglist.d_act + 1, arglist.d_model), nn.SiLU(),
                                 nn.Linear(arglist.d_model, arglist.d_model), nn.SiLU(),
                                 # nn.Linear(arglist.d_model, arglist.d_model), nn.SiLU(),
                                 nn.Linear(arglist.d_model, arglist.d_act))

    def forward(self, O, A, tau):
        return self.mlp(torch.cat([O['proprio'], A, tau], dim=-1))

class FlowModel(nn.Module):
    def __init__(self, arglist):
        super().__init__()
        self.arglist = arglist
        self.vector_field = MLPVectorField1(arglist)
    
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

def normalize(x, mu, sigma):
    return (x-mu)/sigma

class Dataset(torch.utils.data.Dataset):
    """
    dataset['proprio']: N, d_obs
    dataset['action']: N, d_act
    """
    def __init__(self, path, arglist):
        super().__init__()
        dataset = np.load(path)
        self.arglist = arglist
        self.Q = dataset['proprio']
        if self.arglist.image:
            self.I = dataset['image']
        if self.arglist.text:
            self.T = dataset['text']
        self.A = dataset['action']
        if self.arglist.normalize:
            self.Q_mu = dataset['proprio_mean']
            self.Q_sigma = dataset['proprio_std']        
            self.A_mu = dataset['action_mean']
            self.A_sigma = dataset['action_std']
        self.dims = self.A.shape
        self.dtype = torch.get_default_dtype()

    def get_tensor(self, x):
        return torch.from_numpy(x).to(dtype=self.dtype)
    
    def __len__(self):
        return self.dims[0]

    def __getitem__(self, idx):
        n = idx

        if self.arglist.normalize:
            o =  {'proprio': self.get_tensor(normalize(self.Q[n], self.Q_mu, self.Q_sigma))}
        else:
            o =  {'proprio': self.get_tensor(self.Q[n])}
        
        if self.arglist.image:
            o['image'] = self.get_tensor(self.I[n]/np.float32(255.0))
        if self.arglist.text:
            o['text'] = self.get_tensor(self.T[n])
        
        if self.arglist.normalize:
            a = self.get_tensor(normalize(self.A[n], self.A_mu, self.A_sigma))
        else:
            a = self.get_tensor(self.A[n])

        return o, a

def check(data_loader, model):
    O, A = next(iter(data_loader))
    for key in O:
        print(key, O[key].size())
    print("action", A.size())
    loss = model.loss(O, A)
    print("loss", loss)
    a = model.sample(O)
    print("sampled action", a.size())
    sys.exit(0)

def parse_args():
    parser = argparse.ArgumentParser("Flow matching")
    # Settings
    parser.add_argument("--env", type=str, default="pick-place-v3", help="")
    parser.add_argument("--image", action="store_true", default=False)
    parser.add_argument("--text", action="store_true", default=False)
    parser.add_argument("--expt", type=str, default="expt_proprio_128_3", help="expt name")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--T-flow", type=int, default=20, help="flow time steps for sampling")
    parser.add_argument("--d-model", type=int, default=128, help="hidden size dim")
    parser.add_argument("--d-emb", type=int, default=32, help="hidden size dim")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=250, help="number of epochs to train")
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

    if arglist.image:
        render_mode = 'rgb_array'
    else:
        render_mode = 'none'
    print(arglist.env)
    env = gym.make('Meta-World/MT1', env_name=arglist.env, seed=arglist.seed, render_mode=render_mode)
    arglist.d_proprio = 39 # proprio size
    arglist.d_act = 4 # action size
    model = FlowModel(arglist).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    train_data = Dataset(os.path.join(flow_matching_dir, "train.npz"), arglist)
    test_data = Dataset(os.path.join(flow_matching_dir, "test.npz"), arglist)
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
            
            # Evaluate agent performance over several episodes
            metric = []
            for episode in range(5):
                o, info = env.reset()
                while True:
                    if arglist.normalize:
                        O = {'proprio': torch.tensor(normalize(o, train_data.Q_mu, train_data.Q_sigma),
                             dtype=torch.float32, device=device).unsqueeze(0)}
                    else:
                        O = {'proprio': torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)}

                    if arglist.image:
                        O['image'] = torch.tensor(o['image']/np.float32(255.0), dtype=torch.float32, device=device).unsqueeze(0)

                    with torch.no_grad():
                        A = model.sample(O)
                    a = A.cpu().numpy()[0]
                    if arglist.normalize:
                        a = a * train_data.A_sigma + train_data.A_mu
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
