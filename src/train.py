import warnings
warnings.filterwarnings("ignore")
import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from src.model import FlowMatchingModel
from src.utils import normalize, get_tensor, add_expt_config

def parse_args():
    parser = argparse.ArgumentParser("Flow matching")
    parser.add_argument("--expt", type=str, default="expt_4", help="expt name")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    arglist = parser.parse_args()
    arglist = add_expt_config(arglist)
    return arglist

class Dataset(torch.utils.data.Dataset):
    """
    dataset['proprio']: N, d_obs
    dataset['rgb']: N, 3, 240, 240
    dataset['depth']: N, 1, 240, 240
    dataset['text']: N
    dataset['action']: N, d_act
    """
    def __init__(self, arglist, mode):
        super().__init__()
        data_dir = os.path.join("./data/raw", arglist.expt)
        dataset = np.load(os.path.join(data_dir, mode+".npz"), allow_pickle=True)
        stats = np.load(os.path.join(data_dir, "stats.npz"), allow_pickle=True)
        self.arglist = arglist
        self.proprio = dataset['proprio']
        if self.arglist.image:
            self.rgb = dataset['rgb']
            self.depth = dataset['depth']
        if self.arglist.text:
            self.text = np.expand_dims(dataset['text'],axis=1)
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
            o['text'] = get_tensor(self.text[n],dtype=torch.long)
        
        return o, a

def check(data_loader, model):
    O, A = next(iter(data_loader))
    for key in O:
        print(key, O[key].size())
    print("action", A.size())
    loss = model.loss(O, A)
    print("loss", loss)
    sys.exit(0)

def main():
    arglist = parse_args()

    np.random.seed(arglist.seed)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(arglist.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = os.path.join("./models", arglist.expt)
    results_dir = os.path.join("./results", arglist.expt)
    os.mkdir(model_dir)
    os.mkdir(results_dir)
    writer = SummaryWriter(log_dir=results_dir)
    
    model = FlowMatchingModel(arglist).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    train_data = Dataset(arglist, "train")
    test_data = Dataset(arglist, "test")

    if torch.cuda.is_available():
        num_workers=1
        pin_memory=True
    else:
        num_workers=0
        pin_memory=False
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=arglist.batch_size, shuffle=True, 
                                               num_workers=num_workers, pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=arglist.batch_size, shuffle=True, 
                                              num_workers=num_workers, pin_memory=pin_memory)

    # check(test_loader, model)

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

    writer.close()

if __name__ == '__main__':
    main()
