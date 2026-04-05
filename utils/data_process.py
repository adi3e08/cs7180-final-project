from torchvision import datasets, transforms
from torch.utils.data import DataLoader


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

