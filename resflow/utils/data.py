import torch
import torchvision
import torchvision.transforms as transforms

def get_mnist_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.Pad(2), # 28x28 -> 32x32 to make UNet down/upsampling easier
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # scale to [-1, 1]
    ])
    
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader
