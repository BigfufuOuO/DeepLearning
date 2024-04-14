import torch.utils.data
import torchvision
import sys

# Define the transformation: Convert the image to a tensor and normalize the pixel values
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def load_data(batch_size=64):
    # divide the data into train and validation set
    
    raw_train_data = torchvision.datasets.CIFAR10(root=f'ConvolutionalNN\data', train=True, download=False, transform=transform)
    raw_test_data = torchvision.datasets.CIFAR10(root=f'ConvolutionalNN\data', train=False, download=False, transform=transform)
    
    # Randomly split the data into train and validation set
    generator1 = torch.Generator().manual_seed(42)
    train_data, val_data = torch.utils.data.random_split(raw_train_data, [40000, 10000], generator=generator1)
    
    # Load the data into the data loader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(raw_test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


    
    
        