import torchvision
from torch.utils.data import DataLoader

""" Run this to calculate and display mean and std of MNIST dataset.
    Used to normalize dataset in run.py based on calculated values.
""" 

def load_data(train):
    return DataLoader(
        torchvision.datasets.MNIST('/files/', train=train, download=True,
                               transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()])),
        batch_size=100, shuffle=True)

def get_stats(loader):
    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    return mean, std

train_loader = load_data(train=True)
mean, std = get_stats(train_loader)

print('Training data:')
print('    mean: %.4f' % mean)
print('    std: %.4f' % std)

test_loader = load_data(train=False)
mean, std = get_stats(test_loader)

print('Test data:')
print('    mean: %.4f' % mean)
print('    std: %.4f' % std)