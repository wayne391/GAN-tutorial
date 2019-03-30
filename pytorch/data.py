import torchvision.transforms as transforms
from torchvision import datasets


def mnist(data_dir):
    return datasets.MNIST(
        data_dir,
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        )