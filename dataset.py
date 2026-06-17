from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import image_size, batch_size, dataset_path

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def get_dataloader():
    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)