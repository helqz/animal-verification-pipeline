import sys
import os
sys.path.append(os.path.abspath(".."))

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from data.animals.translate import translate

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),    # Resize images to 128x128
    transforms.ToTensor(),            # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize
    transforms.RandomHorizontalFlip(),  # Augmentation: Random Flip
])

# Function to load dataset and DataLoader
def get_dataloader(data_dir, batch_size=32, shuffle=True):
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataset.classes = [translate[x] for x in dataset.classes]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset, dataloader
