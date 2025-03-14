import sys
import os
sys.path.append(os.path.abspath(".."))

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from translate import translate_word

def get_image_transforms():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.RandomHorizontalFlip(),
    ])

def load_image_dataset(data_dir, batch_size=32, shuffle=True):
    transforms = get_image_transforms()
    dataset = datasets.ImageFolder(root=data_dir, transform=transforms)
    
    # Translate class names to English
    dataset.classes = [translate_word(cls) for cls in dataset.classes]
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset, dataloader  