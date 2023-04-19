import numpy as np
import os
import PIL
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, Subset

def load_txt(path :str) -> list:
    return [line.rstrip('\n') for line in open(path)]

corruptions = load_txt('noise-C/corruptions.txt')

class DS(Dataset):
    def __init__(self, images, labels, transforms=None, target_transforms=None):
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels
        self.transforms = transforms
        self.target_transforms = target_transforms
        
    def __getitem__(self, k):
        x = self.images[k]
        y = self.labels[k]
        if self.transforms is not None:
            x = self.transforms(x)
        if self.target_transforms is not None:
            y = self.target_transforms(y)
        return x, y
    
    def __len__(self):
        return len(self.images)

class DatasetC(Dataset):
    def __init__(self, root :str, name :str, noise_level, transforms=None, target_transforms=None, ):
        assert name in corruptions
        data_path = os.path.join(root, name + '.npy')
        target_path = os.path.join(root, 'labels.npy')
        self.images = np.load(data_path)[(noise_level-1)*10000:noise_level*10000]
        self.labels = np.load(target_path)[(noise_level-1)*10000:noise_level*10000]
        self.transforms = transforms
        self.target_transforms = target_transforms
        
    def __getitem__(self, k):
        x = self.images[k]
        y = self.labels[k]
        if self.transforms is not None:
            x = self.transforms(x)
        if self.target_transforms is not None:
            y = self.target_transforms(y)
        return x, y.item()
    
    def __len__(self):
        return len(self.images)


def extract_subset(dataset, num_subset :int, random_subset :bool):
    if random_subset:
        random.seed(0)
        indices = random.sample(list(range(len(dataset))), num_subset)
    else:
        indices = [i for i in range(num_subset)]
    return Subset(dataset, indices)