import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, BatchSampler


class ClipDataset(Dataset):
    def __init__(
        self, images, targets, 
        class_map,
        transforms=None, target_transforms=None,
        k = None,
    ):
        self.images = images
        self.targets = targets
        self.class_map = class_map
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.k = k
        
    def __len__(self):
        assert len(self.images) == len(self.targets)
        if self.k is not None:
            return self.k
        return len(self.images)
    
    
    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.targets[idx]
        if self.transforms is not None:
            x = self.transforms(x)
        if self.target_transforms is not None:
            y = self.target_transforms(y)
        text = f'a {self.class_map[y]}'
        return x, y, text
    
    
class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, targets, n_classes, n_samples=1):
        self.targets = np.asarray(targets)
        self.targets_set = list(set(self.targets))
        self.target_to_indices = {t: np.where(self.targets == t)[0] for t in self.targets_set}
        for t in self.targets_set:
            np.random.shuffle(self.target_to_indices[t])
        self.used_target_indices_count = {target: 0 for target in self.targets_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.targets)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            # print(self.count + self.batch_size)
            classes = np.random.choice(self.targets_set, self.n_classes, replace=False)
            indices = []
            for c in classes:
                indices_t = self.target_to_indices[c]
                count_t = self.used_target_indices_count[c]
                indices.extend(indices_t[count_t:count_t + self.n_samples])
                count_t += self.n_samples
                if count_t + self.n_samples > len(indices_t):
                    np.random.shuffle(indices_t)
                    count_t = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
