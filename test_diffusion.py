from pathlib import Path
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt; plt.style.use('bmh')
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
assert torch.cuda.is_available()
import open_clip
from tqdm import tqdm
from dataset import DS, DatasetC
from conformal import get_pset_size, plot_violin
from zsclip import zero_shot_clip
from stablediffusion import stablediffusion
# import wilds

tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')

data_dir = Path('../data')
# cifar_train = datasets.CIFAR100(root=data_dir, download = True, train=True)
cifar_test = datasets.CIFAR100(root=data_dir, download = True, train=False)
cifar_classes = tuple(cifar_test.classes)
cifar_class_map = dict(map(reversed, cifar_test.class_to_idx.items()))
cifar_test_dataset = DS(
    cifar_test.data, cifar_test.targets,
    transforms=transforms.Compose([
        # transforms.ToTensor(),
        # transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        transforms.ToPILImage(),
    ]), 
)

images = []
labels = []
for i in tqdm(range(len(cifar_test_dataset))):
    x, y = cifar_test_dataset[i]
    x_new = stablediffusion(
        'This is an image of a ' + cifar_classes[y],  
        'This is a watercolor painting of a ' + cifar_classes[y], 
        seed=115, 
        init_image=x,
        init_image_strength=0.7
      )
    images.append(preprocess(x_new))
    labels.append(y)
images = torch.stack(images)    
prompts = ['This is an image of a ' + c for c in cifar_classes]
prompts_new = ['This is a watercolor painting of a ' + c for c in cifar_classes]

print('Conformal analysis for CIFAR100 dataset')

accs = []
psets_sizes = []
str_labels = []        

print('Original Prompts: This is an image of a ...')
cifar_true_class, cifar_pred_scores, cifar_acc = zero_shot_clip(model, tokenizer, images, labels, prompts)
cifar_psets_size = get_pset_size(cifar_true_class, cifar_pred_scores)
accs.append(cifar_acc)
psets_sizes.append(cifar_psets_size)
str_labels.append('original prompts')

print('New Prompts: This is a watercolor painting of a ...')
cifar_true_class, cifar_pred_scores, cifar_acc = zero_shot_clip(model, tokenizer, images, labels, prompts_new)
cifar_psets_size = get_pset_size(cifar_true_class, cifar_pred_scores)
accs.append(cifar_acc)
psets_sizes.append(cifar_psets_size)
str_labels.append('new prompts')
        
plot_violen(num=12, psets_sizes=psets_sizes, str_labels=str_labels, figname='pset_watercolor_painting.png')
