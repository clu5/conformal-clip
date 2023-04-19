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
import os
from dataset import DS, DatasetC, load_txt
from conformal import get_pset_size, plot_violin
from zsclip import zero_shot_clip
# import wilds

    
def add_gaussian_noise(img, sigma):
    # Gaussian distribution parameters
    img = np.array(img)/255.0
    row, col, _ = img.shape
    noise =  np.random.normal(loc=0, scale=1, size=img.shape)
    noise_img = np.clip((img + noise*sigma),0,1)
    noise_img = Image.fromarray(np.uint8(noise_img*255))
    return noise_img

tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
    
    
CORRUPTIONS = load_txt('noise-C/corruptions.txt')
data_dir = Path('../data')
cifar_train = datasets.CIFAR100(root=data_dir, download = True, train=True)
cifar_test = datasets.CIFAR100(root=data_dir, download = True, train=False)
cifar_classes = tuple(cifar_test.classes)
cifar_class_map = dict(map(reversed, cifar_test.class_to_idx.items()))
transform = transforms.Compose([
                    # transforms.ToTensor(),
                    # transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
                    transforms.ToPILImage(),
                    preprocess
                ])
prompts = ['This is an image of a ' + c for c in cifar_classes]
       
for ci, cname in enumerate(CORRUPTIONS):
        # load dataset
        if cname == 'natural':
            cifar_test_dataset = DS(
                cifar_test.data, cifar_test.targets,
                transforms=transform
            )
            images = [cifar_test_dataset[i][0] for i in range(len(cifar_test_dataset))]
            images = torch.stack(images)
            labels = [cifar_test_dataset[i][1] for i in range(len(cifar_test_dataset))]

            print('natural images')
            cifar_true_class, cifar_pred_scores, cifar_acc = zero_shot_clip(model, tokenizer, images, labels, prompts)
            cifar_psets_size = get_pset_size(cifar_true_class, cifar_pred_scores)
            cifar_acc_natural = cifar_acc
            cifar_psets_size_natural = cifar_psets_size
            
            
        else:
            accs = [cifar_acc_natural]
            psets_sizes = [cifar_psets_size_natural]
            str_labels = ['natural']
            
            for noise_level in range(1, 6):
                cifar_test_dataset = DatasetC(
                    os.path.join('../data', 'CIFAR-100-C'),
                    cname, noise_level, transforms=transform
                )
                images = [cifar_test_dataset[i][0] for i in range(len(cifar_test_dataset))]
                images = torch.stack(images)
                labels = [cifar_test_dataset[i][1] for i in range(len(cifar_test_dataset))]
                
                print('corrupion with ', cname, ' with level ', noise_level)
                cifar_true_class, cifar_pred_scores, cifar_acc = zero_shot_clip(model, tokenizer, images, labels, prompts)
                cifar_psets_size = get_pset_size(cifar_true_class, cifar_pred_scores)
                accs.append(cifar_acc)
                psets_sizes.append(cifar_psets_size)
                str_labels.append(cname + ' level ' + str(noise_level))
        
            plot_violin(num=6, psets_sizes=psets_sizes, str_labels=str_labels, figname='noise-C/pset_noise_' + cname + '.png')