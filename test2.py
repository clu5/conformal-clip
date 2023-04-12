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
import os
import open_clip
from tqdm import tqdm
from PIL import Image
from conformal import *
from stablediffusion import *
from dataset import DS, DatasetC, load_txt
# import wilds

    
def add_gaussian_noise(img, sigma):
    # Gaussian distribution parameters
    img = np.array(img)/255.0
    row, col, _ = img.shape
    noise =  np.random.normal(loc=0, scale=1, size=img.shape)
    noise_img = np.clip((img + noise*sigma),0,1)
    noise_img = Image.fromarray(np.uint8(noise_img*255))
    return noise_img

def zero_shot_clip(use_stable=True):
    cifar_true_class = []
    cifar_pred_class = []
    cifar_pred_scores = []

    with torch.no_grad(), torch.cuda.amp.autocast():
        prompt = tokenizer(['This is an image of a ' + c for c in cifar_classes])
        text_features = model.encode_text(prompt)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        for i in tqdm(range(len(cifar_test_dataset))):
            x, y = cifar_test_dataset[i]
            if use_stable:
                x_new = stablediffusion(
                    'This is an image of a ' + cifar_classes[y],  
                    'This is a watercolor painting of a ' + cifar_classes[y], 
                    seed=115, 
                    init_image=x,
                    init_image_strength=0.7
                  )
                x = x_new
            # noise_x = add_gaussian_noise(x, sigma)
            image_features = model.encode_image(preprocess(x).unsqueeze(0))
            image_features /= image_features.norm(dim=-1, keepdim=True)
            scores = (image_features @ text_features.T).softmax(dim=-1)

            cifar_true_class.append(y)
            cifar_pred_class.append(scores.argmax().item())
            cifar_pred_scores.append(scores.detach().cpu().numpy().squeeze())

    cifar_true_class = np.asarray(cifar_true_class)
    cifar_pred_class = np.asarray(cifar_pred_class)
    cifar_pred_scores = np.asarray(cifar_pred_scores)

    cifar_acc = (cifar_true_class == cifar_pred_class).sum() / cifar_true_class.shape[0]
    print(f'CIFAR100 accuracy: {cifar_acc:.1%}')
    return cifar_true_class, cifar_pred_scores, cifar_acc

frac = 0.1
alpha = 0.1

def conformal_clip(cifar_true_class, cifar_pred_scores):
    cifar_n = int(round(frac * len(cifar_pred_scores)))
    cifar_cal_scores = torch.tensor(cifar_pred_scores[:cifar_n])
    cifar_cal_targets = torch.tensor(cifar_true_class[:cifar_n])
    cifar_val_scores = torch.tensor(cifar_pred_scores[cifar_n:])
    cifar_val_targets = torch.tensor(cifar_true_class[cifar_n:])

    cifar_qhat = get_quantile(cifar_cal_scores, cifar_cal_targets, alpha=alpha)
    cifar_psets = make_prediction_sets(cifar_val_scores, cifar_qhat)
    cifar_psets_size = cifar_psets.sum(1)

    print(f'CIFAR100 coverage: {get_coverage(cifar_psets, cifar_val_targets):.1%}')
    print(f'CIFAR100 set size: {get_size(cifar_psets):.1f}')
    return cifar_psets_size

    
# CORRUPTIONS = load_txt('corruptions.txt')
CORRUPTIONS =['natural']
data_dir = Path('../data')
cifar_train = datasets.CIFAR100(root=data_dir, download = True, train=True)
cifar_test = datasets.CIFAR100(root=data_dir, download = True, train=False)
cifar_classes = tuple(cifar_test.classes)
cifar_class_map = dict(map(reversed, cifar_test.class_to_idx.items()))
transform = transforms.Compose([
                    # transforms.ToTensor(),
                    # transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
                    transforms.ToPILImage(),
                ])
accs = []
psets_sizes = []
str_labels = []        
for ci, cname in enumerate(CORRUPTIONS):
        # load dataset
        if cname == 'natural':
            cifar_test_dataset = DS(
                cifar_test.data, cifar_test.targets,
                transforms=transform
            )
            loader_params = dict(batch_size=16, shuffle=False, pin_memory=True, num_workers=8)
            # cifar_test_loader = DataLoader(cifar_test_dataset, **loader_params)

            tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')

            print('natural images')
            cifar_true_class, cifar_pred_scores, cifar_acc = zero_shot_clip()
            cifar_psets_size = conformal_clip(cifar_true_class, cifar_pred_scores)
            accs.append(cifar_acc)
            psets_sizes.append(cifar_psets_size)
            str_labels.append('natural')
            
            
        else:
            for noise_level in range(1, 6):
                cifar_test_dataset = DatasetC(
                    os.path.join('../data', 'CIFAR-100-C'),
                    cname, noise_level, transforms=transform
                )
                
                loader_params = dict(batch_size=16, shuffle=False, pin_memory=True, num_workers=8)
                # cifar_test_loader = DataLoader(cifar_test_dataset, **loader_params)

                tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')
                model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')

                print('corrupion with ', cname, ' with level ', noise_level)
                cifar_true_class, cifar_pred_scores, cifar_acc = zero_shot_clip()
                cifar_psets_size = conformal_clip(cifar_true_class, cifar_pred_scores)
                accs.append(cifar_acc)
                psets_sizes.append(cifar_psets_size)
                str_labels.append(cname + ' level ' + str(noise_level))
        
    
fontsize=15
# num = (len(CORRUPTIONS)-1)*5 + 1
num = 1
# plt.figure(figsize=(15, 8))
plt.violinplot(psets_sizes, vert=False, widths=1.0)
plt.xlabel('Prediction set size', fontsize=fontsize)
plt.xticks(fontsize=fontsize-4)
plt.yticks(ticks=np.arange(1, num+1), labels=str_labels, fontsize=fontsize-6)
plt.savefig('pset_diff.png', bbox_inches = 'tight')