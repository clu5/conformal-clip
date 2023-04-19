import torch
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('bmh')

def get_quantile(scores, targets, alpha=0.1):
    n = torch.tensor(targets.size(0))
    score_dist = torch.take_along_dim(1 - scores, targets.unsqueeze(1), 1).flatten()
    qhat = torch.quantile(score_dist, torch.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher")
    return qhat


def make_prediction_sets(scores, qhat, allow_empty_sets=False):
    n = scores.size(0)
    elements_mask = scores >= (1 - qhat)
    if not allow_empty_sets:
        elements_mask[torch.arange(n), scores.argmax(1)] = True
    return elements_mask

def get_coverage(psets, targets, precision=None):
    psets = psets.clone().detach()
    targets = targets.clone().detach()
    n = psets.shape[0]
    coverage = psets[torch.arange(n), targets].float().mean().item()
    if precision is not None:
        coverage = round(coverage, precision)
    return coverage


def get_size(psets, precision=1):
    psets = psets.clone().detach()
    size = psets.sum(1).float().mean().item()
    if precision is not None:
        size = round(size, precision)
    return size


def get_coverage_by_class(psets, targets, num_classes):
    psets = psets.clone().detach()
    targets = targets.clone().detach()
    results = {}
    for c in range(num_classes):
        index = targets == c
        psets_c = psets[index]
        targets_c = targets[index]
        results[c] = get_coverage(psets_c, targets_c)
    return results


def get_efficiency_by_class(psets, targets, num_classes):
    psets = psets.clone().detach()
    targets = targets.clone().detach()
    sizes = psets.sum(1)
    results = {}
    for c in range(num_classes):
        index = targets == c
        psets_c = psets[index]
        results[c] = get_size(psets_c)
    return results

def get_pset_size(true_class, pred_scores, frac = 0.1, alpha = 0.1):
    n = int(round(frac * len(pred_scores)))
    cal_scores = torch.tensor(pred_scores[:n])
    cal_targets = torch.tensor(true_class[:n])
    val_scores = torch.tensor(pred_scores[n:])
    val_targets = torch.tensor(true_class[n:])

    qhat = get_quantile(cal_scores, cal_targets, alpha=alpha)
    psets = make_prediction_sets(val_scores, qhat)
    psets_size = psets.sum(1)

    print(f'Coverage: {get_coverage(psets, val_targets):.1%}')
    print(f'Set size: {get_size(psets):.1f}')
    return psets_size

def plot_violin(num, psets_sizes, str_labels, figname):
    fontsize=15
    plt.figure()
    # plt.figure(figsize=(15, 8))
    plt.violinplot(psets_sizes, vert=False, widths=1.0)
    plt.xlabel('Prediction set size', fontsize=fontsize)
    plt.xticks(fontsize=fontsize-4)
    plt.yticks(ticks=np.arange(1, num+1), labels=str_labels, fontsize=fontsize-6)
    plt.savefig(figname, bbox_inches = 'tight')