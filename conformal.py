import torch
import numpy as np


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