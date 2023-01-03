from itertools import combinations
import torch


def orthogonality_component_type_wise(reconstructed_tensors_of_each_type):
    l = 0
    for combo in combinations(reconstructed_tensors_of_each_type, 2):
        l += torch.square(torch.sum(combo[0] * combo[1]) / torch.sqrt(torch.sum(combo[0] ** 2)) / torch.sqrt(
            torch.sum(combo[1] ** 2)))
    return l + l2(reconstructed_tensors_of_each_type)


def l2(reconstructed_tensors_of_each_type):
    l = 0
    for t in reconstructed_tensors_of_each_type:
        l += (t ** 2).mean()
    return l
