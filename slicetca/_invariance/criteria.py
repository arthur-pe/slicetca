from itertools import combinations
import torch
from typing import Sequence

# Example of criteria to use for L2 optimization.


def orthogonality_component_type_wise(reconstructed_tensors_of_each_partition: Sequence[torch.Tensor]):
    """
    Penalizes non-orthogonality between the reconstructed tensors of each partition/slicing.

    :param reconstructed_tensors_of_each_partition: The sum of the terms of a given partition/slicing.
    :return: Torch float.
    """

    l = 0
    for combo in combinations(reconstructed_tensors_of_each_partition, 2):
        l += torch.square(torch.sum(combo[0] * combo[1]) / torch.sqrt(torch.sum(combo[0] ** 2)) / torch.sqrt(
            torch.sum(combo[1] ** 2)))
    return l + l2(reconstructed_tensors_of_each_partition)


def l2(reconstructed_tensors_of_each_partition: Sequence[torch.Tensor]):
    """
    Classic L_2 regularization, per reconstructed tensors of each partition/slicing.

    :param reconstructed_tensors_of_each_partition: The sum of the terms of a given partition/slicing.
    :return: Torch float.
    """

    l = 0
    for t in reconstructed_tensors_of_each_partition:
        l += (t ** 2).mean()
    return l
