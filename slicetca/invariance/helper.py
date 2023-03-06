from slicetca.core.decompositions import SliceTCA

import torch
from typing import Sequence

def mm(a: torch.Tensor, b: torch.Tensor):
    """
    Performs generalized matrix multiplication (ijq) x (qkl) -> (ijkl)
    :param a: torch
    :param b:
    :return:
    """
    temp1 = [chr(105+i) for i in range(len(a.size()))]
    temp2 = [chr(105+len(a.size())-1+i) for i in range(len(b.size()))]
    indexes1 = ''.join(temp1)
    indexes2 = ''.join(temp2)
    rhs = ''.join(temp1[:-1])+''.join(temp2[1:])
    formula = indexes1+','+indexes2+'->'+rhs
    return torch.einsum(formula,[a,b])


def batch_outer(a: torch.Tensor, b: torch.Tensor):
    temp1 = [chr(105 + i + 1) for i in range(len(a.size()) - 1)]
    temp2 = [chr(105 + len(a.size()) + i + 1) for i in range(len(b.size()) - 1)]
    indexes1 = ''.join(temp1)
    indexes2 = ''.join(temp2)
    formula = chr(105) + indexes1 + ',' + chr(105) + indexes2 + '->' + chr(105) + indexes1 + indexes2
    return torch.einsum(formula, [a, b])


def construct_per_type(model: SliceTCA, components: Sequence[Sequence[torch.Tensor]]):
    """
    :param model: SliceTCA model.
    :param components: The components to construct.
    :return: Reconstructed tensor.
    """

    temp = [torch.zeros(model.dimensions).to(model.device) for i in range(len(components))]

    for i in range(len(components)):
        for j in range(model.ranks[i]):
            temp[i] += construct_single_component(model, components, i, j)
    return temp


def construct_single_component(model: SliceTCA, components: Sequence[Sequence[torch.Tensor]], partition: int, k: int):

    temp2 = [model.positive_function[partition][q](components[partition][q][k]) for q in range(len(components[partition]))]
    outer = torch.einsum(model.einsums[partition], temp2)
    outer = outer.permute(model.inverse_permutations[partition])

    return outer
